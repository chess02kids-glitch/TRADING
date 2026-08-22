"""Unit tests for execution.order_manager (sizing, guards, execution)."""
from __future__ import annotations

import pytest

from execution.order_manager import (
    MAX_POSITION_FRACTION,
    MIN_NOTIONAL_USD,
    OrderParams,
    SignalInput,
    build_order_params,
    compute_position_size,
    execute_signal,
)


def _signal(direction=1, har_range=200.0, asset="BTC/USDT"):
    return SignalInput(
        timestamp="2024-01-15T00:00:00Z", asset=asset, direction=direction,
        har_predicted_range=har_range, confidence=0.6, regime="high",
    )


class TestComputePositionSize:

    def test_basic_sizing_and_cap(self):
        # account=10000, target_vol=0.01, price=20000, har_range=200 -> har_vol=0.01
        # notional = 10000*0.01/0.01 = 10000; max = 10000*0.10 = 1000 -> capped 1000
        # base_size = 1000/20000 = 0.05
        sig = _signal(har_range=200.0)
        size = compute_position_size(sig, account_size=10000.0, current_price=20000.0)
        assert size == pytest.approx(0.05)

    def test_under_cap_uses_full_vol_size(self):
        # har_vol = 100/50000 = 0.002 -> notional = 10000*0.01/0.002 = 50000
        # capped at 1000 -> base = 1000/50000 = 0.02
        sig = _signal(har_range=100.0)
        size = compute_position_size(sig, 10000.0, 50000.0)
        assert size == pytest.approx(0.02)

    def test_direction_zero_returns_zero(self):
        sig = _signal(direction=0)
        assert compute_position_size(sig, 10000.0, 20000.0) == 0.0

    def test_zero_har_range_returns_zero(self):
        sig = _signal(har_range=0.0)
        assert compute_position_size(sig, 10000.0, 20000.0) == 0.0

    def test_zero_price_returns_zero(self):
        sig = _signal()
        assert compute_position_size(sig, 10000.0, 0.0) == 0.0


class TestBuildOrderParams:

    def test_builds_buy_order(self):
        sig = _signal(direction=1, har_range=200.0)
        params = build_order_params(sig, 10000.0, 20000.0)
        assert isinstance(params, OrderParams)
        assert params.side == "buy"
        assert params.symbol == "BTC/USDT"
        assert params.size == pytest.approx(0.05)
        assert params.direction == 1
        assert params.har_predicted_range == 200.0
        assert params.regime == "high"

    def test_builds_sell_order(self):
        sig = _signal(direction=-1)
        params = build_order_params(sig, 10000.0, 20000.0)
        assert params.side == "sell"
        assert params.direction == -1

    def test_skip_direction_zero(self):
        assert build_order_params(_signal(direction=0), 10000.0, 20000.0) is None

    def test_skip_low_har_vol(self):
        assert build_order_params(_signal(har_range=0.0), 10000.0, 20000.0) is None

    def test_skip_below_min_notional(self):
        # Tiny account + large cap -> notional under $10.
        # account=50 -> max_notional=5 -> below MIN_NOTIONAL_USD(10).
        sig = _signal(har_range=200.0)
        params = build_order_params(sig, account_size=50.0, current_price=20000.0)
        assert params is None

    def test_cap_is_ten_percent(self):
        sig = _signal(har_range=200.0)
        params = build_order_params(sig, 10000.0, 20000.0)
        assert params.size * 20000.0 <= 10000.0 * MAX_POSITION_FRACTION + 1e-9


class TestExecuteSignal:

    class FakeClient:
        def __init__(self, price=20000.0, order_ok=True):
            self._price = price
            self._order_ok = order_ok
            self.placed = []

        def get_ticker(self, symbol):
            return {"last": self._price}

        def place_market_order(self, symbol, side, amount):
            if not self._order_ok:
                return None
            order = {"id": "x1", "symbol": symbol, "side": side, "amount": amount}
            self.placed.append(order)
            return order

    class FakeLogger:
        def __init__(self):
            self.events = []

        def log_order_attempt(self, signal, params):
            self.events.append(("attempt", params.symbol, params.side))

        def log_order_result(self, result):
            self.events.append(("result", result.get("status")))

        def log_skip(self, reason):
            self.events.append(("skip", reason))

    def test_executes_buy_and_logs(self):
        sig = _signal(direction=1, har_range=200.0)
        client = self.FakeClient(price=20000.0)
        log = self.FakeLogger()
        order = execute_signal(sig, client, 10000.0, execution_logger=log)
        assert order is not None
        assert order["side"] == "buy"
        assert client.placed[0]["amount"] == pytest.approx(0.05)
        assert ("attempt", "BTC/USDT", "buy") in log.events
        assert any(e[0] == "result" for e in log.events)

    def test_skip_direction_zero(self):
        sig = _signal(direction=0)
        client = self.FakeClient()
        log = self.FakeLogger()
        assert execute_signal(sig, client, 10000.0, execution_logger=log) is None
        assert any(e[0] == "skip" for e in log.events)
        assert client.placed == []

    def test_skip_no_price(self):
        class NoTicker:
            def get_ticker(self, symbol):
                return None
            def place_market_order(self, *a):
                raise AssertionError("should not place")
        sig = _signal(direction=1)
        assert execute_signal(sig, NoTicker(), 10000.0) is None

    def test_skip_below_minimum(self):
        sig = _signal(har_range=200.0)
        client = self.FakeClient(price=20000.0)
        log = self.FakeLogger()
        # Tiny account -> notional capped below $10.
        assert execute_signal(sig, client, 50.0, execution_logger=log) is None
        assert client.placed == []
        assert any(e[0] == "skip" for e in log.events)

    def test_order_rejected_returns_none(self):
        sig = _signal(direction=1, har_range=200.0)
        client = self.FakeClient(price=20000.0, order_ok=False)
        log = self.FakeLogger()
        assert execute_signal(sig, client, 10000.0, execution_logger=log) is None
        assert any(e[0] == "result" for e in log.events)  # rejection logged
