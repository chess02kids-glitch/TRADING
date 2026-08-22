"""Tests for execution.order_manager (sizing, guards, execution)."""
from __future__ import annotations

import pytest

from execution.order_manager import (
    MAX_POSITION_FRACTION,
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

    def test_capped_at_10_percent(self):
        # har_vol = 200/20000 = 0.01 -> size_usd = 10000*0.01/0.01 = 10000
        # capped at 10000*0.10 = 1000
        sig = _signal(har_range=200.0)
        assert compute_position_size(sig, 10000.0, 20000.0) == pytest.approx(1000.0)

    def test_under_cap_uses_full(self):
        # har_vol = 100/50000 = 0.002 -> size_usd = 10000*0.01/0.002 = 50000
        # capped at 1000
        sig = _signal(har_range=100.0)
        assert compute_position_size(sig, 10000.0, 50000.0) == pytest.approx(1000.0)

    def test_zero_har_vol_returns_zero(self):
        assert compute_position_size(_signal(har_range=0.0), 10000.0, 20000.0) == 0.0

    def test_below_minimum_returns_zero(self):
        # account 50 -> max_size 5 -> below $10 minimum
        assert compute_position_size(_signal(har_range=200.0), 50.0, 20000.0) == 0.0

    def test_zero_price_returns_zero(self):
        assert compute_position_size(_signal(), 10000.0, 0.0) == 0.0


class TestBuildOrderParams:

    def test_builds_buy(self):
        params = build_order_params(_signal(direction=1, har_range=200.0),
                                    10000.0, 20000.0)
        assert isinstance(params, OrderParams)
        assert params.side == "buy"
        assert params.size_usd == pytest.approx(1000.0)
        assert params.size_base == pytest.approx(1000.0 / 20000.0)
        assert params.symbol == "BTC/USDT"
        assert params.har_vol_estimate == pytest.approx(0.01)

    def test_builds_sell(self):
        params = build_order_params(_signal(direction=-1), 10000.0, 20000.0)
        assert params.side == "sell"

    def test_none_when_size_zero(self):
        # account 50 -> below minimum -> size 0 -> None
        assert build_order_params(_signal(), 50.0, 20000.0) is None

    def test_none_when_direction_zero(self):
        assert build_order_params(_signal(direction=0), 10000.0, 20000.0) is None


class TestExecuteSignal:

    class FakeClient:
        def __init__(self, ok=True):
            self.ok = ok
            self.placed = []

        def place_market_order(self, symbol, side, amount):
            if not self.ok:
                return None
            o = {"id": "x1", "symbol": symbol, "side": side, "amount": amount}
            self.placed.append(o)
            return o

    def test_executes_and_returns_order(self):
        sig = _signal(direction=1, har_range=200.0)
        client = self.FakeClient()
        order = execute_signal(sig, client, 10000.0, 20000.0)
        assert order is not None
        assert order["side"] == "buy"
        assert client.placed[0]["amount"] == pytest.approx(1000.0 / 20000.0)

    def test_returns_none_when_size_zero(self):
        sig = _signal(direction=1, har_range=200.0)
        client = self.FakeClient()
        assert execute_signal(sig, client, 50.0, 20000.0) is None
        assert client.placed == []

    def test_returns_none_when_direction_zero(self):
        sig = _signal(direction=0)
        client = self.FakeClient()
        assert execute_signal(sig, client, 10000.0, 20000.0) is None

    def test_returns_none_when_rejected(self):
        sig = _signal(direction=1, har_range=200.0)
        client = self.FakeClient(ok=False)
        assert execute_signal(sig, client, 10000.0, 20000.0) is None
