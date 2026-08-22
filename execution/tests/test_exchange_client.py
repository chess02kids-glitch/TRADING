"""Unit tests for execution.exchange_client (sandbox-enforced, fail-soft)."""
from __future__ import annotations

import pytest

from execution.exchange_client import (
    ExchangeClient,
    ExchangeConfig,
    SandboxViolation,
)


def _valid_config(**overrides):
    base = dict(api_key="k", api_secret="s", api_password="p", sandbox=True)
    base.update(overrides)
    return ExchangeConfig(**base)


class TestSandboxEnforcement:

    def test_sandbox_false_raises(self):
        with pytest.raises(SandboxViolation):
            ExchangeClient(_valid_config(sandbox=False))

    def test_sandbox_false_raises_value_error(self):
        # SandboxViolation is a ValueError subclass (per spec).
        with pytest.raises(ValueError):
            ExchangeClient(_valid_config(sandbox=False))

    def test_default_sandbox_true(self):
        cfg = ExchangeConfig(api_key="k", api_secret="s", api_password="p")
        assert cfg.sandbox is True
        client = ExchangeClient(cfg)
        assert client.config.sandbox is True


class TestConnect:

    def test_connect_success(self, monkeypatch):
        class FakeExchange:
            def __init__(self, *a, **kw):
                pass
            def set_sandbox_mode(self, v):
                assert v is True
            def load_markets(self):
                pass

        monkeypatch.setattr(
            "execution.exchange_client._build_exchange",
            lambda config: FakeExchange(),
        )
        client = ExchangeClient(_valid_config())
        assert client.connect() is True

    def test_connect_failure_returns_false(self, monkeypatch):
        def boom(config):
            raise RuntimeError("network down")

        monkeypatch.setattr("execution.exchange_client._build_exchange", boom)
        client = ExchangeClient(_valid_config())
        assert client.connect() is False
        assert client._exchange is None


class FakeExchange:
    """In-memory exchange double for method tests."""
    def __init__(self):
        self.orders = []
        self.ticker = {"last": 100.0, "symbol": "BTC/USDT"}
        self.balance = {"USDT": {"free": 10000.0}}
        self.open_orders = []
        self._order_db = {}
        self.fail = {}

    def fetch_ticker(self, symbol):
        if self.fail.get("ticker"):
            raise RuntimeError("boom")
        return self.ticker

    def fetch_balance(self):
        if self.fail.get("balance"):
            raise RuntimeError("boom")
        return self.balance

    def fetch_open_orders(self, symbol):
        if self.fail.get("open_orders"):
            raise RuntimeError("boom")
        return self.open_orders

    def create_order(self, symbol, type_, side, amount):
        if self.fail.get("place"):
            raise RuntimeError("boom")
        oid = f"o{len(self.orders) + 1}"
        order = {"id": oid, "symbol": symbol, "side": side, "amount": amount,
                 "status": "closed", "type": type_}
        self.orders.append(order)
        self._order_db[oid] = order
        return order

    def cancel_order(self, order_id, symbol):
        if self.fail.get("cancel"):
            raise RuntimeError("boom")
        return {"id": order_id, "status": "canceled"}

    def fetch_order(self, order_id, symbol):
        if self.fail.get("status"):
            raise RuntimeError("boom")
        return self._order_db.get(order_id, {"id": order_id, "status": "open"})


@pytest.fixture
def client():
    c = ExchangeClient(_valid_config())
    c._exchange = FakeExchange()
    return c


class TestMethodsNeverRaise:

    def test_not_connected_returns_none(self):
        c = ExchangeClient(_valid_config())
        assert c.get_ticker("BTC/USDT") is None
        assert c.get_balance() is None
        assert c.get_open_orders("BTC/USDT") is None
        assert c.place_market_order("BTC/USDT", "buy", 1) is None
        assert c.get_order_status("o1", "BTC/USDT") is None
        assert c.cancel_order("o1", "BTC/USDT") is False

    def test_get_ticker(self, client):
        assert client.get_ticker("BTC/USDT")["last"] == 100.0

    def test_get_ticker_error_returns_none(self, client):
        client._exchange.fail["ticker"] = True
        assert client.get_ticker("BTC/USDT") is None

    def test_get_balance(self, client):
        assert client.get_balance()["USDT"]["free"] == 10000.0

    def test_get_balance_error_returns_none(self, client):
        client._exchange.fail["balance"] = True
        assert client.get_balance() is None

    def test_get_open_orders(self, client):
        assert client.get_open_orders("BTC/USDT") == []

    def test_get_open_orders_error_returns_none(self, client):
        client._exchange.fail["open_orders"] = True
        assert client.get_open_orders("BTC/USDT") is None

    def test_place_market_order(self, client):
        order = client.place_market_order("BTC/USDT", "buy", 0.5)
        assert order["side"] == "buy"
        assert order["amount"] == 0.5

    def test_place_market_order_error_returns_none(self, client):
        client._exchange.fail["place"] = True
        assert client.place_market_order("BTC/USDT", "buy", 0.5) is None

    def test_cancel_order(self, client):
        assert client.cancel_order("o1", "BTC/USDT") is True

    def test_cancel_order_error_returns_false(self, client):
        client._exchange.fail["cancel"] = True
        assert client.cancel_order("o1", "BTC/USDT") is False

    def test_get_order_status(self, client):
        client.place_market_order("BTC/USDT", "buy", 1)
        status = client.get_order_status("o1", "BTC/USDT")
        assert status["status"] == "closed"

    def test_get_order_status_error_returns_none(self, client):
        client._exchange.fail["status"] = True
        assert client.get_order_status("o1", "BTC/USDT") is None
