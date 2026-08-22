"""Tests for execution.exchange_client (sandbox-enforced, fail-soft)."""
from __future__ import annotations

import pytest

from execution.exchange_client import (
    ExchangeClient,
    ExchangeConfig,
    SandboxViolation,
    _build_exchange,
)


class FakeExchange:
    def __init__(self):
        self.fail = {}
        self.placed = []

    def set_sandbox_mode(self, v):
        assert v is True

    def load_markets(self):
        if self.fail.get("connect"):
            raise RuntimeError("boom")

    def fetch_ticker(self, s):
        if self.fail.get("ticker"):
            raise RuntimeError("boom")
        return {"last": 100.0, "symbol": s}

    def fetch_balance(self):
        if self.fail.get("balance"):
            raise RuntimeError("boom")
        return {"USDT": {"free": 10000.0}}

    def fetch_open_orders(self, s):
        if self.fail.get("open"):
            raise RuntimeError("boom")
        return []

    def create_order(self, s, t, side, amt):
        if self.fail.get("place"):
            raise RuntimeError("boom")
        o = {"id": "x1", "symbol": s, "side": side, "amount": amt, "status": "closed"}
        self.placed.append(o)
        return o

    def cancel_order(self, oid, s):
        if self.fail.get("cancel"):
            raise RuntimeError("boom")
        return {"id": oid, "status": "canceled"}

    def fetch_order(self, oid, s):
        if self.fail.get("status"):
            raise RuntimeError("boom")
        return {"id": oid, "status": "closed"}


def _config(**kw):
    base = dict(api_key="k", api_secret="s", api_password="p", sandbox=True)
    base.update(kw)
    return ExchangeConfig(**base)


@pytest.fixture
def client(monkeypatch):
    fake = FakeExchange()
    monkeypatch.setattr("execution.exchange_client._build_exchange",
                        lambda cfg: fake)
    return ExchangeClient(_config()), fake


class TestSandboxEnforcement:

    def test_sandbox_false_raises_value_error(self, monkeypatch):
        monkeypatch.setattr("execution.exchange_client._build_exchange",
                            lambda cfg: FakeExchange())
        with pytest.raises(ValueError):
            ExchangeClient(_config(sandbox=False))

    def test_sandbox_false_raises_specific(self, monkeypatch):
        monkeypatch.setattr("execution.exchange_client._build_exchange",
                            lambda cfg: FakeExchange())
        with pytest.raises(SandboxViolation):
            ExchangeClient(_config(sandbox=False))

    def test_default_sandbox_true(self, client):
        c, _ = client
        assert c.config.sandbox is True


class TestConnect:

    def test_connect_success(self, client):
        c, _ = client
        assert c.connect() is True

    def test_connect_failure_returns_false(self, client):
        c, fake = client
        fake.fail["connect"] = True
        assert c.connect() is False


class TestFailSoftMethods:

    def test_get_ticker(self, client):
        c, _ = client
        assert c.get_ticker("BTC/USDT")["last"] == 100.0

    def test_get_ticker_error_none(self, client):
        c, fake = client
        fake.fail["ticker"] = True
        assert c.get_ticker("BTC/USDT") is None

    def test_get_balance(self, client):
        c, _ = client
        assert c.get_balance()["USDT"]["free"] == 10000.0

    def test_get_balance_error_none(self, client):
        c, fake = client
        fake.fail["balance"] = True
        assert c.get_balance() is None

    def test_get_open_orders(self, client):
        c, _ = client
        assert c.get_open_orders("BTC/USDT") == []

    def test_get_open_orders_error_none(self, client):
        c, fake = client
        fake.fail["open"] = True
        assert c.get_open_orders("BTC/USDT") is None

    def test_place_market_order(self, client):
        c, _ = client
        o = c.place_market_order("BTC/USDT", "buy", 0.05)
        assert o["side"] == "buy" and o["amount"] == 0.05

    def test_place_market_order_error_none(self, client):
        c, fake = client
        fake.fail["place"] = True
        assert c.place_market_order("BTC/USDT", "buy", 0.05) is None

    def test_cancel_order(self, client):
        c, _ = client
        assert c.cancel_order("x1", "BTC/USDT") is True

    def test_cancel_order_error_false(self, client):
        c, fake = client
        fake.fail["cancel"] = True
        assert c.cancel_order("x1", "BTC/USDT") is False

    def test_get_order_status(self, client):
        c, _ = client
        assert c.get_order_status("x1", "BTC/USDT")["status"] == "closed"

    def test_get_order_status_error_none(self, client):
        c, fake = client
        fake.fail["status"] = True
        assert c.get_order_status("x1", "BTC/USDT") is None
