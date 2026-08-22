"""Tests for execution.execution_logger (separate audit log)."""
from __future__ import annotations

import pytest

from execution.execution_logger import (
    configure,
    get_log,
    log_order_attempt,
    log_order_result,
    log_signal,
    log_skip,
    reset,
)
from execution.order_manager import OrderParams, SignalInput


@pytest.fixture(autouse=True)
def _isolated():
    reset()
    configure(None)
    yield
    reset()


def _signal(direction=1):
    return SignalInput(
        timestamp="2024-01-15T00:00:00Z", asset="BTC/USDT", direction=direction,
        har_predicted_range=200.0, confidence=0.6, regime="high",
    )


def _params():
    return OrderParams(
        symbol="BTC/USDT", side="buy", size_usd=1000.0, size_base=0.05,
        target_vol=0.01, har_vol_estimate=0.01, account_size=10000.0,
    )


class TestLogEvents:

    def test_log_signal(self):
        log_signal(_signal())
        log = get_log()
        assert len(log) == 1
        assert log[0]["event_type"] == "signal"
        assert log[0]["signal"]["asset"] == "BTC/USDT"

    def test_log_skip(self):
        log_skip("below minimum", _signal())
        log = get_log()
        assert log[0]["event_type"] == "skip"
        assert log[0]["reason"] == "below minimum"
        assert log[0]["signal"]["asset"] == "BTC/USDT"

    def test_log_order_attempt(self):
        log_order_attempt(_params())
        log = get_log()
        assert log[0]["event_type"] == "order_attempt"
        assert log[0]["order_params"]["side"] == "buy"

    def test_log_order_result_success(self):
        log_order_result({"id": "x1", "status": "closed"}, success=True)
        log = get_log()
        assert log[0]["event_type"] == "order_result"
        assert log[0]["success"] is True
        assert log[0]["result"]["status"] == "closed"

    def test_log_order_result_none_rejected(self):
        log_order_result(None, success=False)
        log = get_log()
        assert log[0]["success"] is False
        assert log[0]["result"] is None


class TestGetLog:

    def test_ordering(self):
        log_signal(_signal())
        log_order_attempt(_params())
        log_order_result({"status": "closed"}, True)
        log_skip("done", _signal())
        types = [e["event_type"] for e in get_log()]
        assert types == ["signal", "order_attempt", "order_result", "skip"]

    def test_empty_after_reset(self):
        assert get_log() == []

    def test_returns_copy(self):
        log_signal(_signal())
        first = get_log()
        first.clear()
        assert len(get_log()) == 1  # internal log unaffected


class TestFileLogging:

    def test_writes_plain_text_when_configured(self, tmp_path):
        configure(str(tmp_path / "exec.log"))
        log_signal(_signal())
        text = (tmp_path / "exec.log").read_text()
        assert "signal" in text and "BTC/USDT" in text
