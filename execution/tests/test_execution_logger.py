"""Unit tests for execution.execution_logger (separate audit log)."""
from __future__ import annotations

import pytest

from execution.execution_logger import ExecutionLogger
from execution.order_manager import OrderParams, SignalInput


@pytest.fixture
def logger(tmp_path):
    return ExecutionLogger(db_path=str(tmp_path / "exec.db"))


def _signal(direction=1):
    return SignalInput(
        timestamp="2024-01-15T00:00:00Z", asset="BTC/USDT", direction=direction,
        har_predicted_range=200.0, confidence=0.6, regime="high",
    )


def _params():
    return OrderParams(
        symbol="BTC/USDT", side="buy", size=0.05, target_vol=0.01,
        har_vol_estimate=0.01, account_size=10000.0, direction=1,
        har_predicted_range=200.0, regime="high",
    )


class TestLogEvents:

    def test_log_signal(self, logger):
        logger.log_signal(_signal())
        rows = logger.get_execution_log()
        assert len(rows) == 1
        assert rows[0]["event_type"] == "signal"
        assert rows[0]["asset"] == "BTC/USDT"
        assert rows[0]["direction"] == 1
        assert rows[0]["details"]["har_predicted_range"] == 200.0

    def test_log_order_attempt(self, logger):
        logger.log_order_attempt(_signal(), _params())
        rows = logger.get_execution_log()
        assert rows[0]["event_type"] == "order_attempt"
        assert rows[0]["details"]["order_params"]["side"] == "buy"

    def test_log_order_result(self, logger):
        logger.log_order_result({"id": "x1", "status": "closed", "symbol": "BTC/USDT"})
        rows = logger.get_execution_log()
        assert rows[0]["event_type"] == "order_result"
        assert rows[0]["asset"] == "BTC/USDT"

    def test_log_skip(self, logger):
        logger.log_skip("direction is 0")
        rows = logger.get_execution_log()
        assert rows[0]["event_type"] == "skip"
        assert rows[0]["reason"] == "direction is 0"


class TestQuery:

    def test_log_ordering_and_count(self, logger):
        logger.log_signal(_signal())
        logger.log_order_attempt(_signal(), _params())
        logger.log_order_result({"status": "closed"})
        logger.log_skip("test")
        rows = logger.get_execution_log()
        assert [r["event_type"] for r in rows] == [
            "signal", "order_attempt", "order_result", "skip"]

    def test_empty_log(self, logger):
        assert logger.get_execution_log() == []

    def test_schema_columns(self, logger):
        logger.log_skip("x")
        row = logger.get_execution_log()[0]
        expected = {"id", "timestamp", "event_type", "asset", "direction",
                    "details", "reason", "created_at"}
        assert expected.issubset(set(row.keys()))

    def test_separate_from_har_predictions(self, tmp_path):
        # The logger DB is its own file, never the har_predictions store.
        import sqlite3
        logger = ExecutionLogger(db_path=str(tmp_path / "exec.db"))
        logger.log_signal(_signal())
        with sqlite3.connect(str(tmp_path / "exec.db")) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "execution_log" in tables
        assert "har_predictions" not in tables  # separate from the research store
