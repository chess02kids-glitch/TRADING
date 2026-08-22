"""Tests for execution.position_tracker (local SQLite paper book)."""
from __future__ import annotations

import pytest

from execution.position_tracker import (
    close_position,
    compute_paper_pnl,
    get_closed_positions,
    get_open_positions,
    initialize_db,
    open_position,
)


@pytest.fixture
def db(tmp_path):
    initialize_db(str(tmp_path / "pp.db"))
    return str(tmp_path / "pp.db")


class TestOpenClose:

    def test_open_returns_id(self, db):
        pid = open_position("BTC/USDT", 1, 20000.0, 0.05, 1000.0, 200.0, "high")
        assert isinstance(pid, int) and pid >= 1

    def test_close_long_pnl(self, db):
        pid = open_position("BTC/USDT", 1, 100.0, 1.0, 100.0, 10.0, "high")
        assert close_position(pid, 110.0) is True
        row = get_closed_positions()[0]
        assert row["pnl_usd"] == pytest.approx(10.0)
        assert row["pnl_pct"] == pytest.approx(10.0)

    def test_close_short_pnl(self, db):
        pid = open_position("BTC/USDT", -1, 100.0, 1.0, 100.0, 10.0, "high")
        close_position(pid, 110.0)  # short loses when price rises
        row = get_closed_positions()[0]
        assert row["pnl_usd"] == pytest.approx(-10.0)

    def test_close_unknown_returns_false(self, db):
        assert close_position(999, 100.0) is False

    def test_double_close_returns_false(self, db):
        pid = open_position("BTC/USDT", 1, 100.0, 1.0, 100.0, 10.0, "high")
        assert close_position(pid, 110.0) is True
        assert close_position(pid, 120.0) is False


class TestQueries:

    def test_open_and_closed_lists(self, db):
        p1 = open_position("BTC/USDT", 1, 100.0, 1.0, 100.0, 10.0, "high")
        p2 = open_position("ETH/USDT", 1, 50.0, 2.0, 100.0, 5.0, "low")
        close_position(p1, 110.0)
        opens = get_open_positions()
        assert len(opens) == 1 and opens[0]["asset"] == "ETH/USDT"
        closed = get_closed_positions()
        assert len(closed) == 1 and closed[0]["asset"] == "BTC/USDT"


class TestPnl:

    def test_compute_paper_pnl(self, db):
        p1 = open_position("BTC/USDT", 1, 100.0, 1.0, 100.0, 10.0, "high")
        p2 = open_position("BTC/USDT", 1, 100.0, 1.0, 100.0, 10.0, "high")
        close_position(p1, 110.0)   # +10
        close_position(p2, 95.0)    # -5
        pnl = compute_paper_pnl()
        assert pnl["total_pnl_usd"] == pytest.approx(5.0)
        assert pnl["closed_trades"] == 2
        assert pnl["open_trades"] == 0
        assert pnl["total_trades"] == 2
        assert pnl["win_rate"] == pytest.approx(0.5)
        assert pnl["avg_pnl_pct"] == pytest.approx(2.5)  # (10 + -5)/2

    def test_pnl_all_open(self, db):
        open_position("BTC/USDT", 1, 100.0, 1.0, 100.0, 10.0, "high")
        pnl = compute_paper_pnl()
        assert pnl["total_pnl_usd"] == 0.0
        assert pnl["open_trades"] == 1
        assert pnl["closed_trades"] == 0


class TestSchema:

    def test_columns_match_contract(self, db):
        open_position("BTC/USDT", 1, 100.0, 1.0, 100.0, 10.0, "high")
        row = get_open_positions()[0]
        expected = {
            "id", "timestamp", "asset", "direction", "entry_price", "size_base",
            "size_usd", "har_predicted_range", "regime", "status", "exit_price",
            "pnl_usd", "pnl_pct", "exit_timestamp", "created_at",
        }
        assert expected.issubset(set(row.keys()))
