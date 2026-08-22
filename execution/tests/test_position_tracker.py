"""Unit tests for execution.position_tracker (local SQLite paper book)."""
from __future__ import annotations

import pytest

from execution.order_manager import OrderParams
from execution.position_tracker import PositionTracker


def _params(symbol="BTC/USDT", side="buy", size=0.1, direction=1,
            har_range=200.0, regime="high"):
    return OrderParams(
        symbol=symbol, side=side, size=size, target_vol=0.01,
        har_vol_estimate=0.01, account_size=10000.0, direction=direction,
        har_predicted_range=har_range, regime=regime,
    )


@pytest.fixture
def tracker(tmp_path):
    return PositionTracker(db_path=str(tmp_path / "positions.db"))


class TestOpenClose:

    def test_open_position_returns_id(self, tracker):
        pid = tracker.open_position(_params(), fill_price=20000.0)
        assert isinstance(pid, int)
        assert pid >= 1

    def test_open_then_close_long(self, tracker):
        pid = tracker.open_position(_params(size=0.5, direction=1), fill_price=20000.0)
        ok = tracker.close_position(pid, exit_price=21000.0)
        assert ok is True
        hist = tracker.get_position_history()
        assert hist[0]["status"] == "closed"
        # long pnl = (21000-20000)*0.5 = 500
        assert hist[0]["pnl"] == pytest.approx(500.0)

    def test_close_short_pnl_sign(self, tracker):
        pid = tracker.open_position(_params(size=0.5, direction=-1, side="sell"),
                                    fill_price=20000.0)
        tracker.close_position(pid, exit_price=21000.0)
        # short pnl = (20000-21000)*0.5 = -500
        hist = tracker.get_position_history()
        assert hist[0]["pnl"] == pytest.approx(-500.0)

    def test_close_unknown_id(self, tracker):
        assert tracker.close_position(999, exit_price=1.0) is False

    def test_double_close_returns_false(self, tracker):
        pid = tracker.open_position(_params(), fill_price=100.0)
        assert tracker.close_position(pid, 110.0) is True
        assert tracker.close_position(pid, 120.0) is False


class TestQueries:

    def test_open_positions_and_history(self, tracker):
        p1 = tracker.open_position(_params(symbol="BTC/USDT"), fill_price=20000.0)
        p2 = tracker.open_position(_params(symbol="ETH/USDT", direction=1), fill_price=1000.0)
        tracker.close_position(p1, 20500.0)
        opens = tracker.get_open_positions()
        assert len(opens) == 1
        assert opens[0]["asset"] == "ETH/USDT"
        hist = tracker.get_position_history()
        assert len(hist) == 2

    def test_open_positions_empty(self, tracker):
        assert tracker.get_open_positions() == []
        assert tracker.get_position_history() == []


class TestPnl:

    def test_compute_paper_pnl(self, tracker):
        p1 = tracker.open_position(_params(size=1.0, direction=1), fill_price=100.0)
        p2 = tracker.open_position(_params(size=1.0, direction=1), fill_price=100.0)
        tracker.close_position(p1, 110.0)   # +10
        tracker.close_position(p2, 95.0)    # -5
        pnl = tracker.compute_paper_pnl()
        assert pnl["realized_pnl"] == pytest.approx(5.0)
        assert pnl["n_closed"] == 2
        assert pnl["n_open"] == 0
        assert pnl["win_rate"] == pytest.approx(0.5)

    def test_pnl_all_open(self, tracker):
        tracker.open_position(_params(), fill_price=100.0)
        pnl = tracker.compute_paper_pnl()
        assert pnl["realized_pnl"] == 0.0
        assert pnl["n_open"] == 1
        assert pnl["n_closed"] == 0


class TestSchemaPersistence:

    def test_columns_match_contract(self, tracker):
        tracker.open_position(_params(), fill_price=100.0)
        row = tracker.get_position_history()[0]
        expected = {
            "id", "timestamp", "asset", "direction", "entry_price", "size",
            "har_predicted_range", "regime", "status", "exit_price", "pnl",
            "exit_timestamp",
        }
        assert expected.issubset(set(row.keys()))
