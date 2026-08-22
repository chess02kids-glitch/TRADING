"""Tests for kronos_trading.alerts.forward_return_logger (in-memory SQLite).

Covers: idempotent table creation, 3-row breakout tracking with idempotent
re-tracking, forward-return filling when the target candle is present, the
silent skip when it is absent, the returned update count, the filled-only
query, and forward-direction sign on gains vs losses.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from kronos_trading.alerts.forward_return_logger import (
    HORIZONS,
    create_phase9a_table,
    get_phase9a_data,
    log_breakout_for_tracking,
    update_forward_returns,
)

BT = "2024-01-15T14:00:00Z"
ASSET = "BTC/USDT"


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


def _count(conn):
    return conn.execute("SELECT COUNT(*) FROM phase9a_forward_returns").fetchone()[0]


def _candles(closes):
    """DataFrame[timestamp(ISO), close] for hours starting at BT."""
    base = pd.Timestamp(BT)
    rows = []
    for i, cl in enumerate(closes):
        ts = (base + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append({"timestamp": ts, "close": float(cl)})
    return pd.DataFrame(rows)


class TestTableCreation:

    def test_table_creation_idempotent(self, conn):
        create_phase9a_table(conn)
        create_phase9a_table(conn)  # second call must not error
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "phase9a_forward_returns" in tables


class TestLogBreakout:

    def test_log_breakout_creates_3_rows(self, conn):
        log_breakout_for_tracking(conn, BT, ASSET, direction=1, close_price=100.0)
        assert _count(conn) == 3
        rows = conn.execute(
            "SELECT horizon, forward_return, forward_direction, "
            "breakout_close_price FROM phase9a_forward_returns ORDER BY horizon"
        ).fetchall()
        assert [r[0] for r in rows] == list(HORIZONS)
        for r in rows:
            assert r[1] is None and r[2] is None  # not filled yet
            assert r[3] == 100.0
        # target timestamps are BT + horizon hours
        targets = [r[0] for r in conn.execute(
            "SELECT target_timestamp FROM phase9a_forward_returns ORDER BY horizon")]
        assert targets == [
            "2024-01-15T15:00:00Z", "2024-01-15T16:00:00Z", "2024-01-15T17:00:00Z"]

    def test_log_breakout_skips_duplicate(self, conn):
        log_breakout_for_tracking(conn, BT, ASSET, direction=1, close_price=100.0)
        log_breakout_for_tracking(conn, BT, ASSET, direction=1, close_price=100.0)
        assert _count(conn) == 3  # UNIQUE constraint -> no duplicates


class TestUpdate:

    def test_update_fills_return_when_candle_found(self, conn):
        log_breakout_for_tracking(conn, BT, ASSET, direction=1, close_price=100.0)
        # target h1 = 15:00 close 110 -> +10%
        candles = {ASSET: _candles([100, 110, 120, 130])}
        now = "2024-01-15T18:00:00Z"
        n = update_forward_returns(conn, now, candles)
        assert n == 3
        rows = conn.execute(
            "SELECT horizon, forward_return, forward_direction "
            "FROM phase9a_forward_returns ORDER BY horizon"
        ).fetchall()
        assert rows[0][1] == pytest.approx(110 / 100 - 1)  # h1
        assert rows[0][2] == 1

    def test_update_skips_when_candle_not_found(self, conn):
        log_breakout_for_tracking(conn, BT, ASSET, direction=1, close_price=100.0)
        # candle window ends before the target bars
        candles = {ASSET: _candles([100])}
        n = update_forward_returns(conn, "2024-01-15T18:00:00Z", candles)
        assert n == 0
        rows = conn.execute(
            "SELECT forward_return FROM phase9a_forward_returns").fetchall()
        assert all(r[0] is None for r in rows)

    def test_update_returns_correct_count(self, conn):
        log_breakout_for_tracking(conn, BT, ASSET, direction=1, close_price=100.0)
        # only h1 target (15:00) is within the candle window + now
        candles = {ASSET: _candles([100, 110])}
        n = update_forward_returns(conn, "2024-01-15T15:30:00Z", candles)
        assert n == 1  # only the h1 row (15:00 <= 15:30)

    def test_forward_direction_positive_on_gain(self, conn):
        log_breakout_for_tracking(conn, BT, ASSET, direction=1, close_price=100.0)
        update_forward_returns(conn, "2024-01-15T18:00:00Z",
                               {ASSET: _candles([100, 150, 160, 170])})
        h1 = conn.execute(
            "SELECT forward_direction FROM phase9a_forward_returns WHERE horizon=1"
        ).fetchone()
        assert h1[0] == 1

    def test_forward_direction_negative_on_loss(self, conn):
        log_breakout_for_tracking(conn, BT, ASSET, direction=1, close_price=100.0)
        update_forward_returns(conn, "2024-01-15T18:00:00Z",
                               {ASSET: _candles([100, 90, 80, 70])})
        h1 = conn.execute(
            "SELECT forward_direction FROM phase9a_forward_returns WHERE horizon=1"
        ).fetchone()
        assert h1[0] == -1


class TestGetData:

    def test_get_phase9a_data_returns_filled_only(self, conn):
        log_breakout_for_tracking(conn, BT, ASSET, direction=1, close_price=100.0)
        # Fill only h1 (candles cover just 15:00; now after 15:00 but not 16/17)
        update_forward_returns(conn, "2024-01-15T15:30:00Z",
                               {ASSET: _candles([100, 110])})
        df = get_phase9a_data(conn)
        assert len(df) == 1  # only the filled h1 row
        assert set(df.columns) >= {
            "breakout_timestamp", "asset", "breakout_direction", "horizon",
            "target_timestamp", "forward_return", "forward_direction",
            "breakout_close_price"}
        assert df.iloc[0]["horizon"] == 1
