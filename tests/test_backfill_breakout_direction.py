"""Tests for scripts/backfill_breakout_direction.py (pure logic + SQLite)."""
from __future__ import annotations

import sqlite3

from scripts.backfill_breakout_direction import (
    compute_direction,
    find_candle,
    run_backfill,
)


class TestPureHelpers:

    def test_compute_direction(self):
        assert compute_direction(100, 110) == 1
        assert compute_direction(110, 100) == -1
        assert compute_direction(100, 100) == 1  # equal -> UP

    def test_find_candle_match_and_miss(self):
        rows = [[1000, 100.0, 105.0, 99.0, 102.0, 1.0],
                [2000, 102.0, 106.0, 100.0, 104.0, 1.0]]
        assert find_candle(rows, 2000) == (102.0, 104.0)
        assert find_candle(rows, 9999) is None


def _make_db(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "b.db"))
    conn.row_factory = sqlite3.Row
    conn.execute(
        'CREATE TABLE har_predictions (id INTEGER PRIMARY KEY, "timestamp" TEXT, '
        "asset TEXT, breakout_flag INTEGER, actual_range REAL, "
        "breakout_direction INTEGER, breakout_candle_open REAL, "
        "breakout_candle_close REAL)")
    return conn


class TestRunBackfill:

    def test_updates_rows_when_candle_found(self, tmp_path):
        conn = _make_db(tmp_path)
        conn.execute(
            'INSERT INTO har_predictions ("timestamp", asset, breakout_flag, '
            "actual_range, breakout_direction) VALUES (?, ?, ?, ?, ?)",
            ("2024-01-15T14:00:00Z", "BTC/USDT", 1, 300.0, None))
        conn.commit()

        def fetcher(exchange, asset, ts_ms):
            # close 102 >= open 100 -> UP (+1)
            return [[ts_ms, 100.0, 105.0, 95.0, 102.0, 1.0]]

        counts = run_backfill(conn, fetcher=fetcher, rate_limit_sleep=0)
        assert counts["found"] == 1
        assert counts["updated"] == 1
        row = conn.execute(
            "SELECT breakout_direction, breakout_candle_open, "
            "breakout_candle_close FROM har_predictions").fetchone()
        assert row["breakout_direction"] == 1
        assert row["breakout_candle_open"] == 100.0
        assert row["breakout_candle_close"] == 102.0
        conn.close()

    def test_skips_when_candle_not_found(self, tmp_path):
        conn = _make_db(tmp_path)
        conn.execute(
            'INSERT INTO har_predictions ("timestamp", asset, breakout_flag, '
            "actual_range, breakout_direction) VALUES (?, ?, ?, ?, ?)",
            ("2024-01-15T14:00:00Z", "BTC/USDT", 1, 300.0, None))
        conn.commit()
        counts = run_backfill(conn, fetcher=lambda *a: [], rate_limit_sleep=0)
        assert counts["not_found"] == 1
        assert counts["updated"] == 0
        conn.close()

    def test_never_overwrites_filled_row(self, tmp_path):
        conn = _make_db(tmp_path)
        conn.execute(
            'INSERT INTO har_predictions ("timestamp", asset, breakout_flag, '
            "actual_range, breakout_direction) VALUES (?, ?, ?, ?, ?)",
            ("2024-01-15T14:00:00Z", "BTC/USDT", 1, 300.0, -1))  # already set
        conn.commit()
        counts = run_backfill(conn, fetcher=lambda *a: [[1, 1, 1, 1, 1, 1]],
                              rate_limit_sleep=0)
        assert counts["found"] == 0  # filtered out (direction not NULL)
        conn.close()

    def test_dry_run_writes_nothing(self, tmp_path):
        conn = _make_db(tmp_path)
        conn.execute(
            'INSERT INTO har_predictions ("timestamp", asset, breakout_flag, '
            "actual_range, breakout_direction) VALUES (?, ?, ?, ?, ?)",
            ("2024-01-15T14:00:00Z", "BTC/USDT", 1, 300.0, None))
        conn.commit()
        counts = run_backfill(conn, dry_run=True,
                              fetcher=lambda ex, a, ts_ms: [[ts_ms, 100.0, 1, 1, 102.0, 1]],
                              rate_limit_sleep=0)
        assert counts["updated"] == 1
        row = conn.execute(
            "SELECT breakout_direction FROM har_predictions").fetchone()
        assert row["breakout_direction"] is None  # nothing written
        conn.close()
