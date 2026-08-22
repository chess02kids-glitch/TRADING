"""Tests for scripts/migrate_phase9a.py core logic (SQLite-backed)."""
from __future__ import annotations

import sqlite3

from scripts.migrate_phase9a import (
    PHASE9A_COLUMNS,
    apply_migration,
    count_rows,
    get_column_names,
)


def _make_db(tmp_path):
    db = str(tmp_path / "h.db")
    conn = sqlite3.connect(db)
    conn.execute(
        'CREATE TABLE har_predictions (id INTEGER PRIMARY KEY, "timestamp" TEXT, '
        "asset TEXT, breakout_flag INTEGER, actual_range REAL)")
    conn.execute(
        'INSERT INTO har_predictions ("timestamp", asset, breakout_flag, '
        "actual_range) VALUES (?, ?, ?, ?)",
        ("2024-01-01T00:00:00Z", "BTC/USDT", 1, 100.0))
    conn.commit()
    return conn


class TestApplyMigration:

    def test_adds_columns_preserves_rows(self, tmp_path):
        conn = _make_db(tmp_path)
        before = count_rows(conn)
        added, skipped = apply_migration(conn)
        assert set(added) == {name for name, _ in PHASE9A_COLUMNS}
        assert skipped == []
        assert count_rows(conn) == before
        cols = get_column_names(conn)
        assert {"breakout_direction", "breakout_candle_open",
                "breakout_candle_close"} <= cols
        conn.close()

    def test_idempotent(self, tmp_path):
        conn = _make_db(tmp_path)
        apply_migration(conn)
        added, skipped = apply_migration(conn)
        assert added == []
        assert set(skipped) == {name for name, _ in PHASE9A_COLUMNS}
        conn.close()

    def test_count_rows_zero_when_missing(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "empty.db"))
        assert count_rows(conn, "nonexistent") == 0
        conn.close()
