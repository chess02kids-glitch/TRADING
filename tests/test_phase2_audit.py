"""Deterministic tests for the read-only Phase 2 candle audit."""

import importlib.util
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase2_audit", ROOT / "scripts" / "setup" / "audit_phase2_db.py")
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)

DAY = 86_400_000
HOUR = 3_600_000
REFERENCE = int(datetime(2026, 8, 10, 12, tzinfo=timezone.utc).timestamp() * 1000)


def _db():
    conn = sqlite3.connect(":memory:")
    # Deliberately no constraints: audit must detect defects in legacy/corrupt DBs.
    conn.execute("""CREATE TABLE ohlcv_raw (
        exchange TEXT, symbol TEXT, timeframe TEXT, timestamp_ms INTEGER,
        timestamp_utc TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL
    )""")
    conn.execute("""CREATE TABLE fetch_metadata (
        fetch_start_ms INTEGER, fetch_end_ms INTEGER, symbol TEXT, timeframe TEXT
    )""")
    return conn


def _insert(conn, ts, timeframe="1h", utc=None, high=101, low=99, open_=100, close=100, volume=1):
    conn.execute("INSERT INTO ohlcv_raw VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
        "binance", "BTC/USDT", timeframe, ts, utc or audit.iso(ts), open_, high, low, close, volume,
    ))


@pytest.mark.parametrize("days", [30, 90, 730])
def test_approximate_requested_range_and_open_last_candle_pass(days):
    conn = _db()
    # First open is a clean boundary, while a fetch request could have started mid-candle.
    first = (REFERENCE - days * DAY) // HOUR * HOUR
    last = REFERENCE // HOUR * HOUR
    for ts in range(first, last + HOUR, HOUR):
        _insert(conn, ts)
    result = audit.audit_series(conn, "BTC/USDT", "1h", days, now_ms=REFERENCE)
    assert result["pass"]
    assert result["last_candle_currently_forming"] is True
    assert result["closed_candle_only_training_available"] is True


def test_daily_closed_last_candle_is_not_a_failure():
    conn = _db()
    last = REFERENCE // DAY * DAY - DAY  # yesterday's daily candle is closed
    first = last - 729 * DAY
    for ts in range(first, last + DAY, DAY):
        _insert(conn, ts, timeframe="1d")
    result = audit.audit_series(conn, "BTC/USDT", "1d", 730, now_ms=REFERENCE)
    assert result["pass"]
    assert result["last_candle_currently_forming"] is False
    assert result["closed_candle_rows"] == 730


def test_request_window_timestamps_are_never_treated_as_candles():
    conn = _db()
    first = (REFERENCE - 30 * DAY) // HOUR * HOUR
    for ts in range(first, REFERENCE // HOUR * HOUR + HOUR, HOUR):
        _insert(conn, ts)
    # Arbitrary non-boundary provenance must not create a false gap/alignment error.
    conn.execute("INSERT INTO fetch_metadata VALUES (?, ?, ?, ?)",
                 (first + 123_456, REFERENCE - 654_321, "BTC/USDT", "1h"))
    result = audit.audit_series(conn, "BTC/USDT", "1h", 30, now_ms=REFERENCE)
    assert result["pass"]
    assert result["missing_candle_count"] == 0
    assert result["misaligned_timestamp_count"] == 0


def test_real_gap_is_reported_exactly_and_fails_without_filling():
    conn = _db()
    first = REFERENCE // HOUR * HOUR - 30 * DAY
    missing = first + 10 * HOUR
    for ts in range(first, first + 30 * DAY + HOUR, HOUR):
        if ts != missing:
            _insert(conn, ts)
    result = audit.audit_series(conn, "BTC/USDT", "1h", 30, now_ms=REFERENCE)
    assert not result["pass"]
    assert result["missing_candle_count"] == 1
    assert result["missing_ranges"] == [{"start_ms": missing, "end_ms": missing,
                                           "start_utc": audit.iso(missing), "end_utc": audit.iso(missing),
                                           "missing_count": 1}]
    assert "genuinely missing" in "; ".join(result["reasons"])


def test_duplicate_invalid_ohlc_misalignment_and_non_utc_are_distinct_defects():
    conn = _db()
    first = REFERENCE // HOUR * HOUR - 30 * DAY
    for ts in range(first, REFERENCE // HOUR * HOUR + HOUR, HOUR):
        _insert(conn, ts)
    _insert(conn, first, utc="2020-01-01T00:00:00+00:00")  # duplicate and inconsistent UTC text
    _insert(conn, first + HOUR, high=90, low=99)  # invalid OHLC
    _insert(conn, first + 2 * HOUR + 1, utc="2026-01-01T00:00:00+02:00")  # misaligned + non-UTC
    result = audit.audit_series(conn, "BTC/USDT", "1h", 30, now_ms=REFERENCE)
    assert not result["pass"]
    assert result["duplicate_timestamp_groups"] == 2
    assert result["invalid_ohlc_count"] == 1
    assert result["misaligned_timestamp_count"] == 1
    assert result["non_utc_count"] >= 2
