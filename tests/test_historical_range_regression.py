#!/usr/bin/env python3
"""
Phase 2.5 - Historical Data Range Regression Tests
===================================================

Locks in the fix for the reported defect:
    python scripts/data/run_fetch.py --assets BTC/USDT ETH/USDT \\
        --timeframes 1h 4h 1d --days 730 --incremental
appeared to produce only ~30 days of data.

Two-part root cause (see docs/PHASE_2_5_AUDIT.md):
  1. CLI: argparse had no ``--incremental`` flag (only ``--no-incremental``),
     so the exact documented command exited with code 2 - it fetched NOTHING
     and the user observed the stale pre-existing DB contents (~30 days).
  2. (Already fixed in run_all before this audit) incremental mode previously
     ignored ``--days`` when the DB already contained data, so a short
     history was never backfilled. The backfill logic is kept and now locked
     by these tests.

What is proven here (with a MOCKED exchange - no API keys, no network,
no dependence on live Binance availability):
  * --days 30  requests ~30 days
  * --days 90  requests ~90 days
  * --days 730 requests ~730 days
  * requests beyond exchange history are capped at exchange availability
  * Binance 1000-candle/request pagination limit is respected
  * incremental mode backfills a short DB to the requested --days depth
  * duplicate protection (repeated runs insert 0 new rows)
  * no silent candle filling (exchange gaps are reported, never fabricated)
  * UTC timestamps throughout, OHLC sanity preserved
  * deterministic storage/export (byte-identical CSV on re-export)

Run: pytest tests/test_historical_range_regression.py -v
"""

import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

import run_fetch
from fetcher import BinancePublicFetcher, TIMEFRAME_MS
from storage import SQLiteStorage
from validator import DataValidator

logging.getLogger().setLevel(logging.CRITICAL)

# Binance spot listed BTC/USDT and ETH/USDT on 2017-08-17 00:00:00 UTC.
# Used by the mock to emulate "exchange availability" - the regression suite
# itself never touches the real exchange.
BINANCE_SPOT_GENESIS_MS = 1502928000000  # 2017-08-17T00:00:00+00:00
DAY_MS = 24 * 3600000


class MockBinanceKlines:
    """
    Offline, deterministic emulation of ccxt binance.fetch_ohlcv.

    Emulates real Binance /api/v3/klines semantics:
      * returns rows with open_time >= since (aligned up to the timeframe
        boundary, never before genesis_ms - that is "exchange availability")
      * returns at most max_limit rows per request (Binance hard cap: 1000)
      * like the real SQL-backed klines endpoint, a page holds up to
        ``limit`` *existing* klines; an exchange outage gap simply means
        those rows do not exist (subsequent candles fill the page instead)
      * data exists up to "now" (the current forming candle included,
        exactly like real Binance)
      * no API key of any kind is involved
    """

    def __init__(self, genesis_ms=BINANCE_SPOT_GENESIS_MS, max_limit=1000,
                 gap_start_ms=None, gap_end_ms=None):
        self.genesis_ms = genesis_ms
        self.max_limit = max_limit
        self.gap = (gap_start_ms, gap_end_ms) if gap_start_ms is not None else None
        self.calls = []  # [{symbol, timeframe, since, limit, rows_returned}]

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        tf_ms = TIMEFRAME_MS[timeframe]
        effective_limit = min(limit if limit is not None else 500, self.max_limit)
        assert 1 <= effective_limit <= self.max_limit, \
            f"pipeline must respect Binance limit, asked {limit}"

        start = max(since if since is not None else self.genesis_ms, self.genesis_ms)
        first = ((start + tf_ms - 1) // tf_ms) * tf_ms  # align to candle boundary >= start
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        out = []
        ts = first
        while len(out) < effective_limit and ts <= now_ms:
            in_gap = self.gap is not None and self.gap[0] <= ts < self.gap[1]
            if not in_gap:
                out.append([ts, 100.0, 110.0, 90.0, 105.0, 10.0])
            ts += tf_ms

        self.calls.append({
            "symbol": symbol, "timeframe": timeframe,
            "since": since, "limit": effective_limit, "rows_returned": len(out),
        })
        return out


class _TestStorage(SQLiteStorage):
    """SQLiteStorage with CSV export redirected out of the repo tree."""
    export_dir = None

    def export_csv(self, symbol, timeframe, exchange="binance", output_dir=None):
        return super().export_csv(symbol, timeframe, exchange,
                                  output_dir=output_dir or self.export_dir)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """
    Wire run_fetch to a mocked exchange + temp DB/CSV dirs.
    Returns a run() helper: run(days, timeframe, incremental, mock, db_path)
    -> (SQLiteStorage handle, MockBinanceKlines used)
    """
    _TestStorage.export_dir = tmp_path / "raw"
    holder = {"mock": None}

    def fetcher_factory(*args, **kwargs):
        # Real fetcher object (retry/pagination/dedup logic intact), but its
        # network exchange is replaced by the mock and load_markets is a no-op.
        fetcher = BinancePublicFetcher(enable_rate_limit=False)
        fetcher.exchange = holder["mock"]
        fetcher.load_markets = lambda: None
        return fetcher

    monkeypatch.setattr(run_fetch, "BinancePublicFetcher", fetcher_factory)
    monkeypatch.setattr(run_fetch, "SQLiteStorage", _TestStorage)
    # Keep the run's reporting step from writing into the repo during tests
    monkeypatch.setattr(run_fetch, "generate_final_report", lambda *a, **k: {})

    def run(days, timeframe="1h", symbol="BTC/USDT", incremental=True,
            mock=None, db_path=None):
        holder["mock"] = mock if mock is not None else MockBinanceKlines()
        db_path = db_path or (tmp_path / f"test_{symbol.replace('/', '')}_{timeframe}_{days}d.db")
        run_fetch.run_all(
            assets=[symbol],
            timeframes=[timeframe],
            days_history=days,
            incremental=incremental,
            db_path=db_path,
        )
        return SQLiteStorage(db_path=db_path), holder["mock"]

    return run


def _now_ms():
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _db_span_days(storage, symbol, timeframe):
    first_ms, last_ms = storage.get_date_range(symbol, timeframe)
    if first_ms is None:
        return None, None, None
    return first_ms, last_ms, (last_ms - first_ms) / DAY_MS


# =====================================================================
# 1. CLI contract - the exact command from the defect report must parse
# =====================================================================

def test_cli_accepts_documented_incremental_command():
    """
    Regression: argparse previously rejected '--incremental' (exit code 2),
    so the documented command fetched nothing at all.
    """
    parser = run_fetch.build_parser()
    args = parser.parse_args([
        "--assets", "BTC/USDT", "ETH/USDT",
        "--timeframes", "1h", "4h", "1d",
        "--days", "730", "--incremental",
    ])
    assert args.incremental is True
    assert args.days == 730
    assert args.assets == ["BTC/USDT", "ETH/USDT"]
    assert args.timeframes == ["1h", "4h", "1d"]


def test_cli_defaults_and_no_incremental():
    parser = run_fetch.build_parser()
    defaults = parser.parse_args([])
    assert defaults.incremental is True, "incremental must remain the default"
    assert defaults.days == 730, "default history must remain 730 days"
    assert defaults.assets == ["BTC/USDT", "ETH/USDT"]
    assert defaults.timeframes == ["1h", "4h", "1d"]

    no_inc = parser.parse_args(["--no-incremental"])
    assert no_inc.incremental is False, "--no-incremental must still force full fetch"

    explicit = parser.parse_args(["--incremental"])
    assert explicit.incremental is True


def test_days_flag_reaches_date_calculation():
    """--days N must translate into since_ms = now - N days (UTC)."""
    before = _now_ms()
    since_ms = run_fetch.get_default_since_ms("1h", 730)
    after = _now_ms()
    assert before - 730 * DAY_MS - 1000 <= since_ms <= after - 730 * DAY_MS + 1000, \
        "since_ms must be ~now - 730d"


# =====================================================================
# 2. --days 30 / 90 / 730 request approximately 30 / 90 / 730 days
# =====================================================================

def test_days30_requests_approximately_30_days(pipeline):
    t0 = _now_ms()
    storage, mock = pipeline(days=30)
    t1 = _now_ms()

    # The first page requested must start ~30 days ago (within one candle)
    first_since = mock.calls[0]["since"]
    assert t0 - 30 * DAY_MS - TIMEFRAME_MS["1h"] <= first_since <= t1 - 30 * DAY_MS + TIMEFRAME_MS["1h"], \
        f"requested since must be ~now-30d, got {first_since}"

    first_ms, last_ms, span = _db_span_days(storage, "BTC/USDT", "1h")
    count = storage.count_candles("BTC/USDT", "1h")
    assert 29.5 <= span <= 30.2, f"~30 days expected, got {span:.2f}d"
    assert 718 <= count <= 722, f"~720 hourly candles expected, got {count}"
    assert len(mock.calls) == 1, "720 candles fit in one Binance page (<=1000)"
    storage.close()


def test_days90_requests_approximately_90_days(pipeline):
    t0 = _now_ms()
    storage, mock = pipeline(days=90)
    t1 = _now_ms()

    first_since = mock.calls[0]["since"]
    assert t0 - 90 * DAY_MS - TIMEFRAME_MS["1h"] <= first_since <= t1 - 90 * DAY_MS + TIMEFRAME_MS["1h"]

    _, _, span = _db_span_days(storage, "BTC/USDT", "1h")
    count = storage.count_candles("BTC/USDT", "1h")
    assert 89.5 <= span <= 90.2, f"~90 days expected, got {span:.2f}d"
    assert 2158 <= count <= 2162, f"~2160 hourly candles expected, got {count}"
    # 2160 candles at <=1000/request => exactly 3 pages
    assert len(mock.calls) == 3, f"expected 3 pages for 90d of 1h, got {len(mock.calls)}"
    storage.close()


def test_days730_requests_approximately_730_days(pipeline):
    t0 = _now_ms()
    storage, mock = pipeline(days=730)
    t1 = _now_ms()

    first_since = mock.calls[0]["since"]
    assert t0 - 730 * DAY_MS - TIMEFRAME_MS["1h"] <= first_since <= t1 - 730 * DAY_MS + TIMEFRAME_MS["1h"], \
        "the 730d request must reach back ~2 years, not ~30 days"

    first_ms, last_ms, span = _db_span_days(storage, "BTC/USDT", "1h")
    count = storage.count_candles("BTC/USDT", "1h")
    assert 729.5 <= span <= 730.2, f"~730 days expected after fix, got {span:.2f}d (the ~30d regression)"
    assert 17518 <= count <= 17522, f"~17520 hourly candles expected, got {count}"
    # 17520 candles at <=1000/request => exactly 18 pages of pagination
    assert len(mock.calls) == 18, f"expected 18 pages for 730d of 1h, got {len(mock.calls)}"
    assert all(c["limit"] <= 1000 for c in mock.calls), "Binance 1000-candle cap must never be exceeded"
    sinces = [c["since"] for c in mock.calls]
    assert sinces == sorted(sinces), "pagination must advance monotonically"
    storage.close()


def test_days730_supported_by_all_three_timeframes(pipeline):
    """730d is inside Binance spot availability (genesis 2017-08-17) for 1h/4h/1d."""
    for tf, expected_pages, approx_candles in (("1h", 18, 17520), ("4h", 5, 4380), ("1d", 1, 730)):
        storage, mock = pipeline(days=730, timeframe=tf)
        _, _, span = _db_span_days(storage, "BTC/USDT", tf)
        count = storage.count_candles("BTC/USDT", tf)
        assert 729.0 <= span <= 730.2, f"{tf}: got {span:.2f}d"
        assert abs(count - approx_candles) <= 3, f"{tf}: {count} candles"
        assert len(mock.calls) == expected_pages, f"{tf}: pages {len(mock.calls)} != {expected_pages}"
        all_ts = [r[3] for r in storage.get_candles("BTC/USDT", tf)]
        assert len(all_ts) == len(set(all_ts)), f"{tf}: duplicate timestamps in DB"
        storage.close()


# =====================================================================
# 3. Incremental mode must backfill to --days even when DB already has data
# =====================================================================

def test_incremental_backfills_30_day_db_to_730_days(pipeline, tmp_path):
    """Reproduces the exact user DB state: only ~30 days present, then the
    --days 730 --incremental run must backfill the missing ~700 days AND
    fetch forward, ending near 730 days of coverage."""
    db_path = tmp_path / "seeded.db"
    tf_ms = TIMEFRAME_MS["1h"]
    now = _now_ms()
    seed_first = (now - 30 * DAY_MS) // tf_ms * tf_ms
    seed = [[seed_first + i * tf_ms, 100.0, 110.0, 90.0, 105.0, 10.0] for i in range(720)]
    seeder = SQLiteStorage(db_path=db_path)
    seeder.insert_ohlcv("binance", "BTC/USDT", "1h", seed)
    seeder_first, seeder_last = seeder.get_date_range("BTC/USDT", "1h")
    assert (seeder_last - seeder_first) / DAY_MS < 30.1  # starts as the ~30-day DB
    seeder.close()

    storage, mock = pipeline(days=730, db_path=db_path)

    first_ms, last_ms, span = _db_span_days(storage, "BTC/USDT", "1h")
    count = storage.count_candles("BTC/USDT", "1h")
    assert 729.0 <= span <= 730.2, \
        f"incremental run must backfill to ~730d, got {span:.2f}d (the reported ~30d defect)"
    assert 17518 <= count <= 17522
    # A backfill page must have reached back ~730 days
    min_since = min(c["since"] for c in mock.calls)
    assert min_since <= now - 729 * DAY_MS, "no deep backfill request was made"
    storage.close()


def test_incremental_second_run_is_a_no_op(pipeline, tmp_path):
    """Incremental updates stay deterministic: up-to-date DB -> nothing new."""
    db_path = tmp_path / "twice.db"
    storage1, mock1 = pipeline(days=30, db_path=db_path)
    count1 = storage1.count_candles("BTC/USDT", "1h")
    first1, last1 = storage1.get_date_range("BTC/USDT", "1h")
    storage1.close()

    storage2, mock2 = pipeline(days=30, db_path=db_path)
    count2 = storage2.count_candles("BTC/USDT", "1h")
    first2, last2 = storage2.get_date_range("BTC/USDT", "1h")
    storage2.close()

    assert count2 == count1, "second incremental run must not change row count"
    assert first2 == first1 and last2 == last1, "date range must be stable"


# =====================================================================
# 4. Duplicate protection (non-incremental full re-fetch)
# =====================================================================

def test_full_refetch_inserts_no_duplicates(pipeline, tmp_path):
    db_path = tmp_path / "refetch.db"
    s1, _ = pipeline(days=90, incremental=False, db_path=db_path)
    count1 = s1.count_candles("BTC/USDT", "1h")
    s1.close()

    s2, mock2 = pipeline(days=90, incremental=False, db_path=db_path)
    count2 = s2.count_candles("BTC/USDT", "1h")
    cur = s2.conn.cursor()
    cur.execute(
        "SELECT candles_inserted, duplicates_skipped FROM fetch_metadata "
        "WHERE symbol='BTC/USDT' AND timeframe='1h' ORDER BY id DESC LIMIT 1"
    )
    inserted, dups = cur.fetchone()
    s2.close()

    assert count2 == count1, "full re-fetch must be idempotent (PRIMARY KEY dedup)"
    assert inserted == 0, f"re-fetch must insert 0, inserted {inserted}"
    assert dups > 0, "duplicates must be detected and skipped, not silently re-inserted"


# =====================================================================
# 5. Exchange availability is respected (requests beyond genesis are capped)
# =====================================================================

def test_days_beyond_exchange_availability_caps_at_genesis(pipeline):
    """Asking for 4000 days (pre-listing) must yield exactly what the exchange
    has: from 2017-08-17 to now (~3280 days), not fabricated earlier candles."""
    storage, mock = pipeline(days=4000, timeframe="1d")
    first_ms, last_ms, span = _db_span_days(storage, "BTC/USDT", "1d")
    count = storage.count_candles("BTC/USDT", "1d")
    storage.close()

    assert first_ms == BINANCE_SPOT_GENESIS_MS, \
        f"first candle must equal Binance spot genesis 2017-08-17, got {first_ms}"
    # The available span grows by one day per real-world day, so it is asserted
    # against the genesis-to-today distance rather than a hardcoded constant.
    # This preserves the semantic guarantee - requests beyond listing are capped
    # at exchange availability and never fabricate pre-genesis candles - while
    # remaining correct as time moves forward.
    genesis_day = BINANCE_SPOT_GENESIS_MS // DAY_MS
    today_day = int(datetime.now(timezone.utc).timestamp() * 1000) // DAY_MS
    expected_span = today_day - genesis_day
    assert abs(span - expected_span) <= 1, \
        f"availability-capped span should track genesis->today (~{expected_span}d), got {span:.1f}d"
    assert count == int(span) + 1
    # Sanity: every stored candle is within exchange availability
    assert first_ms >= BINANCE_SPOT_GENESIS_MS


# =====================================================================
# 6. No silent candle filling - gaps are detected, reported, preserved
# =====================================================================

def test_exchange_gap_reported_never_filled(pipeline):
    tf_ms = TIMEFRAME_MS["1h"]
    now = _now_ms()
    gap_start = (now - 11 * DAY_MS) // tf_ms * tf_ms
    gap_end = gap_start + 24 * tf_ms  # 24 missing hourly candles (exchange outage)
    mock = MockBinanceKlines(gap_start_ms=gap_start, gap_end_ms=gap_end)

    storage, used_mock = pipeline(days=30, mock=mock)

    rows = storage.get_candles("BTC/USDT", "1h")
    ts_list = [r[3] for r in rows]
    count = storage.count_candles("BTC/USDT", "1h")

    # 1. Nothing fabricated inside the outage window
    assert not any(gap_start <= ts < gap_end for ts in ts_list), \
        "candles were silently fabricated inside the exchange gap"
    # 2. Row count equals what the exchange actually had (~720 - 24)
    assert 692 <= count <= 698, f"expected ~696 candles (720-24), got {count}"
    # 3. The gap surfaces in stored metadata/validation (reported, not hidden)
    cur = storage.conn.cursor()
    cur.execute(
        "SELECT missing_candles_detected FROM fetch_metadata "
        "WHERE symbol='BTC/USDT' AND timeframe='1h' ORDER BY id DESC LIMIT 1"
    )
    (missing_meta,) = cur.fetchone()
    assert missing_meta >= 24, f"24-candle gap must be reported, got {missing_meta}"
    # 4. And the independent validator agrees on the stored series
    candles = [[r[3], r[5], r[6], r[7], r[8], r[9]] for r in rows]
    _, missing_count, gaps = DataValidator("1h").check_missing_candles(candles)
    assert missing_count == 24 and len(gaps) == 1
    storage.close()


# =====================================================================
# 7. UTC timestamps + OHLC validation on the 730d result
# =====================================================================

def test_utc_timestamps_and_ohlc_sanity_730_days(pipeline):
    storage, _ = pipeline(days=730)
    rows = storage.get_candles("BTC/USDT", "1h")
    assert len(rows) >= 17518
    for r in rows[:500] + rows[-500:]:
        ts_ms, ts_utc = r[3], r[4]
        o, h, l, c, v = r[5], r[6], r[7], r[8], r[9]
        assert ts_utc.endswith("+00:00"), f"timestamp_utc must be UTC, got {ts_utc}"
        assert int(datetime.fromisoformat(ts_utc).timestamp() * 1000) == ts_ms
        assert ts_ms % TIMEFRAME_MS["1h"] == 0, "candles must be boundary-aligned UTC"
        assert h >= l and h >= o and h >= c and l <= o and l <= c, "OHLC invariant broken"
        assert o > 0 and h > 0 and l > 0 and c > 0 and v >= 0
    candles = [[r[3], r[5], r[6], r[7], r[8], r[9]] for r in rows]
    ok, invalid_count, _ = DataValidator("1h").check_invalid_ohlc(candles)
    assert ok and invalid_count == 0
    _, missing, _ = DataValidator("1h").check_missing_candles(candles)
    assert missing == 0, "contiguous 730d series must have no gaps"
    storage.close()


# =====================================================================
# 8. Deterministic storage / export
# =====================================================================

def test_export_is_deterministic(pipeline):
    storage, _ = pipeline(days=90)
    p1 = storage.export_csv("BTC/USDT", "1h")
    b1 = p1.read_bytes()
    p2 = storage.export_csv("BTC/USDT", "1h")
    b2 = p2.read_bytes()
    assert b1 == b2, "CSV export derived from SQLite must be byte-identical"
    assert b1.count(b"\n") == storage.count_candles("BTC/USDT", "1h") + 1  # header + rows
    storage.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
