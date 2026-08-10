"""
Phase 2 Final Verification - READ-ONLY database audit.

Audits the actual ingested OHLCV database without modifying anything:
  * opens SQLite with mode=ro (file is never written; WAL untouched)
  * SELECT-only queries

Per symbol x timeframe (BTC/USDT, ETH/USDT x 1h, 4h, 1d) it reports:
  - row count
  - first / last timestamp (ms + UTC ISO)
  - expected candle interval (ms and human-readable)
  - span in days vs the requested history (default 730)
  - number of gaps (missing candles, exchange outages - must NOT be filled)
  - number of duplicate timestamps (must be 0; PRIMARY KEY dedup)
  - number of invalid OHLC rows (must be 0; CHECK constraints also guard)
  - candle-boundary alignment violations (must be 0; catches non-candle ts)
  - non-UTC timestamp_utc strings (must be 0)
  - whether the last candle is the currently-forming one (expected/legitimate)

Usage:
    python scripts/setup/audit_phase2_db.py                      # default db from config
    python scripts/setup/audit_phase2_db.py --db data/db/kronos_trading.db
    python scripts/setup/audit_phase2_db.py --days 730
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

TIMEFRAME_MS = {"1h": 3600000, "4h": 14400000, "1d": 86400000}
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAMES = ["1h", "4h", "1d"]
TF_HUMAN = {"1h": "1 hour", "4h": "4 hours", "1d": "1 day"}
DAY_MS = 86400000


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def default_db_path() -> Path:
    try:
        import yaml
        cfg_path = PROJECT_ROOT / "config" / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            rel = cfg.get("data", {}).get("db_path", "data/db/kronos_trading.db")
            return PROJECT_ROOT / rel
    except Exception:
        pass
    return PROJECT_ROOT / "data" / "db" / "kronos_trading.db"


def audit_series(conn, symbol: str, timeframe: str, days: int) -> dict:
    tf_ms = TIMEFRAME_MS[timeframe]
    cur = conn.cursor()

    cur.execute(
        "SELECT COUNT(*), MIN(timestamp_ms), MAX(timestamp_ms) FROM ohlcv_raw "
        "WHERE exchange='binance' AND symbol=? AND timeframe=?",
        (symbol, timeframe),
    )
    count, first_ms, last_ms = cur.fetchone()
    if not count:
        return {"symbol": symbol, "timeframe": timeframe, "empty": True}

    # Duplicates (PRIMARY KEY should make this impossible, but verify)
    cur.execute(
        "SELECT COUNT(*) FROM (SELECT timestamp_ms FROM ohlcv_raw "
        "WHERE exchange='binance' AND symbol=? AND timeframe=? "
        "GROUP BY timestamp_ms HAVING COUNT(*) > 1)",
        (symbol, timeframe),
    )
    dup_groups = cur.fetchone()[0]

    # Invalid OHLC rows (mirror of CHECK constraints, audited explicitly)
    cur.execute(
        "SELECT COUNT(*) FROM ohlcv_raw WHERE exchange='binance' AND symbol=? AND timeframe=? "
        "AND (high < low OR high < open OR high < close OR low > open OR low > close "
        "OR open <= 0 OR high <= 0 OR low <= 0 OR close <= 0 OR volume < 0)",
        (symbol, timeframe),
    )
    invalid_ohlc = cur.fetchone()[0]

    # Boundary alignment: a stored 1d candle with a non-midnight ts is a defect
    cur.execute(
        "SELECT COUNT(*) FROM ohlcv_raw WHERE exchange='binance' AND symbol=? AND timeframe=? "
        "AND (timestamp_ms % ?) != 0",
        (symbol, timeframe, tf_ms),
    )
    misaligned = cur.fetchone()[0]

    # UTC text check
    cur.execute(
        "SELECT COUNT(*) FROM ohlcv_raw WHERE exchange='binance' AND symbol=? AND timeframe=? "
        "AND timestamp_utc NOT LIKE '%+00:00'",
        (symbol, timeframe),
    )
    non_utc = cur.fetchone()[0]

    # Gaps: count missing candles between consecutive rows (detect only, never fill)
    cur.execute(
        "SELECT timestamp_ms FROM ohlcv_raw WHERE exchange='binance' AND symbol=? AND timeframe=? "
        "ORDER BY timestamp_ms ASC",
        (symbol, timeframe),
    )
    ts = [r[0] for r in cur.fetchall()]
    gaps = 0
    gap_windows = []
    for a, b in zip(ts, ts[1:]):
        if b - a > tf_ms:
            missing = (b - a) // tf_ms - 1
            gaps += missing
            if len(gap_windows) < 5:
                gap_windows.append((iso(a), iso(b), missing))

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    span_days = (last_ms - first_ms) / DAY_MS
    forming_last = (last_ms + tf_ms) > now_ms  # last candle still open?
    desired_min_span = days - 1.5  # since=now-days lands mid-candle for coarse TFs

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "empty": False,
        "rows": count,
        "first_ms": first_ms,
        "last_ms": last_ms,
        "first_utc": iso(first_ms),
        "last_utc": iso(last_ms),
        "span_days": round(span_days, 2),
        "expected_interval_ms": tf_ms,
        "expected_interval": TF_HUMAN[timeframe],
        "gaps_missing_candles": gaps,
        "gap_windows": gap_windows,
        "duplicate_ts_groups": dup_groups,
        "invalid_ohlc_rows": invalid_ohlc,
        "misaligned_rows": misaligned,
        "non_utc_rows": non_utc,
        "last_candle_forming": forming_last,
        "pass": (
            span_days >= desired_min_span
            and dup_groups == 0
            and invalid_ohlc == 0
            and misaligned == 0
            and non_utc == 0
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2 read-only DB audit (SELECT only, mode=ro)")
    ap.add_argument("--db", default=None, help="SQLite path (default: config data.db_path)")
    ap.add_argument("--days", type=int, default=730, help="requested history depth")
    args = ap.parse_args()

    db_path = Path(args.db) if args.db else default_db_path()
    print("=" * 78)
    print("PHASE 2 READ-ONLY DATABASE AUDIT")
    print("=" * 78)
    print(f"DB: {db_path}")
    if not db_path.exists():
        print(f"FAIL: database not found at {db_path}")
        return 2

    size_mb = db_path.stat().st_size / 1e6
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)  # hard read-only
    print(f"Size: {size_mb:.1f} MB | opened read-only (mode=ro) | SELECT-only audit")

    all_pass = True
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            r = audit_series(conn, symbol, tf, args.days)
            print("-" * 78)
            print(f"{r['symbol']} {r['timeframe']}")
            if r.get("empty"):
                print("  FAIL: no rows")
                all_pass = False
                continue
            print(f"  rows:                {r['rows']}")
            print(f"  first timestamp:     {r['first_utc']}  ({r['first_ms']})")
            print(f"  last timestamp:      {r['last_utc']}  ({r['last_ms']})")
            print(f"  span:                {r['span_days']} days (requested ~{args.days})")
            print(f"  expected interval:   {r['expected_interval']} ({r['expected_interval_ms']} ms)")
            print(f"  gaps (missing):      {r['gaps_missing_candles']}", 
                  "" if r['gaps_missing_candles'] == 0 else "(reported, NOT filled - per no-silent-filling rule)")
            for a, b, m in r["gap_windows"]:
                print(f"      gap {a} -> {b}: {m} missing")
            print(f"  duplicate ts groups: {r['duplicate_ts_groups']}")
            print(f"  invalid OHLC rows:   {r['invalid_ohlc_rows']}")
            print(f"  misaligned rows:     {r['misaligned_rows']}  (0 = every ts is a clean candle boundary)")
            print(f"  non-UTC rows:        {r['non_utc_rows']}")
            print(f"  last candle forming: {r['last_candle_forming']}  (True = live-open candle, filterable at training time)")
            print(f"  VERDICT:             {'PASS' if r['pass'] else 'FAIL'}")
            all_pass = all_pass and r["pass"]

    # Latest fetch metadata per series (provenance)
    print("-" * 78)
    print("Latest fetch_metadata per series (request window vs stored window):")
    cur = conn.cursor()
    cur.execute(
        """
        SELECT symbol, timeframe, fetch_start_ms, fetch_end_ms,
               first_timestamp_utc, last_timestamp_utc, status, created_at
        FROM fetch_metadata f1
        WHERE id = (SELECT MAX(id) FROM fetch_metadata f2
                    WHERE f2.symbol = f1.symbol AND f2.timeframe = f1.timeframe)
        ORDER BY symbol, timeframe
        """
    )
    for sym, tf, fs, fe, f_utc, l_utc, status, created in cur.fetchall():
        print(f"  {sym} {tf}: req window {iso(fs)} -> {iso(fe)} | stored {f_utc} -> {l_utc} | {status} | at {created}")
    conn.close()

    print("=" * 78)
    print(f"OVERALL: {'PASS - database matches Phase 2 requirements' if all_pass else 'FAIL - see lines above'}")
    print("=" * 78)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())