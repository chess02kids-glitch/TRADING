#!/usr/bin/env python3
"""Read-only Phase 2 historical OHLCV audit.

This script audits candle rows only.  ``fetch_metadata.fetch_start_ms`` and
``fetch_end_ms`` are request provenance, not candle timestamps, and are never
used to infer gaps or duplicate candles.

Usage (Windows compatible)::

    python scripts/setup/audit_phase2_db.py --db data/db/kronos_trading.db --days 730

The database is opened with SQLite ``mode=ro`` and this module issues SELECT
queries only.  A reported gap is intentionally not repaired by this tool.
"""

import argparse
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TIMEFRAME_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAMES = ["1h", "4h", "1d"]
TF_HUMAN = {"1h": "1 hour", "4h": "4 hours", "1d": "1 day"}
DAY_MS = TIMEFRAME_MS["1d"]


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def default_db_path() -> Path:
    """Read the configured DB location without requiring PyYAML for auditing."""
    try:
        import yaml
        with (PROJECT_ROOT / "config" / "config.yaml").open(encoding="utf-8") as handle:
            configured = yaml.safe_load(handle).get("data", {}).get("db_path")
        if configured:
            return PROJECT_ROOT / configured
    except Exception:
        pass
    return PROJECT_ROOT / "data" / "db" / "kronos_trading.db"


def _utc_text_matches(timestamp_ms: int, timestamp_utc: object) -> bool:
    """Require an explicitly UTC ISO timestamp representing the stored epoch."""
    if not isinstance(timestamp_utc, str):
        return False
    try:
        parsed = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        return False
    return int(parsed.timestamp() * 1000) == int(timestamp_ms)


def _gap_ranges(timestamps: List[int], timeframe_ms: int) -> List[Dict[str, object]]:
    """Return exact missing candle-open ranges between actual candle rows only."""
    ranges = []
    for previous, following in zip(timestamps, timestamps[1:]):
        difference = following - previous
        if difference > timeframe_ms:
            # Alignment is independently audited.  Floor keeps this report factual
            # even for a corrupt/non-aligned timestamp without inventing a candle.
            missing = max(0, difference // timeframe_ms - 1)
            if missing:
                start_ms = previous + timeframe_ms
                end_ms = following - timeframe_ms
                ranges.append({
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "start_utc": iso(start_ms),
                    "end_utc": iso(end_ms),
                    "missing_count": missing,
                })
    return ranges


def audit_series(conn: sqlite3.Connection, symbol: str, timeframe: str, days: int,
                 now_ms: Optional[int] = None) -> Dict[str, object]:
    """Audit one canonical ``binance/symbol/timeframe`` candle series.

    ``now_ms`` exists for deterministic tests. It is deliberately unrelated to
    historical fetch request metadata.
    """
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    tf_ms = TIMEFRAME_MS[timeframe]
    reference_ms = now_ms if now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
    cur = conn.cursor()
    cur.execute(
        "SELECT timestamp_ms, timestamp_utc, open, high, low, close, volume "
        "FROM ohlcv_raw WHERE exchange=? AND symbol=? AND timeframe=? "
        "ORDER BY timestamp_ms ASC",
        ("binance", symbol, timeframe),
    )
    rows = cur.fetchall()
    if not rows:
        return {"symbol": symbol, "timeframe": timeframe, "empty": True,
                "pass": False, "reasons": ["no candle rows"]}

    timestamps = [int(row[0]) for row in rows]
    duplicate_groups = sum(1 for _ts in set(timestamps) if timestamps.count(_ts) > 1)
    # Above stays valid when auditing a malformed/non-PK legacy DB. Production's
    # canonical PK prevents it; the query itself deliberately excludes metadata.
    invalid_ohlc = 0
    non_utc = 0
    for timestamp_ms, timestamp_utc, open_, high, low, close, volume in rows:
        values = (open_, high, low, close, volume)
        valid_numbers = all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
        valid_ohlc = (valid_numbers and open_ > 0 and high > 0 and low > 0 and close > 0
                      and volume >= 0 and high >= low and high >= open_ and high >= close
                      and low <= open_ and low <= close)
        if not valid_ohlc:
            invalid_ohlc += 1
        if not _utc_text_matches(timestamp_ms, timestamp_utc):
            non_utc += 1

    first_ms, last_ms = timestamps[0], timestamps[-1]
    misaligned = sum(timestamp % tf_ms != 0 for timestamp in timestamps)
    gap_ranges = _gap_ranges(timestamps, tf_ms)
    missing_count = sum(item["missing_count"] for item in gap_ranges)
    # A requested range can start at an arbitrary instant. A valid first open can
    # therefore be almost one full candle after it; compare only candle coverage.
    # This does not inspect fetch_metadata request bounds.
    min_acceptable_span_ms = max(0, days * DAY_MS - tf_ms)
    actual_span_ms = last_ms - first_ms
    forming_last = last_ms <= reference_ms < last_ms + tf_ms
    closed_rows = sum(timestamp + tf_ms <= reference_ms for timestamp in timestamps)
    reasons: List[str] = []
    if actual_span_ms < min_acceptable_span_ms:
        reasons.append(f"candle span {actual_span_ms / DAY_MS:.2f}d is shorter than requested ~{days}d")
    if missing_count:
        reasons.append(f"{missing_count} genuinely missing candle(s) in {len(gap_ranges)} range(s)")
    if duplicate_groups:
        reasons.append(f"{duplicate_groups} duplicate canonical timestamp group(s)")
    if invalid_ohlc:
        reasons.append(f"{invalid_ohlc} invalid OHLC row(s)")
    if misaligned:
        reasons.append(f"{misaligned} timestamp(s) not aligned to {timeframe}")
    if non_utc:
        reasons.append(f"{non_utc} timestamp_utc value(s) are non-UTC or inconsistent with timestamp_ms")

    return {
        "symbol": symbol, "timeframe": timeframe, "empty": False,
        "rows": len(rows), "first_ms": first_ms, "last_ms": last_ms,
        "first_utc": iso(first_ms), "last_utc": iso(last_ms),
        "actual_span_days": actual_span_ms / DAY_MS,
        "expected_interval_ms": tf_ms, "expected_interval": TF_HUMAN[timeframe],
        "missing_candle_count": missing_count, "missing_ranges": gap_ranges,
        "duplicate_timestamp_groups": duplicate_groups, "invalid_ohlc_count": invalid_ohlc,
        "misaligned_timestamp_count": misaligned, "non_utc_count": non_utc,
        "last_candle_currently_forming": forming_last,
        "closed_candle_rows": closed_rows,
        "closed_candle_only_training_available": closed_rows > 0,
        "pass": not reasons, "reasons": reasons or ["all candle-row checks passed"],
    }


def _print_series(result: Dict[str, object], days: int) -> None:
    print("-" * 86)
    print(f"{result['symbol']} {result['timeframe']}")
    if result["empty"]:
        print("  VERDICT: FAIL — no candle rows")
        return
    print(f"  row count:                 {result['rows']}")
    print(f"  first candle:              {result['first_utc']} ({result['first_ms']})")
    print(f"  last candle:               {result['last_utc']} ({result['last_ms']})")
    print(f"  actual span:               {result['actual_span_days']:.2f} days (requested ~{days})")
    print(f"  expected interval:         {result['expected_interval']} ({result['expected_interval_ms']} ms)")
    print(f"  missing candle count:      {result['missing_candle_count']}")
    for gap in result["missing_ranges"]:
        print(f"    missing {gap['start_utc']} -> {gap['end_utc']} ({gap['missing_count']} candle(s))")
    print(f"  duplicate timestamp groups:{result['duplicate_timestamp_groups']}")
    print(f"  invalid OHLC count:        {result['invalid_ohlc_count']}")
    print(f"  misaligned timestamp count:{result['misaligned_timestamp_count']}")
    print(f"  non-UTC count:             {result['non_utc_count']}")
    print(f"  last candle currently forming: {result['last_candle_currently_forming']} (not a failure)")
    print(f"  closed-candle-only training available: {result['closed_candle_only_training_available']} ({result['closed_candle_rows']} row(s))")
    print(f"  VERDICT: {'PASS' if result['pass'] else 'FAIL'} — {'; '.join(result['reasons'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Phase 2 candle database audit")
    parser.add_argument("--db", help="SQLite path (default: config data.db_path)")
    parser.add_argument("--days", type=int, default=730, help="requested approximate history depth")
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be positive")
    db_path = Path(args.db) if args.db else default_db_path()
    print("=" * 86 + f"\nPHASE 2 READ-ONLY CANDLE AUDIT\nDB: {db_path}")
    if not db_path.exists():
        print("FAIL: database not found; no audit was performed.")
        return 2
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        results = [audit_series(conn, symbol, timeframe, args.days)
                   for symbol in SYMBOLS for timeframe in TIMEFRAMES]
        for result in results:
            _print_series(result, args.days)
    finally:
        conn.close()
    passed = all(result["pass"] for result in results)
    print("=" * 86)
    print("OVERALL: " + ("PASS — all candle-row checks passed" if passed else "FAIL — see exact candle-row reasons above"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
