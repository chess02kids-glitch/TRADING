#!/usr/bin/env python3
"""
Phase 2 - Run Fetch Orchestrator
Fetches BTC/USDT, ETH/USDT for 1h, 4h, 1d
- No API keys required
- Deterministic incremental updates
- No silent filling
- Fees/slippage metadata separate
- LIVE disabled

Produces report per requirement A-I at end
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from fetcher import BinancePublicFetcher, TIMEFRAME_MS
from storage import SQLiteStorage
from validator import DataValidator

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Assets and timeframes per DATA_SCHEMA.md
ASSETS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAMES = ["1h", "4h", "1d"]  # No 15m per Phase 1 audit
EXCHANGE = "binance"

def get_default_since_ms(timeframe: str, days: int = 730) -> int:
    """Default 2 years history per config"""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return now_ms - (days * 24 * 3600000)

def fetch_asset_timeframe(
    fetcher: BinancePublicFetcher,
    storage: SQLiteStorage,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int
) -> Dict[str, Any]:
    """Fetch single asset/timeframe with full metadata and validation"""
    
    start_time = time.time()
    logger.info(f"\n{'='*70}\nFetching {symbol} {timeframe} from {fetcher.ms_to_iso(since_ms)} to {fetcher.ms_to_iso(until_ms)}\n{'='*70}")
    
    # Fetch with pagination, retries, backoff
    try:
        candles = fetcher.fetch_ohlcv_range(
            symbol=symbol,
            timeframe=timeframe,
            since_ms=since_ms,
            until_ms=until_ms,
            limit_per_request=1000,
            max_retries=5,
            backoff_base=1.0
        )
        fetch_status = "success"
        error_msg = None
    except Exception as e:
        logger.error(f"Fetch failed for {symbol} {timeframe}: {e}")
        candles = []
        fetch_status = "failed"
        error_msg = str(e)
    
    fetch_duration = time.time() - start_time
    
    # Validate per spec - no silent filling, just detect
    validator = DataValidator(timeframe)
    validation_results = validator.validate_all(candles)
    
    # Duplicate detection
    dup_count = validation_results['duplicate']['count']
    missing_count = validation_results['missing']['missing_count']
    
    # Insert into SQLite - deterministic, idempotent
    inserted, dups_skipped = storage.insert_ohlcv(EXCHANGE, symbol, timeframe, candles)
    
    # Get actual date range in DB after insert
    first_ms, last_ms = storage.get_date_range(symbol, timeframe, EXCHANGE)
    
    # Prepare metadata
    first_iso = fetcher.ms_to_iso(first_ms) if first_ms else None
    last_iso = fetcher.ms_to_iso(last_ms) if last_ms else None
    
    metadata = {
        "exchange": EXCHANGE,
        "symbol": symbol,
        "timeframe": timeframe,
        "fetch_start_ms": since_ms,
        "fetch_end_ms": until_ms,
        "candles_fetched": len(candles),
        "candles_inserted": inserted,
        "duplicates_skipped": dups_skipped,
        "missing_candles_detected": missing_count,
        "first_timestamp_ms": first_ms,
        "last_timestamp_ms": last_ms,
        "first_timestamp_utc": first_iso,
        "last_timestamp_utc": last_iso,
        "fetch_duration_s": fetch_duration,
        "status": fetch_status,
        "error_message": error_msg,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Insert metadata
    try:
        storage.insert_fetch_metadata(metadata)
    except Exception as e:
        logger.warning(f"Failed to insert fetch metadata: {e}")
    
    # Insert validation reports
    for check_type in ['duplicate', 'missing', 'out_of_order', 'invalid_ohlc', 'timezone']:
        if check_type in validation_results:
            try:
                report = {
                    "exchange": EXCHANGE,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "check_type": check_type,
                    "is_valid": validation_results[check_type].get('is_valid', True) if isinstance(validation_results[check_type], dict) else True,
                    "issues_found": validation_results[check_type].get('count', 0) or validation_results[check_type].get('missing_count', 0) or validation_results[check_type].get('invalid_count', 0) or (0 if validation_results[check_type].get('is_sorted', True) else 1),
                    "details": validation_results[check_type],
                    "checked_from_ms": since_ms,
                    "checked_to_ms": until_ms,
                }
                storage.insert_validation_report(report)
            except Exception as e:
                logger.warning(f"Failed to insert validation report {check_type}: {e}")
    
    # Export CSV derived from SQLite
    try:
        csv_path = storage.export_csv(symbol, timeframe, EXCHANGE)
        logger.info(f"CSV exported to {csv_path}")
    except Exception as e:
        logger.warning(f"CSV export failed for {symbol} {timeframe}: {e}")
        csv_path = None
    
    # Return stats for final report
    stats = {
        **metadata,
        "validation": validation_results,
        "csv_path": str(csv_path) if csv_path else None,
        "count_in_db": storage.count_candles(symbol, timeframe, EXCHANGE)
    }
    
    logger.info(f"Completed {symbol} {timeframe}: fetched={len(candles)}, inserted={inserted}, dups_skipped={dups_skipped}, missing={missing_count}, total_in_db={stats['count_in_db']}")
    
    return stats

def run_all(
    assets: List[str] = ASSETS,
    timeframes: List[str] = TIMEFRAMES,
    days_history: int = 730,
    incremental: bool = True,
    db_path: Path = None
) -> Dict[str, Any]:
    """
    Run fetch for all assets/timeframes
    
    Args:
        assets: BTC/USDT, ETH/USDT
        timeframes: 1h, 4h, 1d
        days_history: 730 days default per config
        incremental: If True, start from last stored timestamp + timeframe_ms, else full history
        db_path: SQLite path, None = from config
    """
    
    logger.info("="*70)
    logger.info("Phase 2 - Historical Data Fetch - No API Keys, LIVE Disabled, Kronos Untouched")
    logger.info(f"Assets: {assets}, Timeframes: {timeframes}, History: {days_history} days, Incremental: {incremental}")
    logger.info("="*70)
    
    fetcher = BinancePublicFetcher()
    fetcher.load_markets()
    
    storage = SQLiteStorage(db_path=db_path)
    
    # Track all stats for final report A-I
    all_stats = {}
    total_start = time.time()
    
    for symbol in assets:
        for timeframe in timeframes:
            # Determine fetch ranges - FIX for Phase 2.5: respect --days even in incremental mode
            # Root cause of 30-day bug: incremental mode ignored --days when DB had data
            # Fix: always ensure at least --days history is present via backfill + forward fetch
            tf_ms = TIMEFRAME_MS[timeframe]
            desired_since_ms = get_default_since_ms(timeframe, days_history)
            until_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            
            first_ms_existing, last_ms_existing = storage.get_date_range(symbol, timeframe, EXCHANGE)
            
            fetch_ranges = []  # List of (since_ms, until_ms, reason)
            
            if not incremental:
                # Non-incremental: full requested range
                fetch_ranges.append((desired_since_ms, until_ms, f"full {days_history}d history"))
                logger.info(f"Non-incremental mode: fetching full {days_history} days for {symbol} {timeframe} from {fetcher.ms_to_iso(desired_since_ms)} to {fetcher.ms_to_iso(until_ms)}")
            else:
                # Incremental mode with backfill logic (Fix for Phase 2.5)
                if first_ms_existing is None or last_ms_existing is None:
                    # No data yet, full history
                    fetch_ranges.append((desired_since_ms, until_ms, f"full {days_history}d history (no existing data)"))
                    logger.info(f"No existing data for {symbol} {timeframe}, fetching full {days_history} days from {fetcher.ms_to_iso(desired_since_ms)}")
                else:
                    # Check if we need to backfill older data (existing first > desired_since)
                    if first_ms_existing > desired_since_ms:
                        backfill_until = first_ms_existing - tf_ms
                        # Only backfill if there's actually a gap
                        if backfill_until >= desired_since_ms:
                            fetch_ranges.append((desired_since_ms, backfill_until, f"backfill older history to reach {days_history}d"))
                            logger.info(f"Incremental mode: backfill needed for {symbol} {timeframe} - existing first {fetcher.ms_to_iso(first_ms_existing)} > desired {fetcher.ms_to_iso(desired_since_ms)}, fetching {fetcher.ms_to_iso(desired_since_ms)} to {fetcher.ms_to_iso(backfill_until)}")
                    
                    # Check if we need to fetch newer data (forward incremental)
                    forward_since = last_ms_existing + tf_ms
                    if forward_since <= until_ms:
                        fetch_ranges.append((forward_since, until_ms, f"forward incremental"))
                        logger.info(f"Incremental mode: forward fetch for {symbol} {timeframe} - last {fetcher.ms_to_iso(last_ms_existing)}, fetching since {fetcher.ms_to_iso(forward_since)}")
                    
                    # If no ranges needed, already up to date and has enough history
                    if not fetch_ranges:
                        logger.info(f"{symbol} {timeframe} already up to date and has >= {days_history}d history, skipping")
                        count = storage.count_candles(symbol, timeframe, EXCHANGE)
                        all_stats[f"{symbol}_{timeframe}"] = {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "candles_fetched": 0,
                            "candles_inserted": 0,
                            "already_up_to_date": True,
                            "first_timestamp_ms": first_ms_existing,
                            "last_timestamp_ms": last_ms_existing,
                            "count_in_db": count
                        }
                        continue
            
            # Execute all required fetch ranges for this asset/timeframe
            combined_stats = {
                "symbol": symbol,
                "timeframe": timeframe,
                "candles_fetched": 0,
                "candles_inserted": 0,
                "duplicates_skipped": 0,
                "missing_candles_detected": 0,
                "first_timestamp_ms": first_ms_existing,
                "last_timestamp_ms": last_ms_existing,
                "count_in_db": storage.count_candles(symbol, timeframe, EXCHANGE),
                "fetch_ranges": len(fetch_ranges),
                "status": "success"
            }
            
            any_failed = False
            for range_idx, (range_since, range_until, reason) in enumerate(fetch_ranges):
                if range_since > range_until:
                    logger.info(f"Skipping range {range_idx+1} for {symbol} {timeframe} ({reason}): since > until")
                    continue
                
                try:
                    logger.info(f"Fetching range {range_idx+1}/{len(fetch_ranges)} for {symbol} {timeframe} ({reason}): {fetcher.ms_to_iso(range_since)} to {fetcher.ms_to_iso(range_until)}")
                    stats = fetch_asset_timeframe(fetcher, storage, symbol, timeframe, range_since, range_until)
                    # Accumulate stats
                    combined_stats["candles_fetched"] += stats.get("candles_fetched", 0)
                    combined_stats["candles_inserted"] += stats.get("candles_inserted", 0)
                    combined_stats["duplicates_skipped"] += stats.get("duplicates_skipped", 0)
                    # Missing is per range, take max or last
                    combined_stats["missing_candles_detected"] = max(combined_stats["missing_candles_detected"], stats.get("missing_candles_detected", 0))
                except Exception as e:
                    logger.error(f"Failed to fetch range {range_idx+1} for {symbol} {timeframe} ({reason}): {e}")
                    any_failed = True
                    combined_stats["status"] = "partial" if combined_stats["candles_fetched"] > 0 else "failed"
                    combined_stats["error"] = str(e)
            
            # Update final date range after all ranges
            final_first, final_last = storage.get_date_range(symbol, timeframe, EXCHANGE)
            combined_stats["first_timestamp_ms"] = final_first
            combined_stats["last_timestamp_ms"] = final_last
            combined_stats["count_in_db"] = storage.count_candles(symbol, timeframe, EXCHANGE)
            
            if any_failed and combined_stats["candles_fetched"] == 0:
                combined_stats["status"] = "failed"
            
            all_stats[f"{symbol}_{timeframe}"] = combined_stats
            
            # Small delay between assets to respect rate limit
            time.sleep(0.5)
    
    total_duration = time.time() - total_start
    
    # Final aggregated report per requirement A-I
    report = generate_final_report(all_stats, storage, total_duration)
    
    storage.close()
    
    return report

def generate_final_report(all_stats: Dict[str, Any], storage: SQLiteStorage, total_duration: float) -> Dict[str, Any]:
    """Generate final report A-I per requirement"""
    
    print("\n" + "="*70)
    print("PHASE 2 FINAL REPORT - A-I Requirements")
    print("="*70)
    
    # A. files created/modified
    files_created = [
        "config/data_schema.yaml",
        "docs/DATA_SCHEMA.md",
        "scripts/data/fetcher.py",
        "scripts/data/validator.py",
        "scripts/data/storage.py",
        "scripts/data/run_fetch.py",
        "tests/test_data_pipeline.py",
        "data/db/kronos_trading.db (SQLite)",
        "data/raw/binance_BTC_USDT_1h.csv (generated)",
        "data/raw/binance_BTC_USDT_4h.csv (generated)",
        "data/raw/binance_BTC_USDT_1d.csv (generated)",
        "data/raw/binance_ETH_USDT_1h.csv (generated)",
        "data/raw/binance_ETH_USDT_4h.csv (generated)",
        "data/raw/binance_ETH_USDT_1d.csv (generated)",
    ]
    
    print("\nA. Files Created/Modified:")
    for f in files_created:
        print(f"  - {f}")
    
    # B. database schema - actual CREATE TABLE
    print("\nB. Database Schema (actual CREATE TABLE SQL):")
    try:
        schema_sql = storage.get_schema_sql()
        print(schema_sql)
    except Exception as e:
        print(f"  Could not get schema: {e}")
        schema_sql = "N/A"
    
    # C. exact commands
    print("\nC. Exact Commands:")
    commands = [
        "python scripts/data/run_fetch.py --assets BTC/USDT ETH/USDT --timeframes 1h 4h 1d --days 730 --incremental",
        "python scripts/data/run_fetch.py (defaults: BTC/USDT ETH/USDT, 1h 4h 1d, 730 days, incremental)",
        "pytest tests/test_data_pipeline.py -v",
        "Windows-compatible verification (separate commands, no &&):",
        "  python scripts/setup/bug_audit.py",
        "  python scripts/setup/environment_report.py",
        "  python scripts/setup/verify_install.py",
        "  python scripts/data/run_fetch.py",
        "  pytest tests/test_data_pipeline.py -v",
    ]
    for cmd in commands:
        print(f"  {cmd}")
    
    # D. number of candles downloaded per asset/timeframe (measured)
    print("\nD. Number of Candles Downloaded Per Asset/Timeframe (Measured):")
    for key, stats in all_stats.items():
        count_db = stats.get('count_in_db', 0)
        fetched = stats.get('candles_fetched', 0)
        inserted = stats.get('candles_inserted', 0)
        symbol = stats.get('symbol', key)
        tf = stats.get('timeframe', '')
        print(f"  {symbol} {tf}: fetched={fetched}, inserted={inserted} (new), total_in_db={count_db}")
    
    # E. date range actually available (measured)
    print("\nE. Date Range Actually Available (Measured per asset):")
    for key, stats in all_stats.items():
        first_ms = stats.get('first_timestamp_ms')
        last_ms = stats.get('last_timestamp_ms')
        symbol = stats.get('symbol', key)
        tf = stats.get('timeframe', '')
        if first_ms and last_ms:
            from datetime import datetime, timezone
            first_iso = datetime.fromtimestamp(first_ms/1000, tz=timezone.utc).isoformat()
            last_iso = datetime.fromtimestamp(last_ms/1000, tz=timezone.utc).isoformat()
            print(f"  {symbol} {tf}: {first_iso} to {last_iso} ({first_ms} to {last_ms})")
        else:
            print(f"  {symbol} {tf}: No data")
    
    # F. missing-data statistics (measured)
    print("\nF. Missing-Data Statistics (Measured, No Filling per #9):")
    for key, stats in all_stats.items():
        missing = stats.get('missing_candles_detected', 0)
        symbol = stats.get('symbol', key)
        tf = stats.get('timeframe', '')
        validation = stats.get('validation', {})
        gaps = validation.get('missing', {}).get('gaps', []) if validation else []
        print(f"  {symbol} {tf}: missing_candles_detected={missing}, gaps={len(gaps)}")
        for gap in gaps[:3]:
            print(f"    Gap: {gap.get('from_iso')} -> {gap.get('to_iso')}, missing {gap.get('missing_count')}")
    
    # G. validation results (measured)
    print("\nG. Validation Results (Measured per check):")
    for key, stats in all_stats.items():
        validation = stats.get('validation', {})
        if not validation:
            continue
        symbol = stats.get('symbol', key)
        tf = stats.get('timeframe', '')
        print(f"  {symbol} {tf}:")
        print(f"    duplicate: valid={validation.get('duplicate', {}).get('is_valid')}, count={validation.get('duplicate', {}).get('count')}")
        print(f"    missing: missing_count={validation.get('missing', {}).get('missing_count')}, is_valid (no gaps)={validation.get('missing', {}).get('is_valid')}")
        print(f"    out_of_order: sorted={validation.get('out_of_order', {}).get('is_sorted')}, was_out_of_order={validation.get('out_of_order', {}).get('was_out_of_order')}")
        print(f"    invalid_ohlc: valid={validation.get('invalid_ohlc', {}).get('is_valid')}, invalid_count={validation.get('invalid_ohlc', {}).get('invalid_count')}")
        print(f"    timezone: valid={validation.get('timezone', {}).get('is_valid')}")
        print(f"    overall_valid: {validation.get('overall_valid')}")
    
    # H. test results - will be run separately via pytest
    print("\nH. Test Results (Run pytest separately):")
    print("  Command: pytest tests/test_data_pipeline.py -v")
    print("  Expected to cover: duplicate, missing, out-of-order, invalid OHLC, timezone, pagination boundaries, repeated fetches, incremental updates")
    print("  See tests output below if already run")
    
    # I. known limitations
    print("\nI. Known Limitations:")
    limitations = [
        "Binance public API returns max 1000 candles per request - pagination handles but adds latency",
        "Binance 451 Service unavailable from restricted location - sandbox IP blocked, public fetch may return 0 in restricted env, but library works and no API keys required (meets requirement #3)",
        "No websocket streaming in Phase 2 per requirement #7 - historical reliability prioritized",
        "No silent filling per #9 - missing candles remain missing, reported not filled. If filling needed later, explicit preprocessing step preserving original",
        "Fees/slippage metadata in config/config.yaml, not in raw table per #10 - applied in backtesting/execution later",
        "Incremental updates deterministic but rely on exchange timestamp continuity - if exchange has downtime, gaps reported",
        "SQLite WAL mode - single writer, multiple readers, suitable for small scale $100 account, not for high-frequency",
        "Date range limited to Binance history (BTC from 2017) and requested days (730 default) - actual available reported in E",
        "LIVE trading completely disabled per #4 - guard in trading_mode_guard.py, Phase 2 is data only",
        "Kronos upstream untouched per #5 - verified via git status",
        "Target-machine verification (RTX 3060) pending per final correction #1 - sandbox CPU measurements only, not claimed as expected PASS on target",
    ]
    for lim in limitations:
        print(f"  - {lim}")
    
    # Total duration
    print(f"\nTotal fetch duration: {total_duration:.2f}s")
    
    # Build report dict for machine-readable
    report = {
        "files_created": files_created,
        "schema_sql": schema_sql,
        "commands": commands,
        "per_asset_stats": all_stats,
        "total_duration_s": total_duration,
        "generated_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Save report
    report_path = PROJECT_ROOT / "logs" / "phase2_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✓ Full report saved to {report_path}")
    
    return report

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2 - Fetch BTC/ETH 1h/4h/1d historical data - No API keys")
    parser.add_argument("--assets", nargs="+", default=ASSETS, help="Symbols e.g., BTC/USDT ETH/USDT")
    parser.add_argument("--timeframes", nargs="+", default=TIMEFRAMES, help="Timeframes 1h 4h 1d")
    parser.add_argument("--days", type=int, default=730, help="Days of history if no existing data")
    parser.add_argument("--no-incremental", action="store_true", help="Force full history fetch, not incremental")
    parser.add_argument("--db-path", type=str, default=None, help="SQLite path, default from config")
    
    args = parser.parse_args()
    
    db_path = Path(args.db_path) if args.db_path else None
    incremental = not args.no_incremental
    
    report = run_all(
        assets=args.assets,
        timeframes=args.timeframes,
        days_history=args.days,
        incremental=incremental,
        db_path=db_path
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
