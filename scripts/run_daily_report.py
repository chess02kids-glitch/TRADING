#!/usr/bin/env python3
"""
HAR Alert Bot — Daily Status Report

Fetches prediction history from Supabase, computes calibration statistics,
and sends a formatted daily report to Telegram.

Usage:
    python scripts/run_daily_report.py

Run automatically via GitHub Actions at 08:00 UTC every day.
This is a research monitoring tool only. No trades are placed.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path when this file is executed directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos_trading.alerts.breakout_detector import get_live_calibration
from kronos_trading.alerts.prediction_logger import (
    DEFAULT_DB_PATH,
    get_pending_predictions,
    get_prediction_history,
    initialize_db,
)
from kronos_trading.alerts.telegram_sender import (
    AssetDailyStats,
    TelegramConfig,
    send_daily_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ASSETS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"
CALIBRATION_TOTAL_DAYS = 30


def compute_calibration_day(db_path: str, asset: str, timeframe: str) -> int:
    """Return the one-based day since the market's oldest prediction.

    The value falls back to day 1 when there is no completed history or the
    database cannot be queried.
    """
    try:
        history = get_prediction_history(db_path, asset, timeframe, n=99999)
        if not history:
            return 1

        # History is newest-first, so the final item is the oldest.
        oldest_ts_str = history[-1]["timestamp"]
        oldest_ts = datetime.fromisoformat(oldest_ts_str.replace("Z", "+00:00"))
        if oldest_ts.tzinfo is None:
            oldest_ts = oldest_ts.replace(tzinfo=timezone.utc)

        delta = datetime.now(timezone.utc) - oldest_ts
        return max(1, int(delta.total_seconds() / 86400) + 1)
    except Exception as exc:  # noqa: BLE001 - a report must remain best-effort
        logger.warning("Could not compute calibration day: %s", exc)
        return 1


def build_asset_stats(
    db_path: str,
    asset: str,
    timeframe: str,
    days_running: int,
) -> AssetDailyStats:
    """Fetch one market's history and compute its calibration statistics.

    Database failures are converted to an empty result so one unavailable
    market cannot crash the daily reporting process.
    """
    try:
        history = get_prediction_history(db_path, asset, timeframe, n=720)
        pending = get_pending_predictions(db_path, asset, timeframe)
        completed = len(history)
        n_pending = len(pending)

        return AssetDailyStats(
            asset=asset,
            timeframe=timeframe,
            total_predictions=completed + n_pending,
            completed=completed,
            pending=n_pending,
            calibration=get_live_calibration(history),
            days_running=days_running,
        )
    except Exception as exc:  # noqa: BLE001 - report empty stats, never raise
        logger.error("Failed to build stats for %s %s: %s", asset, timeframe, exc)
        return AssetDailyStats(
            asset=asset,
            timeframe=timeframe,
            total_predictions=0,
            completed=0,
            pending=0,
            calibration=None,
            days_running=days_running,
        )


def main() -> int:
    """Send the daily report; return zero on success and one on failure."""
    load_dotenv()
    logger.info("Starting daily status report")

    try:
        telegram_config = TelegramConfig.from_env()
    except EnvironmentError as exc:
        logger.error("Telegram config error: %s", exc)
        return 1

    db_path = DEFAULT_DB_PATH
    try:
        initialize_db(db_path)
    except Exception as exc:  # noqa: BLE001 - log a clear fatal DB error
        logger.error("DB init failed: %s", exc)
        return 1

    calibration_day = compute_calibration_day(
        db_path, ASSETS[0], TIMEFRAME
    )
    logger.info(
        "Calibration day: %s of %s", calibration_day, CALIBRATION_TOTAL_DAYS
    )

    all_stats = []
    for asset in ASSETS:
        logger.info("Building stats for %s %s", asset, TIMEFRAME)
        stats = build_asset_stats(db_path, asset, TIMEFRAME, calibration_day)
        all_stats.append(stats)
        logger.info(
            "%s: %s completed, %s pending",
            asset,
            stats.completed,
            stats.pending,
        )

    logger.info("Sending daily report to Telegram")
    result = send_daily_report(
        telegram_config,
        all_stats,
        calibration_day,
        CALIBRATION_TOTAL_DAYS,
    )
    
    from kronos_trading.alerts.prediction_logger import log_daily_report
    import dataclasses
    
    report_data = {
        "stats": [dataclasses.asdict(s) for s in all_stats],
        "calibration_day": calibration_day,
        "calibration_total_days": CALIBRATION_TOTAL_DAYS,
    }
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    try:
        log_daily_report(db_path, date_str, report_data)
        logger.info("Saved daily report to database.")
    except Exception as exc:
        logger.error("Failed to save daily report to database: %s", exc)

    if result.success:
        logger.info("Daily report sent: message_id=%s", result.message_id)
        return 0

    logger.error("Daily report failed: %s", result.error)
    return 1


if __name__ == "__main__":
    sys.exit(main())
