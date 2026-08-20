#!/usr/bin/env python3
"""
HAR Alert Bot — Entry Point

Usage:
    python scripts/run_alert_bot.py
    python scripts/run_alert_bot.py --dry-run
    python scripts/run_alert_bot.py --once
    python scripts/run_alert_bot.py --calibrate
    python scripts/run_alert_bot.py --status
    python scripts/run_alert_bot.py --help

This is a PAPER RESEARCH tool only.
No trades are placed at any time.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from unittest import mock

# --- Path bootstrap (same pattern as scripts/data/run_fetch.py) -------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

import kronos_trading.alerts.telegram_sender as ts  # noqa: E402
from kronos_trading.alerts.breakout_detector import get_live_calibration  # noqa: E402
from kronos_trading.alerts.prediction_logger import (  # noqa: E402
    get_pending_predictions,
    get_prediction_history,
    initialize_db,
)
from kronos_trading.alerts.scheduler import (  # noqa: E402
    SchedulerConfig,
    run_calibration_cycle,
    run_forever,
    run_single_cycle,
)
from kronos_trading.alerts.telegram_sender import (  # noqa: E402
    SendResult,
    TelegramConfig,
)

logger = logging.getLogger(__name__)

# Relative to the repo root (mirrors validate_environment()).
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "har_bot.log"
STATUS_ASSETS = ["BTC/USDT", "ETH/USDT"]
STATUS_TIMEFRAME = "1h"

# Third-party loggers that are chatty at INFO level.
QUIET_LOGGERS = ("ccxt", "urllib3", "requests", "hpack", "asyncio")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments (argparse stdlib only)."""
    parser = argparse.ArgumentParser(
        prog="run_alert_bot",
        description="HAR volatility alert bot - paper research only, no trades.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Run one cycle; print messages instead of sending "
                             "to Telegram; still logs to DB.")
    parser.add_argument("--once", action="store_true",
                        help="Run exactly one cycle then exit (no "
                             "startup/shutdown messages).")
    parser.add_argument("--calibrate", action="store_true",
                        help="Send the calibration report for all assets then "
                             "exit (no forecast cycle).")
    parser.add_argument("--status", action="store_true",
                        help="Print DB status and exit (no Telegram).")
    parser.add_argument("--db", default=None,
                        help="Override the default SQLite DB path.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Log level (default: INFO).")
    parser.add_argument("--no-telegram", action="store_true",
                        help="Suppress all Telegram sends; log what would "
                             "have been sent instead.")
    return parser.parse_args(argv)


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger: console + append to ``logs/har_bot.log``.

    Idempotent: handlers are only attached once (repeated calls - e.g. in
    tests - just adjust the level). Third-party loggers are silenced to
    WARNING to reduce noise.
    """
    root = logging.getLogger()
    if not root.handlers:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        root.addHandler(stream)
        file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    root.setLevel(level.upper())
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def validate_environment(
    no_telegram: bool = False,
    project_root: Optional[Path] = None,
) -> List[str]:
    """Check that the environment is ready; return a list of error strings.

    Checks (when ``no_telegram`` is False): TELEGRAM_BOT_TOKEN and
    TELEGRAM_CHAT_ID are set. Always: ``data/`` exists; ``data/db/`` and
    ``logs/`` exist or are created (creation failure is an error).

    Does NOT validate credentials against the API - presence only.
    ``project_root`` is a test hook (defaults to the repo root).
    """
    errors: List[str] = []
    if not no_telegram:
        if not _env_get("TELEGRAM_BOT_TOKEN"):
            errors.append("TELEGRAM_BOT_TOKEN is not set (use --no-telegram "
                          "to run without Telegram)")
        if not _env_get("TELEGRAM_CHAT_ID"):
            errors.append("TELEGRAM_CHAT_ID is not set (use --no-telegram to "
                          "run without Telegram)")

    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    data_dir = root / "data"
    if not data_dir.is_dir():
        errors.append(f"data/ directory not found at {data_dir}")

    db_dir = root / "data" / "db"
    if not db_dir.is_dir():
        try:
            db_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f"could not create {db_dir}: {exc}")

    logs_dir = root / "logs"
    if not logs_dir.is_dir():
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f"could not create {logs_dir}: {exc}")

    return errors


def _env_get(name: str) -> Optional[str]:
    """Read an env var, returning None when missing or blank."""
    value = os.environ.get(name, "")
    return value if value else None


def print_status(db_path: str) -> None:
    """Print a formatted status report to stdout (never sends Telegram).

    Uses ``get_prediction_history`` / ``get_pending_predictions`` for the
    counts and ``get_live_calibration`` (Step 3) for the calibration block -
    it is the only aggregator that also reports the degradation flag.
    """
    bar = "═" * 26
    lines = [
        bar,
        "HAR Alert Bot — Status Report",
        bar,
        "",
        f"Database: {db_path}",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} UTC",
    ]
    for asset in STATUS_ASSETS:
        history = get_prediction_history(db_path, asset, STATUS_TIMEFRAME)
        pending = get_pending_predictions(db_path, asset, STATUS_TIMEFRAME)
        total = len(history) + len(pending)
        completed = len(history)
        last = (pending[-1]["timestamp"] if pending
                else (history[0]["timestamp"] if history else "none"))
        lines += [
            "",
            f"{asset} {STATUS_TIMEFRAME}:",
            f"  Total predictions logged: {total}",
            f"  Completed (actual filled): {completed}",
            f"  Pending (awaiting close): {len(pending)}",
            f"  Last prediction: {last}",
            "",
            "  Calibration (last 720 bars):",
        ]
        cal = get_live_calibration(history)
        if cal is None:
            lines += [
                "    HAR MAE:         N/A",
                "    Persistence MAE: N/A",
                "    HAR beats naive: N/A",
                "    Breakout rate:   N/A",
                "    Degrading:       N/A",
            ]
        else:
            lines += [
                f"    HAR MAE:         {cal.har_mae:.4f}",
                f"    Persistence MAE: {cal.persistence_mae:.4f}",
                f"    HAR beats naive: {'YES' if cal.har_beats_persistence else 'NO'}",
                f"    Breakout rate:   {cal.breakout_rate:.1%}",
                f"    Degrading:       {'YES' if cal.is_degrading else 'NO'}",
            ]
    print("\n".join(lines))


def make_no_telegram_config() -> TelegramConfig:
    """Placeholder credentials for --no-telegram/--dry-run modes.

    The ``send_*`` functions are monkey-patched in those modes, so these
    values are never actually sent to the API.
    """
    return TelegramConfig(bot_token="no-telegram-mode", chat_id="0")


def _dry_send(config: TelegramConfig, text: str, label: str = "DRY RUN",
              **kwargs) -> SendResult:
    """Stand-in for ``telegram_sender.send_message`` in suppressed modes."""
    if text:
        shown = str(text)
        if len(shown) > 200:
            shown = shown[:200] + "..."
        print(f"[{label}] Would send:\n{shown}")
    return SendResult(success=True, message_id=None, error=None, attempts=0)


def _suppress_telegram(label: str):
    """Context manager replacing send_message with a console-printing stub."""
    return mock.patch.object(
        ts, "send_message",
        side_effect=functools.partial(_dry_send, label=label))


def run_dry_cycle(scheduler_config: SchedulerConfig) -> None:
    """Run one full cycle with Telegram sends replaced by console prints.

    Data fetch, HAR computation and DB logging all run normally; only the
    send layer is stubbed.
    """
    with _suppress_telegram("DRY RUN"):
        result = run_single_cycle(make_no_telegram_config(), scheduler_config)
    print("[DRY RUN] Cycle complete.")
    print(f"Assets: {result.assets_processed}")
    print(f"Errors: {result.errors}")
    print(f"Duration: {result.duration_seconds:.2f}s")


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point: parse args, wire everything, route to the right mode.

    Returns an exit code (0 = success, 1 = error). Never calls ``sys.exit``.
    Any unhandled exception is logged as fatal and mapped to exit code 1.
    """
    try:
        return _main(argv)
    except Exception:  # noqa: BLE001 - top-level safety net
        logger.exception("Fatal error")
        return 1


def _main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    load_dotenv()

    config = SchedulerConfig.default()
    if args.db:
        config = dataclasses.replace(config, db_path=args.db)

    # --status is a local read-only report: works without credentials.
    if args.status:
        print_status(config.db_path)
        return 0

    # --dry-run also skips the credential check (spec: verify the pipeline
    # before connecting real Telegram credentials).
    errors = validate_environment(no_telegram=args.no_telegram or args.dry_run)
    if errors:
        for error in errors:
            print(error)
        return 1

    if args.dry_run:
        run_dry_cycle(config)
        return 0

    if args.no_telegram:
        telegram_config = make_no_telegram_config()
    else:
        try:
            telegram_config = TelegramConfig.from_env()
        except EnvironmentError as exc:
            print(f"Config error: {exc}")
            return 1

    if args.calibrate:
        initialize_db(config.db_path)
        if args.no_telegram:
            with _suppress_telegram("NO-TELEGRAM"):
                run_calibration_cycle(telegram_config, config)
        else:
            run_calibration_cycle(telegram_config, config)
        return 0

    if args.once:
        initialize_db(config.db_path)
        if args.no_telegram:
            with _suppress_telegram("NO-TELEGRAM"):
                result = run_single_cycle(telegram_config, config)
        else:
            result = run_single_cycle(telegram_config, config)
        if result.errors:
            logger.warning("Cycle had errors: %s", result.errors)
        return 0

    if args.no_telegram:
        with _suppress_telegram("NO-TELEGRAM"):
            run_forever(telegram_config, config)
    else:
        run_forever(telegram_config, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
