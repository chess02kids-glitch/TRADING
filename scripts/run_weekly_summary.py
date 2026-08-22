#!/usr/bin/env python3
"""
HAR Alert Bot — Weekly Research Summary

Fetches the last 7 days of completed predictions from Supabase, computes
weekly calibration statistics (MAE vs persistence, bias, breakouts,
regime distribution, profit factor), compares against the previous week,
tracks 30-day calibration progress, and sends a comprehensive weekly
summary to Telegram.

Usage:
    python scripts/run_weekly_summary.py

Run automatically via GitHub Actions every Monday at 09:00 UTC.
This is a research monitoring tool only. No trades are placed.
"""

import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Add project root to path when this file is executed directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos_trading.alerts.prediction_logger import (
    DEFAULT_DB_PATH,
    get_pending_predictions,
    get_prediction_history,
    initialize_db,
)
from kronos_trading.alerts.telegram_sender import (
    SendResult,
    TelegramConfig,
    send_message,
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
WEEK_HOURS = 168  # 7 × 24
MIN_WEEKLY_OBS = 24  # minimum completed rows for accuracy statistics

# ─── helpers ───────────────────────────────────────────────────────────────


def _parse_ts(value: Any) -> Optional[datetime]:
    """Parse an ISO8601 UTC timestamp (with or without timezone)."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        logger.warning("Unparseable timestamp %r — skipping row", value)
        return None


def _filter_window(
    rows: List[Dict[str, Any]],
    hours_ago_start: int,
    hours_ago_end: int,
) -> List[Dict[str, Any]]:
    """Keep rows whose ``timestamp`` falls in [now-end, now-start) hours.

    ``hours_ago_start=0, hours_ago_end=168`` means the last 168 hours
    (this week). Windows are half-open so consecutive windows never
    overlap: a row belongs to exactly one week.
    """
    now = datetime.now(timezone.utc)
    start_cutoff = now - timedelta(hours=hours_ago_end)
    end_cutoff = now - timedelta(hours=hours_ago_start)
    out = []
    for row in rows:
        ts = _parse_ts(row.get("timestamp"))
        if ts is None:
            continue
        if start_cutoff <= ts < end_cutoff:
            out.append(row)
    return out


def _mae_stats(chrono: List[Dict[str, Any]]) -> Dict[str, float]:
    """HAR MAE / persistence MAE / bias over chronological rows."""
    n_obs = len(chrono)
    har_errors = [
        float(r["actual_range"]) - float(r["har_predicted_range"])
        for r in chrono
    ]
    har_mae = sum(abs(e) for e in har_errors) / n_obs
    # Persistence: previous hour's HAR prediction predicts this hour's
    # actual. First chronological row has no previous row — skipped.
    persistence_errors = [
        float(chrono[i]["actual_range"])
        - float(chrono[i - 1]["har_predicted_range"])
        for i in range(1, n_obs)
    ]
    persistence_mae = (
        sum(abs(e) for e in persistence_errors) / len(persistence_errors)
        if persistence_errors
        else 0.0
    )
    return {
        "har_mae": har_mae,
        "persistence_mae": persistence_mae,
        "mean_bias": sum(har_errors) / n_obs,
    }


# ─── weekly statistics ─────────────────────────────────────────────────────


def fetch_week_predictions(
    db_url: str,
    asset: str,
    timeframe: str,
    hours_ago_start: int = 0,
    hours_ago_end: int = 168,
) -> List[Dict[str, Any]]:
    """
    Fetch completed predictions within the specified hour window.

    hours_ago_start=0, hours_ago_end=168 means: last 168 hours
    (this week). Returns rows newest-first (matching the underlying
    history query).
    """
    rows = get_prediction_history(
        db_url or DEFAULT_DB_PATH, asset, timeframe, n=99999
    )
    return _filter_window(rows, hours_ago_start, hours_ago_end)


def fetch_pending_window(
    db_url: str,
    asset: str,
    timeframe: str,
    hours_ago_start: int = 0,
    hours_ago_end: int = 168,
) -> List[Dict[str, Any]]:
    """
    Fetch *pending* (not yet completed) predictions within the window.

    Used only to report "Predictions this week" totals alongside the
    completed count. Same window semantics as ``fetch_week_predictions``.
    """
    rows = get_pending_predictions(db_url or DEFAULT_DB_PATH, asset, timeframe)
    return _filter_window(rows, hours_ago_start, hours_ago_end)


def compute_regime_distribution(
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Count rows by regime and express low/medium/high as percentages.

    Rows with a null or unrecognised regime count as "unknown".
    Percentages are relative to the total row count (including unknown).
    """
    counts = {"low": 0, "medium": 0, "high": 0, "unknown": 0}
    for row in rows:
        regime = str(row.get("regime") or "").strip().lower()
        if regime in ("low", "medium", "high"):
            counts[regime] += 1
        else:
            counts["unknown"] += 1
    total = len(rows)
    denom = total if total else 1
    return {
        "low": counts["low"],
        "medium": counts["medium"],
        "high": counts["high"],
        "unknown": counts["unknown"],
        "low_pct": counts["low"] / denom,
        "medium_pct": counts["medium"] / denom,
        "high_pct": counts["high"] / denom,
    }


def compute_weekly_stats(
    rows: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Compute all weekly statistics from completed prediction rows.

    Returns None when fewer than ``MIN_WEEKLY_OBS`` (24) rows, or when
    no usable rows remain after dropping rows with missing values.

    Returned dict:
        n_obs, har_mae, persistence_mae, har_beats_persistence,
        mean_bias, breakout_count, breakout_rate, worst_ratio,
        best_ratio, profit_factor, regime (distribution dict)
    """
    if not rows:
        return None
    usable = [
        r for r in rows
        if r.get("actual_range") is not None
        and r.get("har_predicted_range") is not None
    ]
    if len(usable) < MIN_WEEKLY_OBS:
        return None

    chrono = list(reversed(usable))  # oldest -> newest
    n_obs = len(chrono)
    mae = _mae_stats(chrono)

    breakout_count = sum(1 for r in chrono if r.get("breakout_flag"))
    breakout_rate = breakout_count / n_obs

    # Ratios only where predicted range is strictly positive.
    ratios = []
    for r in chrono:
        predicted = float(r["har_predicted_range"])
        if predicted > 0:
            ratios.append(float(r["actual_range"]) / predicted)
    worst_ratio = max(ratios) if ratios else None
    best_ratio = min(ratios) if ratios else None

    # Profit factor: gross positive errors / |gross negative errors|.
    har_errors = [
        float(r["actual_range"]) - float(r["har_predicted_range"])
        for r in chrono
    ]
    pos_sum = sum(e for e in har_errors if e > 0)
    neg_sum = abs(sum(e for e in har_errors if e < 0))
    profit_factor = (pos_sum / neg_sum) if neg_sum > 0 else None

    return {
        "n_obs": n_obs,
        "har_mae": mae["har_mae"],
        "persistence_mae": mae["persistence_mae"],
        "har_beats_persistence": mae["har_mae"] < mae["persistence_mae"],
        "mean_bias": mae["mean_bias"],
        "breakout_count": breakout_count,
        "breakout_rate": breakout_rate,
        "worst_ratio": worst_ratio,
        "best_ratio": best_ratio,
        "profit_factor": profit_factor,
        "regime": compute_regime_distribution(chrono),
    }


# ─── calibration progress ──────────────────────────────────────────────────


def compute_calibration_day(db_url: str, asset: str, timeframe: str) -> int:
    """Return the one-based day since the market's oldest prediction."""
    try:
        history = get_prediction_history(
            db_url or DEFAULT_DB_PATH, asset, timeframe, n=99999
        )
        if not history:
            return 1
        oldest_ts = _parse_ts(history[-1]["timestamp"])
        if oldest_ts is None:
            return 1
        delta = datetime.now(timezone.utc) - oldest_ts
        return max(1, int(delta.total_seconds() / 86400) + 1)
    except Exception as exc:  # noqa: BLE001 - report must remain best-effort
        logger.warning("Could not compute calibration day: %s", exc)
        return 1


def compute_weeks_beating(
    db_url: str,
    asset: str,
    timeframe: str,
) -> Dict[str, int]:
    """
    Count how many completed weeks HAR beat persistence.

    Completed predictions are split into chronological chunks of
    ``WEEK_HOURS`` rows (one week of 1h bars each). Chunks with fewer
    than ``MIN_WEEKLY_OBS`` rows are skipped (partial weeks). Returns
    ``{"wins": X, "total": Y}``.
    """
    rows = get_prediction_history(
        db_url or DEFAULT_DB_PATH, asset, timeframe, n=99999
    )
    chrono = list(reversed(rows))
    wins = 0
    total = 0
    for i in range(0, len(chrono), WEEK_HOURS):
        chunk = chrono[i:i + WEEK_HOURS]
        if len(chunk) < MIN_WEEKLY_OBS:
            continue
        total += 1
        mae = _mae_stats(chunk)
        if mae["har_mae"] < mae["persistence_mae"]:
            wins += 1
    return {"wins": wins, "total": total}


def _asset_degrading(
    current: Optional[Dict[str, Any]],
    previous: Optional[Dict[str, Any]],
) -> bool:
    """True when this week's MAE is > 1.5x the previous week's MAE."""
    if current is None or previous is None:
        return False
    if current.get("insufficient") or previous.get("insufficient"):
        return False
    return current["har_mae"] > previous["har_mae"] * 1.5


def compute_calibration_progress(
    btc_current: Optional[Dict[str, Any]],
    btc_previous: Optional[Dict[str, Any]],
    eth_current: Optional[Dict[str, Any]],
    eth_previous: Optional[Dict[str, Any]],
    btc_weeks: Dict[str, int],
    eth_weeks: Dict[str, int],
) -> Dict[str, Any]:
    """
    Aggregate the 30-day calibration progress block.

    Returns a dict with btc/eth week win counts, a ``degrading`` flag
    (any asset sharply worse than its previous week) and an overall
    verdict (``"ON TRACK"`` / ``"AT RISK"``).
    """
    degrading = (
        _asset_degrading(btc_current, btc_previous)
        or _asset_degrading(eth_current, eth_previous)
    )
    return {
        "btc_wins": int(btc_weeks.get("wins", 0)),
        "btc_total": int(btc_weeks.get("total", 0)),
        "eth_wins": int(eth_weeks.get("wins", 0)),
        "eth_total": int(eth_weeks.get("total", 0)),
        "degrading": degrading,
        "verdict": "AT RISK" if degrading else "ON TRACK",
    }


# ─── message formatting ────────────────────────────────────────────────────


def _format_asset_section(
    asset: str,
    stats: Optional[Dict[str, Any]],
) -> str:
    """One asset's section of the weekly message."""
    if stats is None:
        return (
            f"━━━ {asset} 1h ━━━\n"
            "Predictions this week: N/A\n"
            "Completed: N/A\n"
            "Insufficient data for accuracy stats"
        )
    if stats.get("insufficient"):
        return (
            f"━━━ {asset} 1h ━━━\n"
            f"Predictions this week: {stats.get('n_predictions', 0)}\n"
            f"Completed: {stats.get('n_completed', stats.get('n_obs', 0))}\n"
            "Insufficient data for accuracy stats"
        )

    beats = "✅ YES" if stats["har_beats_persistence"] else "❌ NO"
    bias = stats["mean_bias"]
    pf = stats.get("profit_factor")
    pf_str = f"{pf:.2f}" if pf is not None else "N/A"
    worst = stats.get("worst_ratio")
    best = stats.get("best_ratio")
    worst_str = f"{worst:.2f}×" if worst is not None else "N/A"
    best_str = f"{best:.2f}×" if best is not None else "N/A"

    regime = stats.get("regime") or {}
    lines = [
        f"━━━ {asset} 1h ━━━",
        f"Predictions this week: {stats.get('n_predictions', stats['n_obs'])}",
        f"Completed: {stats['n_obs']}",
        "",
        "Accuracy:",
        f"  HAR MAE:         {stats['har_mae']:.2f}",
        f"  Persistence MAE: {stats['persistence_mae']:.2f}",
        f"  HAR beats naive: {beats}",
        f"  Mean bias:       {bias:+.1f}",
        f"  Profit factor:   {pf_str}",
        "",
        "Volatility events:",
        f"  Breakouts:    {stats['breakout_count']} ({stats['breakout_rate']:.1%})",
        f"  Worst ratio:  {worst_str}",
        f"  Best ratio:   {best_str}",
        "",
        "Regime distribution:",
        f"  🟢 Low:    {regime.get('low_pct', 0) * 100:.0f}% "
        f"({regime.get('low', 0)} hours)",
        f"  🟡 Medium: {regime.get('medium_pct', 0) * 100:.0f}% "
        f"({regime.get('medium', 0)} hours)",
        f"  🔴 High:   {regime.get('high_pct', 0) * 100:.0f}% "
        f"({regime.get('high', 0)} hours)",
    ]
    if regime.get("unknown", 0) > 0:
        unknown_pct = regime.get("unknown", 0) / max(
            sum(regime.get(k, 0) for k in ("low", "medium", "high", "unknown")), 1
        )
        lines.append(
            f"  ⚪ Unknown: {unknown_pct * 100:.0f}% "
            f"({regime.get('unknown', 0)} hours)"
        )
    return "\n".join(lines)


def _format_week_over_week(
    current: Optional[Dict[str, Any]],
    previous: Optional[Dict[str, Any]],
    asset: str,
) -> str:
    """One asset's week-over-week MAE comparison line."""
    usable = (
        current is not None
        and previous is not None
        and not current.get("insufficient")
        and not previous.get("insufficient")
    )
    if not usable:
        return f"{asset} MAE: First week — no comparison yet"
    trend = "better" if current["har_mae"] < previous["har_mae"] else "worse"
    return (
        f"{asset} MAE: {current['har_mae']:.2f} vs "
        f"{previous['har_mae']:.2f} last week\n         {trend}"
    )


def _stats_empty(stats: Optional[Dict[str, Any]]) -> bool:
    """True when a stats dict represents "no data at all".

    Full stats dicts (from ``compute_weekly_stats``) carry ``n_obs``;
    insufficient-data dicts (built by ``main``) carry ``n_predictions``.
    Either being zero means there is nothing to report yet.
    """
    if stats is None:
        return True
    if stats.get("insufficient"):
        return stats.get("n_predictions", 0) == 0
    return stats.get("n_obs", 0) == 0


def format_weekly_message(
    btc_stats: Optional[Dict[str, Any]],
    eth_stats: Optional[Dict[str, Any]],
    btc_prev: Optional[Dict[str, Any]],
    eth_prev: Optional[Dict[str, Any]],
    calibration_day: int,
    weeks_beating: Dict[str, Any],
) -> str:
    """
    Build the complete Telegram message.

    Never raises. When there is no data at all, returns the
    "calibration started" placeholder. ``weeks_beating`` carries the
    calibration progress block (see ``compute_calibration_progress``).
    """
    try:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        days_remaining = max(0, CALIBRATION_TOTAL_DAYS - calibration_day)

        btc_empty = _stats_empty(btc_stats)
        eth_empty = _stats_empty(eth_stats)
        if btc_empty and eth_empty:
            return (
                "📊 HAR Weekly Research Summary\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Week ending: {date_str} UTC\n"
                f"Calibration: Day {calibration_day} of "
                f"{CALIBRATION_TOTAL_DAYS}\n"
                "Calibration started — first weekly report next Monday\n\n"
                "⚠️ Research tool only. Not financial advice.\n"
                "No trades are placed."
            )

        weeks = weeks_beating or {}
        degrading = weeks.get("degrading", False)
        verdict = weeks.get("verdict", "ON TRACK")
        degrading_str = "YES ⚠️" if degrading else "NO ✅"

        btc_week = _format_week_over_week(btc_stats, btc_prev, "BTC")
        eth_week = _format_week_over_week(eth_stats, eth_prev, "ETH")

        message = (
            "📊 HAR Weekly Research Summary\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Week ending: {date_str} UTC\n"
            f"Calibration: Day {calibration_day} of {CALIBRATION_TOTAL_DAYS}\n"
            "\n"
            f"{_format_asset_section('BTC/USDT', btc_stats)}\n"
            "\n"
            f"{_format_asset_section('ETH/USDT', eth_stats)}\n"
            "\n"
            "━━━ Week-over-Week ━━━\n"
            f"{btc_week}\n"
            f"{eth_week}\n"
            "\n"
            "━━━ 30-Day Calibration Progress ━━━\n"
            f"Day {calibration_day} of {CALIBRATION_TOTAL_DAYS} "
            f"({days_remaining} remaining)\n"
            f"BTC HAR beating naive: ✅ {weeks.get('btc_wins', 0)} / "
            f"{weeks.get('btc_total', 0)} weeks\n"
            f"ETH HAR beating naive: ✅ {weeks.get('eth_wins', 0)} / "
            f"{weeks.get('eth_total', 0)} weeks\n"
            f"Model degrading: {degrading_str}\n"
            f"Overall verdict: {verdict}\n"
            "\n"
            "⚠️ Research tool only. Not financial advice.\n"
            "No trades are placed."
        )
        return message
    except Exception as exc:  # noqa: BLE001 - formatter must never raise
        logger.error("Weekly message formatting failed: %s", exc)
        return (
            "📊 HAR Weekly Research Summary\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "An error occurred while building this report.\n"
            "Check the weekly-summary workflow logs.\n\n"
            "⚠️ Research tool only. Not financial advice.\n"
            "No trades are placed."
        )


def send_weekly_report(
    config: TelegramConfig,
    message: str,
) -> SendResult:
    """Send the weekly report via the existing send_message()."""
    # Plain text: the report contains emoji, × and box characters that
    # should never be interpreted as HTML.
    return send_message(config, message, parse_mode=None)


# ─── entry point ───────────────────────────────────────────────────────────


def main() -> int:
    """Send the weekly summary; return zero on success and one on failure."""
    load_dotenv()
    logger.info("Starting weekly research summary")

    try:
        telegram_config = TelegramConfig.from_env()
    except EnvironmentError as exc:
        logger.error("Telegram config error: %s", exc)
        return 1

    db_url = os.environ.get("SUPABASE_DB_URL") or DEFAULT_DB_PATH
    try:
        initialize_db(db_url)
    except Exception as exc:  # noqa: BLE001 - log a clear fatal DB error
        logger.error("DB init failed: %s", exc)
        return 1

    calibration_day = compute_calibration_day(db_url, ASSETS[0], TIMEFRAME)
    logger.info(
        "Calibration day: %s of %s", calibration_day, CALIBRATION_TOTAL_DAYS
    )

    current: Dict[str, Optional[Dict[str, Any]]] = {}
    previous: Dict[str, Optional[Dict[str, Any]]] = {}
    weeks: Dict[str, Dict[str, int]] = {}

    for asset in ASSETS:
        logger.info("Building weekly stats for %s %s", asset, TIMEFRAME)
        rows = fetch_week_predictions(db_url, asset, TIMEFRAME, 0, WEEK_HOURS)
        pending = fetch_pending_window(
            db_url, asset, TIMEFRAME, 0, WEEK_HOURS
        )
        prev_rows = fetch_week_predictions(
            db_url, asset, TIMEFRAME, WEEK_HOURS, 2 * WEEK_HOURS
        )

        stats = compute_weekly_stats(rows)
        if stats is None:
            stats = {
                "insufficient": True,
                "n_predictions": len(rows) + len(pending),
                "n_completed": len(rows),
            }
        else:
            stats["n_predictions"] = len(rows) + len(pending)
        current[asset] = stats

        previous[asset] = compute_weekly_stats(prev_rows)
        weeks[asset] = compute_weeks_beating(db_url, asset, TIMEFRAME)

        logger.info(
            "%s: %s completed, %s pending this week",
            asset,
            len(rows),
            len(pending),
        )

    progress = compute_calibration_progress(
        current[ASSETS[0]],
        previous[ASSETS[0]],
        current[ASSETS[1]],
        previous[ASSETS[1]],
        weeks[ASSETS[0]],
        weeks[ASSETS[1]],
    )
    message = format_weekly_message(
        current[ASSETS[0]],
        current[ASSETS[1]],
        previous[ASSETS[0]],
        previous[ASSETS[1]],
        calibration_day,
        progress,
    )

    logger.info("Sending weekly summary to Telegram")
    result = send_weekly_report(telegram_config, message)
    if result.success:
        logger.info("Weekly summary sent: message_id=%s", result.message_id)
        return 0

    logger.error("Weekly summary failed: %s", result.error)
    return 1


if __name__ == "__main__":
    sys.exit(main())
