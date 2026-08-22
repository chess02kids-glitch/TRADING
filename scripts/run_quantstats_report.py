#!/usr/bin/env python3
"""
HAR Research - QuantStats Report Generator

Reads completed HAR predictions from Supabase, converts forecast accuracy into
a "returns-like" series (prediction accuracy metric - NOT investment returns),
and produces:

* ``reports/btc_har_report.html``   - QuantStats report, BTC/USDT
* ``reports/eth_har_report.html``   - QuantStats report, ETH/USDT
* ``reports/combined_summary.html`` - plain, standalone HTML (no QuantStats)

It also sends a compact Telegram summary and uploads the generated HTML files
as GitHub Actions artifacts (see ``.github/workflows/quantstats_report.yml``).

This is a research/calibration tool only. No trades are placed.

Usage:
    python scripts/run_quantstats_report.py
"""

from __future__ import annotations

import html
import json
import logging
import math
import os
import string
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# --- Path bootstrap (same pattern as the other scripts in this repo) --------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

try:
    import quantstats as qs  # noqa: F401 - only used for HTML generation

    QUANTSTATS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on minimal installations
    qs = None  # type: ignore[assignment]
    QUANTSTATS_AVAILABLE = False
    logger.warning(
        "quantstats not installed. "
        "HTML report will not be generated. "
        "Summary stats still computed."
    )

ASSETS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"
CALIBRATION_TOTAL_DAYS = 30
MIN_ROWS_FOR_SERIES = 10
REPORTS_DIR = "reports"
COMBINED_SUMMARY_FILENAME = "combined_summary.html"
PREDICTION_HISTORY_LIMIT = 50


def _empty_daily_series() -> pd.Series:
    """Return an empty, UTC-indexed pandas Series (daily aggregation format)."""
    return pd.Series(
        dtype=float,
        index=pd.DatetimeIndex([], tz="UTC"),
    )


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse an ISO timestamp into an aware UTC datetime.

    Returns ``None`` when the value cannot be parsed.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        ts = value
    else:
        try:
            ts = pd.to_datetime(value, utc=True).to_pydatetime()
        except Exception:  # noqa: BLE001 - malformed timestamp, skip row
            return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _to_float(value: Any) -> Optional[float]:
    """Convert a numeric-like value to float, returning None on bad/missing data."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _sort_ts(row: Dict[str, Any]) -> datetime:
    """Sort key for rows by parsed timestamp, so lag is always chronological."""
    parsed = _parse_timestamp(row.get("timestamp"))
    if parsed is not None:
        return parsed
    return datetime.min.replace(tzinfo=timezone.utc)


def _is_scoreable_row(row: Dict[str, Any]) -> bool:
    """A row can be scored only when actual range is positive and finite."""
    actual = _to_float(row.get("actual_range"))
    if actual is None or actual <= 0.0:
        return False
    return True


def _abs_prediction_error(row: Dict[str, Any]) -> Optional[float]:
    """Prefer ``abs_prediction_error``; fall back to ``abs(prediction_error)``."""
    value = _to_float(row.get("abs_prediction_error"))
    if value is not None:
        return value
    error = _to_float(row.get("prediction_error"))
    if error is not None:
        return abs(error)
    return None


def compute_har_score(row: Dict[str, Any]) -> Optional[float]:
    """Return the HAR accuracy score for one completed prediction.

    ``har_score = 1 - abs_prediction_error / actual_range``, clipped to
    ``[-1, 1]``. ``1.0`` means a perfect prediction. Returns ``None`` when the
    row cannot be scored (missing/zero/negative actual range).
    """
    if not _is_scoreable_row(row):
        return None
    actual = float(_to_float(row.get("actual_range")))
    abs_error = _abs_prediction_error(row)
    if abs_error is None:
        return None
    score = 1.0 - (abs_error / actual)
    return float(np.clip(score, -1.0, 1.0))


def compute_persistence_score(
    row: Dict[str, Any],
    prev_predicted: Optional[float],
) -> Optional[float]:
    """Return the persistence accuracy score for one completed prediction.

    Persistence uses the previous row's ``har_predicted_range`` as its
    prediction. ``persistence_score = 1 - |actual - prev_predicted| / actual``,
    clipped to ``[-1, 1]``. Returns ``None`` when the row cannot be scored or
    there is no previous prediction.
    """
    if not _is_scoreable_row(row) or prev_predicted is None:
        return None
    actual = float(_to_float(row.get("actual_range")))
    error = abs(actual - float(prev_predicted))
    score = 1.0 - (error / actual)
    return float(np.clip(score, -1.0, 1.0))


def _day(timestamp: datetime) -> datetime:
    """Return the UTC calendar-day start for a Python datetime object."""
    return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)


def _daily_aggregate(pairs: List[tuple]) -> pd.Series:
    """Aggregate (timestamp, value) pairs into one mean value per UTC day."""
    if not pairs:
        return _empty_daily_series()
    index = pd.DatetimeIndex(
        [_day(ts) for ts, _ in pairs],
        tz="UTC",
    )
    series = pd.Series([value for _, value in pairs], index=index, dtype=float)
    return series.groupby(level=0).mean().sort_index()


def predictions_to_returns(rows: List[Dict[str, Any]]) -> pd.Series:
    """Convert HAR prediction errors into a "returns-like" daily series.

    For each completed prediction:
        har_score            = 1 - abs_prediction_error / actual_range
        persistence_score    = persistence accuracy based on the lag-1
                               HAR prediction (previous row's predicted range)
        har_return           = har_score - persistence_score

    All scores are clipped to ``[-1, 1]``. The result is aggregated to one
    mean value per UTC calendar day. Rows with a non-positive actual range are
    skipped safely (they cannot be normalized). The first chronological row has
    no persistence benchmark and is therefore excluded.

    Returns an empty series when fewer than ``MIN_ROWS_FOR_SERIES`` rows are
    supplied.
    """
    if not rows or len(rows) < MIN_ROWS_FOR_SERIES:
        return _empty_daily_series()

    rows_asc = sorted(rows, key=_sort_ts)
    pairs: List[tuple] = []
    prev_predicted: Optional[float] = None

    for row in rows_asc:
        timestamp = _parse_timestamp(row.get("timestamp"))
        if timestamp is None:
            continue

        # Use the previous row's prediction as the persistence benchmark,
        # then update the lag for the next row. Never use the current row.
        har_score = compute_har_score(row)
        persistence_score = compute_persistence_score(row, prev_predicted)
        if har_score is not None and persistence_score is not None:
            pairs.append((timestamp, float(har_score - persistence_score)))

        predicted = _to_float(row.get("har_predicted_range"))
        if predicted is not None:
            prev_predicted = predicted

    result = _daily_aggregate(pairs)
    if len(result) == 0:
        return _empty_daily_series()
    return result


def predictions_to_benchmark(rows: List[Dict[str, Any]]) -> pd.Series:
    """Convert persistence predictions into a benchmark "returns-like" series.

    The benchmark uses the previous row's ``har_predicted_range`` as its
    prediction, so it is directly comparable to ``predictions_to_returns``.

    Returns an empty series when fewer than ``MIN_ROWS_FOR_SERIES`` rows are
    supplied, or when no comparable rows remain after lag alignment.
    """
    if not rows or len(rows) < MIN_ROWS_FOR_SERIES:
        return _empty_daily_series()

    rows_asc = sorted(rows, key=_sort_ts)
    pairs: List[tuple] = []
    prev_predicted: Optional[float] = None

    for row in rows_asc:
        timestamp = _parse_timestamp(row.get("timestamp"))
        if timestamp is None:
            continue

        # Use the previous row's prediction, then update the lag.
        persistence_score = compute_persistence_score(row, prev_predicted)
        if persistence_score is not None:
            pairs.append((timestamp, float(persistence_score)))

        predicted = _to_float(row.get("har_predicted_range"))
        if predicted is not None:
            prev_predicted = predicted

    result = _daily_aggregate(pairs)
    if len(result) == 0:
        return _empty_daily_series()
    return result


def fetch_all_predictions(
    db_url: str,
    asset: str = "BTC/USDT",
    timeframe: str = "1h",
) -> List[Dict[str, Any]]:
    """Fetch ALL completed predictions from Supabase for the given asset.

    Completed means ``actual_range IS NOT NULL``. Rows are returned in
    chronological (timestamp ASC) order.

    This function never raises: database/network/configuration errors are
    logged and converted to an empty list.
    """
    if not db_url:
        logger.warning("fetch_all_predictions: SUPABASE_DB_URL is empty")
        return []

    asset = asset.strip().upper()
    timeframe = timeframe.strip().lower()

    try:
        import psycopg
        from psycopg.rows import dict_row

        conn = psycopg.connect(
            db_url,
            autocommit=True,
            row_factory=dict_row,
            prepare_threshold=None,
            keepalives=1,
            keepalives_idle=5,
            keepalives_interval=2,
            keepalives_count=3,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM public.har_predictions
                    WHERE asset = %s
                      AND timeframe = %s
                      AND actual_range IS NOT NULL
                    ORDER BY "timestamp" ASC
                    """,
                    (asset, timeframe),
                )
                rows = [dict(row) for row in cur.fetchall()]
                logger.info(
                    "fetch_all_predictions: %s %s -> %s completed rows",
                    asset,
                    timeframe,
                    len(rows),
                )
                return rows
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - fetch must never raise
        logger.warning(
            "fetch_all_predictions failed for %s %s: %s", asset, timeframe, exc
        )
        return []


def fetch_both_assets(
    db_url: str,
    timeframe: str = "1h",
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch completed predictions for both BTC/USDT and ETH/USDT."""
    return {
        "BTC/USDT": fetch_all_predictions(db_url, "BTC/USDT", timeframe),
        "ETH/USDT": fetch_all_predictions(db_url, "ETH/USDT", timeframe),
    }


def compute_summary_stats(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Compute calibration summary statistics without QuantStats.

    Returns ``None`` for empty/unusable input. Otherwise returns a dict with:
        n_predictions, n_days, har_mae, persistence_mae, improvement_pct,
        har_beats, breakout_count, breakout_rate, regime_counts,
        best_day_error, worst_day_error, mean_bias, calibration_day.
    """
    if not rows:
        return None

    usable: List[Dict[str, Any]] = []
    parseable: List[tuple] = []
    for row in rows:
        actual = _to_float(row.get("actual_range"))
        predicted = _to_float(row.get("har_predicted_range"))
        timestamp = _parse_timestamp(row.get("timestamp"))
        if actual is None or predicted is None or timestamp is None:
            continue
        usable.append(row)
        parseable.append((timestamp, row))

    if not usable:
        return None

    chrono = [row for _, row in sorted(parseable, key=lambda pair: pair[0])]
    n_predictions = len(chrono)

    har_errors: List[float] = []
    for row in chrono:
        actual = float(_to_float(row.get("actual_range")))
        predicted = float(_to_float(row.get("har_predicted_range")))
        har_errors.append(actual - predicted)
    har_mae = float(np.mean(np.abs(np.asarray(har_errors))))

    persistence_errors: List[float] = []
    for i in range(1, n_predictions):
        actual = float(_to_float(chrono[i].get("actual_range")))
        prev_pred = float(_to_float(chrono[i - 1].get("har_predicted_range")))
        persistence_errors.append(actual - prev_pred)

    if persistence_errors:
        persistence_mae = float(np.mean(np.abs(np.asarray(persistence_errors))))
    else:
        persistence_mae = None

    if persistence_mae is not None:
        har_beats = bool(har_mae < persistence_mae)
        if persistence_mae > 0.0:
            improvement_pct = float(
                (persistence_mae - har_mae) / persistence_mae * 100.0
            )
        else:
            improvement_pct = None
    else:
        har_beats = None
        improvement_pct = None

    breakout_count = int(
        sum(1 for row in chrono if bool(row.get("breakout_flag")))
    )
    breakout_rate = float(breakout_count / n_predictions)

    regime_counts = {"low": 0, "medium": 0, "high": 0}
    for row in chrono:
        regime = str(row.get("regime") or "").strip().lower()
        if regime in regime_counts:
            regime_counts[regime] += 1

    mean_bias = float(np.mean(np.asarray(har_errors)))

    # Daily best/worst: daily mean absolute prediction error.
    daily_abs_errors: Dict[Any, List[float]] = {}
    for timestamp, row in parseable:
        actual = float(_to_float(row.get("actual_range")))
        predicted = float(_to_float(row.get("har_predicted_range")))
        day = _day(timestamp)
        daily_abs_errors.setdefault(day, []).append(abs(actual - predicted))

    best_day_error: Optional[float] = None
    worst_day_error: Optional[float] = None
    if daily_abs_errors:
        daily_means = [
            float(np.mean(np.asarray(errors)))
            for errors in daily_abs_errors.values()
        ]
        best_day_error = min(daily_means)
        worst_day_error = max(daily_means)

    timestamps = [timestamp for timestamp, _ in parseable]
    first_ts = min(timestamps)
    now = datetime.now(timezone.utc)
    elapsed_days = int((now - first_ts).total_seconds() // 86400)
    calibration_day = max(1, elapsed_days + 1)

    days = sorted({_day(ts) for ts in timestamps})
    n_days = len(days)

    return {
        "n_predictions": n_predictions,
        "n_days": n_days,
        "har_mae": har_mae,
        "persistence_mae": persistence_mae,
        "improvement_pct": improvement_pct,
        "har_beats": har_beats,
        "breakout_count": breakout_count,
        "breakout_rate": breakout_rate,
        "regime_counts": regime_counts,
        "best_day_error": best_day_error,
        "worst_day_error": worst_day_error,
        "mean_bias": mean_bias,
        "calibration_day": calibration_day,
    }


def _to_naive_utc(series: pd.Series) -> pd.Series:
    """Convert a tz-aware UTC series to naive UTC for QuantStats.

    QuantStats' internal date comparisons do not handle tz-aware indexes on
    some pandas versions. The public conversion functions still return
    tz-aware indexes; only the QuantStats emitter normalizes them.
    """
    if isinstance(series, pd.Series) and series.index.tz is not None:
        series = series.copy()
        series.index = series.index.tz_convert(None)
    return series


def _emit_quantstats_html(
    returns: pd.Series,
    benchmark: pd.Series,
    output: str,
    title: str,
    benchmark_title: str,
) -> None:
    """Thin wrapper around ``quantstats.reports.html`` for mocking in tests."""
    import quantstats as qs  # noqa: F401

    returns = _to_naive_utc(returns)
    benchmark = _to_naive_utc(benchmark)

    kwargs = {
        "output": output,
        "title": title,
        "benchmark_title": benchmark_title,
    }
    if len(benchmark) > 0:
        kwargs["benchmark"] = benchmark
    qs.reports.html(returns, **kwargs)


def _esc(value: Any) -> str:
    """HTML-escape a value for safe inclusion in the standalone summary."""
    if value is None:
        return "N/A"
    return html.escape(str(value))


def _fmt(value: Optional[float], precision: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def _non_empty_series(series: pd.Series) -> bool:
    return isinstance(series, pd.Series) and len(series) > 0


def _metrics_from_returns(returns: pd.Series) -> Dict[str, Any]:
    """Minimal metrics used by the combined summary when rows/stats are absent."""
    if not _non_empty_series(returns):
        return {
            "n_predictions": None,
            "n_days": 0,
            "har_mae": None,
            "persistence_mae": None,
            "improvement_pct": None,
            "har_beats": None,
            "breakout_count": None,
            "breakout_rate": None,
            "mean_bias": None,
            "best_day_error": None,
            "worst_day_error": None,
            "regime_counts": {"low": 0, "medium": 0, "high": 0},
            "calibration_day": None,
        }
    return {
        "n_predictions": None,
        "n_days": int(len(returns)),
        "har_mae": None,
        "persistence_mae": None,
        "improvement_pct": None,
        "har_beats": None,
        "breakout_count": None,
        "breakout_rate": None,
        "mean_bias": float(returns.mean()) if len(returns) else None,
        "best_day_error": None,
        "worst_day_error": None,
        "regime_counts": {"low": 0, "medium": 0, "high": 0},
        "calibration_day": None,
    }


def _metric_cards(stats: Dict[str, Any]) -> str:
    """Build the side-by-side metric cards for the combined summary."""
    beats = stats.get("har_beats")
    beats_color = "good" if beats else "bad"
    beats_text = "✅ YES" if beats else ("❌ NO" if beats is not None else "N/A")
    improvement = stats.get("improvement_pct")
    improvement_text = (
        "N/A" if improvement is None else f"{improvement:+.1f}%"
    )
    breakout_rate = stats.get("breakout_rate")
    breakout_text = (
        "N/A" if breakout_rate is None else f"{breakout_rate:.1%}"
    )
    return "".join(
        [
            '<div class="metric">',
            "<h3>Predictions</h3>",
            _esc(stats.get("n_predictions")),
            "</div>",
            '<div class="metric">',
            "<h3>Days</h3>",
            _esc(stats.get("n_days")),
            "</div>",
            '<div class="metric">',
            "<h3>HAR MAE</h3>",
            _fmt(stats.get("har_mae")),
            "</div>",
            '<div class="metric">',
            "<h3>Persistence MAE</h3>",
            _fmt(stats.get("persistence_mae")),
            "</div>",
            '<div class="metric">',
            "<h3>Improvement</h3>",
            improvement_text,
            "</div>",
            '<div class="metric">',
            "<h3>HAR beats naive</h3>",
            f'<span class="{beats_color}">{beats_text}</span>',
            "</div>",
            '<div class="metric">',
            "<h3>Breakouts</h3>",
            f"{_esc(stats.get('breakout_count'))} ({breakout_text})",
            "</div>",
            '<div class="metric">',
            "<h3>Mean bias</h3>",
            _fmt(stats.get("mean_bias"), 1),
            "</div>",
        ]
    )


def _regime_json(
    btc_rows: List[Dict[str, Any]],
    eth_rows: List[Dict[str, Any]],
) -> str:
    counts = {"low": 0, "medium": 0, "high": 0}
    for row in list(btc_rows or []) + list(eth_rows or []):
        regime = str(row.get("regime") or "").strip().lower()
        if regime in counts:
            counts[regime] += 1
    return json.dumps(counts)


def _history_rows_html(
    btc_rows: List[Dict[str, Any]],
    eth_rows: List[Dict[str, Any]],
) -> str:
    combined = list(btc_rows or []) + list(eth_rows or [])
    combined.sort(key=_sort_ts, reverse=True)
    recent = combined[:PREDICTION_HISTORY_LIMIT]
    if not recent:
        return "<tr><td colspan='8'>No completed predictions yet.</td></tr>"

    rows = []
    for row in recent:
        timestamp = row.get("timestamp", "")
        asset = row.get("asset", "")
        predicted = _fmt(_to_float(row.get("har_predicted_range")))
        actual = _fmt(_to_float(row.get("actual_range")))
        error = _fmt(_to_float(row.get("prediction_error")), 1)
        abs_error = _fmt(_to_float(row.get("abs_prediction_error")), 1)
        breakout = (
            "✅" if bool(row.get("breakout_flag"))
            else "—"
        )
        regime = _esc(row.get("regime") or "N/A")
        rows.append(
            f"<tr><td>{_esc(timestamp)}</td><td>{_esc(asset)}</td>"
            f"<td>{predicted}</td><td>{actual}</td><td>{error}</td>"
            f"<td>{abs_error}</td><td>{breakout}</td><td>{regime}</td></tr>"
        )
    return "".join(rows)


def _breakout_rows_html(
    btc_rows: List[Dict[str, Any]],
    eth_rows: List[Dict[str, Any]],
) -> str:
    combined = [
        row
        for row in list(btc_rows or []) + list(eth_rows or [])
        if bool(row.get("breakout_flag"))
    ]
    combined.sort(key=_sort_ts, reverse=True)
    recent = combined[:100]
    if not recent:
        return "<tr><td colspan='6'>No breakout events yet.</td></tr>"

    rows = []
    for row in recent:
        timestamp = row.get("timestamp", "")
        asset = row.get("asset", "")
        predicted = _fmt(_to_float(row.get("har_predicted_range")))
        actual = _fmt(_to_float(row.get("actual_range")))
        error = _fmt(_to_float(row.get("prediction_error")), 1)
        regime = _esc(row.get("regime") or "N/A")
        rows.append(
            f"<tr><td>{_esc(timestamp)}</td><td>{_esc(asset)}</td>"
            f"<td>{predicted}</td><td>{actual}</td><td>{error}</td>"
            f"<td>{regime}</td></tr>"
        )
    return "".join(rows)


def _generate_combined_summary(
    btc_stats: Optional[Dict[str, Any]],
    eth_stats: Optional[Dict[str, Any]],
    btc_rows: Optional[List[Dict[str, Any]]],
    eth_rows: Optional[List[Dict[str, Any]]],
    btc_returns: pd.Series,
    eth_returns: pd.Series,
    calibration_day: Optional[int],
    total_days: int,
    output_dir: str,
) -> Optional[str]:
    """Generate the standalone ``combined_summary.html`` (no QuantStats)."""
    try:
        btc_stats = btc_stats or _metrics_from_returns(btc_returns)
        eth_stats = eth_stats or _metrics_from_returns(eth_returns)

        if calibration_day is None:
            calibration_day = btc_stats.get("calibration_day") or eth_stats.get(
                "calibration_day"
            ) or 1
        calibration_day = int(calibration_day)
        pct = min(100.0, max(0.0, calibration_day / total_days * 100.0))

        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        breakout_rows = _breakout_rows_html(
            btc_rows or [], eth_rows or []
        )
        history_rows = _history_rows_html(btc_rows or [], eth_rows or [])
        regime_data = _regime_json(btc_rows or [], eth_rows or [])

        template = string.Template(
            """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>HAR Research Report</title>
  <style>
    body { font-family: monospace;
           background: #1a1a1a; color: #e0e0e0;
           max-width: 1200px; margin: 0 auto;
           padding: 20px; }
    .metric { display: inline-block;
              margin: 10px; padding: 15px;
              background: #2a2a2a;
              border-radius: 8px; }
    .good { color: #00ff88; }
    .bad { color: #ff4444; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 8px; border: 1px solid #444; }
    th { background: #2a2a2a; }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <h1>📊 HAR Research Platform Report</h1>
  <p>Generated: $generated | Day $calibration_day of $total_days</p>

  <h2>Calibration Progress</h2>
  <div style="background:#333; border-radius:10px;
              height:30px; width:100%;">
    <div style="background:#00ff88;
                border-radius:10px;
                height:30px; width:$pct%;
                text-align:center;
                line-height:30px;">
      Day $calibration_day/$total_days ($pct_label%)
    </div>
  </div>

  <h2>Performance Summary</h2>
  <h3>BTC/USDT 1h</h3>
  $btc_metrics
  <h3>ETH/USDT 1h</h3>
  $eth_metrics

  <h2>Prediction History</h2>
  <table>
    <thead>
      <tr>
        <th>Timestamp</th><th>Asset</th><th>Predicted</th>
        <th>Actual</th><th>Error</th><th>|Error|</th>
        <th>Breakout</th><th>Regime</th>
      </tr>
    </thead>
    <tbody>
      $history_rows
    </tbody>
  </table>

  <h2>Breakout Events</h2>
  <table>
    <thead>
      <tr>
        <th>Timestamp</th><th>Asset</th><th>Predicted</th>
        <th>Actual</th><th>Error</th><th>Regime</th>
      </tr>
    </thead>
    <tbody>
      $breakout_rows
    </tbody>
  </table>

  <h2>Regime Distribution</h2>
  <canvas id="regimeChart" width="400" height="400"></canvas>
  <script>
    const regimeData = $regime_data;
    const ctx = document.getElementById("regimeChart").getContext("2d");
    new Chart(ctx, {
      type: "pie",
      data: {
        labels: ["Low", "Medium", "High"],
        datasets: [{
          data: [regimeData.low, regimeData.medium, regimeData.high],
          backgroundColor: ["#00ff88", "#ffcc00", "#ff4444"]
        }]
      },
      options: {
        responsive: true,
        plugins: {
          title: { display: true, text: "Regime Distribution",
                   color: "#e0e0e0" },
          legend: { labels: { color: "#e0e0e0" } }
        }
      }
    });
  </script>

  <p><em>Prediction-accuracy proxy metric. Not investment returns.
     Research tool only. Not financial advice.</em></p>
</body>
</html>
"""
        )

        os.makedirs(output_dir, exist_ok=True)
        output_path = str(
            Path(output_dir) / COMBINED_SUMMARY_FILENAME
        )
        content = template.substitute(
            generated=generated,
            calibration_day=calibration_day,
            total_days=total_days,
            pct=f"{pct:.0f}",
            pct_label=f"{pct:.0f}",
            btc_metrics=_metric_cards(btc_stats),
            eth_metrics=_metric_cards(eth_stats),
            history_rows=history_rows,
            breakout_rows=breakout_rows,
            regime_data=regime_data,
        )
        Path(output_path).write_text(content, encoding="utf-8")
        logger.info("Combined summary generated: %s", output_path)
        return output_path
    except Exception as exc:  # noqa: BLE001 - report must be best-effort
        logger.error("Could not generate combined summary: %s", exc)
        return None


def generate_html_report(
    btc_returns: pd.Series,
    eth_returns: pd.Series,
    btc_benchmark: pd.Series,
    eth_benchmark: pd.Series,
    output_dir: str = REPORTS_DIR,
    btc_rows: Optional[List[Dict[str, Any]]] = None,
    eth_rows: Optional[List[Dict[str, Any]]] = None,
    btc_stats: Optional[Dict[str, Any]] = None,
    eth_stats: Optional[Dict[str, Any]] = None,
    calibration_day: Optional[int] = None,
) -> Dict[str, str]:
    """Generate QuantStats HTML reports plus the standalone combined summary.

    Creates (in ``output_dir``):
        btc_har_report.html         - QuantStats report when available
        eth_har_report.html         - QuantStats report when available
        combined_summary.html       - always generated when the directory is
                                      writable

    QuantStats reports are skipped gracefully when ``quantstats`` is not
    installed. This function never raises; it returns ``{}`` on error.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        results: Dict[str, str] = {}

        if QUANTSTATS_AVAILABLE:
            btc_enough = _non_empty_series(btc_returns) and (
                len(btc_returns) >= MIN_ROWS_FOR_SERIES
                or (btc_rows is not None and len(btc_rows) >= MIN_ROWS_FOR_SERIES)
            )
            if btc_enough:
                try:
                    btc_path = str(Path(output_dir) / "btc_har_report.html")
                    _emit_quantstats_html(
                        btc_returns,
                        btc_benchmark,
                        btc_path,
                        title="HAR Model \u2014 BTC/USDT 1h",
                        benchmark_title="Persistence Baseline",
                    )
                    results["btc"] = btc_path
                    logger.info("BTC QuantStats report generated: %s", btc_path)
                except Exception as exc:  # noqa: BLE001 - best effort
                    logger.warning("BTC QuantStats report failed: %s", exc)

            eth_enough = _non_empty_series(eth_returns) and (
                len(eth_returns) >= MIN_ROWS_FOR_SERIES
                or (eth_rows is not None and len(eth_rows) >= MIN_ROWS_FOR_SERIES)
            )
            if eth_enough:
                try:
                    eth_path = str(Path(output_dir) / "eth_har_report.html")
                    _emit_quantstats_html(
                        eth_returns,
                        eth_benchmark,
                        eth_path,
                        title="HAR Model \u2014 ETH/USDT 1h",
                        benchmark_title="Persistence Baseline",
                    )
                    results["eth"] = eth_path
                    logger.info("ETH QuantStats report generated: %s", eth_path)
                except Exception as exc:  # noqa: BLE001 - best effort
                    logger.warning("ETH QuantStats report failed: %s", exc)
        else:
            logger.info(
                "quantstats not available - generating combined summary only"
            )

        combined_path = _generate_combined_summary(
            btc_stats=btc_stats,
            eth_stats=eth_stats,
            btc_rows=btc_rows,
            eth_rows=eth_rows,
            btc_returns=btc_returns,
            eth_returns=eth_returns,
            calibration_day=calibration_day,
            total_days=CALIBRATION_TOTAL_DAYS,
            output_dir=output_dir,
        )
        if combined_path:
            results["combined"] = combined_path

        return results
    except Exception as exc:  # noqa: BLE001 - report must never raise
        logger.error("generate_html_report failed: %s", exc)
        return {}


def _format_stats_section(label: str, stats: Optional[Dict[str, Any]]) -> str:
    """Build one asset section for the Telegram summary."""
    if not stats:
        return f"{label}\n  No data available."

    beats = stats.get("har_beats")
    beats_text = "✅" if beats else ("❌" if beats is not None else "N/A")
    improvement = stats.get("improvement_pct")
    improvement_text = (
        "N/A" if improvement is None else f"{improvement:+.1f}%"
    )
    persistence_mae = stats.get("persistence_mae")
    persistence_text = (
        "N/A" if persistence_mae is None else f"{persistence_mae:.2f}"
    )
    breakout_rate = stats.get("breakout_rate")
    breakout_text = (
        "N/A" if breakout_rate is None else f"{breakout_rate:.1%}"
    )
    mean_bias = stats.get("mean_bias")
    bias_text = "N/A" if mean_bias is None else f"{mean_bias:+.1f}"
    n_days = stats.get("n_days", 0)

    return (
        f"{label}\n"
        f"  Predictions: {stats.get('n_predictions', 0)} ({n_days} days)\n"
        f"  HAR MAE: {_fmt(stats.get('har_mae'))}\n"
        f"  Persistence MAE: {persistence_text}\n"
        f"  Improvement: {improvement_text}\n"
        f"  HAR beats naive: {beats_text}\n"
        f"  Breakouts: {stats.get('breakout_count', 0)} ({breakout_text})\n"
        f"  Mean bias: {bias_text}"
    )


def build_quantstats_telegram_message(
    btc_stats: Optional[Dict[str, Any]],
    eth_stats: Optional[Dict[str, Any]],
    report_paths: Dict[str, str],
) -> str:
    """Build a Telegram message summarizing the QuantStats report."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    calibration_day = (
        btc_stats.get("calibration_day")
        if btc_stats
        else (eth_stats.get("calibration_day") if eth_stats else None)
    )
    calibration_text = (
        f"{calibration_day} of {CALIBRATION_TOTAL_DAYS}"
        if calibration_day is not None
        else "N/A"
    )

    btc_section = _format_stats_section("BTC/USDT 1h Summary:", btc_stats)
    eth_section = _format_stats_section("ETH/USDT 1h Summary:", eth_stats)

    if "btc" in report_paths:
        btc_file = "✅ btc_har_report.html"
    else:
        btc_file = "— btc_har_report.html"
    if "eth" in report_paths:
        eth_file = "✅ eth_har_report.html"
    else:
        eth_file = "— eth_har_report.html"

    return (
        "📊 HAR Research — QuantStats Report\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"Generated: {generated} UTC\n"
        f"Calibration Day: {calibration_text}\n\n"
        f"{btc_section}\n\n"
        f"{eth_section}\n\n"
        "Report files generated:\n"
        f"  {btc_file}\n"
        f"  {eth_file}\n\n"
        "To view: Download from GitHub Actions\n"
        "artifacts or run locally.\n\n"
        "⚠️ Research tool only.\n"
        "Not financial advice."
    )


def _send_telegram_message(config: Any, text: str) -> Any:
    """Wrapper around the repository Telegram sender (mockable in tests)."""
    from kronos_trading.alerts.telegram_sender import send_message

    return send_message(config, text, parse_mode=None)


def main() -> int:
    """Full QuantStats report pipeline.

    Steps:
        1. Load .env and validate credentials.
        2. Fetch completed predictions from Supabase.
        3. Convert prediction errors to returns-like series.
        4. Generate HTML reports (QuantStats + combined summary).
        5. Compute summary stats.
        6. Send Telegram summary.

    Returns 0 on success (including partial success when quantstats is not
    installed), 1 on missing DB/Telegram configuration or send failures.
    """
    load_dotenv()
    logger.info("Starting HAR QuantStats report")

    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        logger.error("SUPABASE_DB_URL is not set")
        return 1

    try:
        from kronos_trading.alerts.telegram_sender import TelegramConfig

        telegram_config = TelegramConfig.from_env()
    except EnvironmentError as exc:
        logger.error("Telegram config error: %s", exc)
        return 1

    logger.info("Fetching predictions from Supabase")
    rows_by_asset = fetch_both_assets(db_url, TIMEFRAME)

    btc_rows = rows_by_asset.get("BTC/USDT", [])
    eth_rows = rows_by_asset.get("ETH/USDT", [])

    logger.info("BTC rows: %s | ETH rows: %s", len(btc_rows), len(eth_rows))

    btc_returns = predictions_to_returns(btc_rows)
    eth_returns = predictions_to_returns(eth_rows)
    btc_benchmark = predictions_to_benchmark(btc_rows)
    eth_benchmark = predictions_to_benchmark(eth_rows)

    btc_stats = compute_summary_stats(btc_rows)
    eth_stats = compute_summary_stats(eth_rows)

    calibration_day = (
        btc_stats.get("calibration_day")
        if btc_stats
        else (eth_stats.get("calibration_day") if eth_stats else 1)
    )

    logger.info(
        "BTC daily points: %s | ETH daily points: %s",
        len(btc_returns),
        len(eth_returns),
    )
    logger.info("Generating HTML reports")
    report_paths = generate_html_report(
        btc_returns,
        eth_returns,
        btc_benchmark,
        eth_benchmark,
        output_dir=REPORTS_DIR,
        btc_rows=btc_rows,
        eth_rows=eth_rows,
        btc_stats=btc_stats,
        eth_stats=eth_stats,
        calibration_day=calibration_day,
    )
    if report_paths:
        for name, path in report_paths.items():
            logger.info("Report ready: %s -> %s", name, path)
    else:
        logger.warning("No reports were generated")

    logger.info("Building Telegram summary")
    message = build_quantstats_telegram_message(
        btc_stats, eth_stats, report_paths
    )

    logger.info("Sending Telegram summary")
    result = _send_telegram_message(telegram_config, message)
    if result.success:
        logger.info(
            "Telegram summary sent: message_id=%s", result.message_id
        )
        return 0

    logger.error("Telegram summary send failed: %s", result.error)
    return 1


if __name__ == "__main__":
    sys.exit(main())
