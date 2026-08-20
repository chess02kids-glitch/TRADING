"""Breakout detection and live calibration for the alert bot (Step 3).

Pure functions only: input in, dataclass out. No DB access, no network access,
no side effects. This module answers one question every hour: "was the
realized range abnormal relative to what HAR predicted?"

* ``check_breakout``            - classifies one actual-vs-predicted comparison
                                 (ratio + fixed severity bands). It only
                                 classifies and quantifies - it never makes
                                 trading decisions.
* ``compute_prediction_error``  - signed / absolute / percent prediction error.
* ``get_live_calibration``      - aggregates a completed prediction history
                                 (``get_prediction_history`` output) into
                                 calibration statistics, including a model
                                 degradation flag.
* ``format_breakout_message`` /
  ``format_calibration_message`` - Telegram message text. Sending is Step 4's
                                 job; these functions never touch Telegram.

Severity bands are fixed a priori (independent of the threshold argument):
ratio < 2.0 -> none, 2.0-3.0 -> moderate, 3.0-5.0 -> severe, >= 5.0 -> extreme.

Boundary note: the Step 2 logger flags a breakout with a strict ``>``
(``actual > 2.0 * predicted``) while ``check_breakout`` uses ``>=`` per the
alert spec. At exactly 2.0x they disagree by design - both are kept as
specified.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Fixed constants (mirror the Step 2 logger: MIN_CALIBRATION_OBS=24 there).
MIN_CALIBRATION_OBS = 24    # minimum observations for any calibration result
BREAKOUT_THRESHOLD = 2.0    # default breakout ratio
RECENT_7D_ROWS = 168        # 7 days of 1h bars
RECENT_30D_ROWS = 720       # 30 days of 1h bars
DEGRADATION_FACTOR = 1.5    # recent MAE > overall MAE * this -> degrading
_RECENT_MIN_OBS = 24        # minimum rows inside a recent window


@dataclass
class BreakoutResult:
    """One actual-vs-predicted range comparison."""
    is_breakout: bool
    actual_range: float
    predicted_range: float
    ratio: Optional[float]          # None when predicted <= 0
    threshold: float
    severity: str                   # "none" / "moderate" / "severe" / "extreme"


@dataclass
class PredictionError:
    """Signed / absolute / percent error of one prediction."""
    actual: float
    predicted: float
    error: float                    # actual - predicted
    abs_error: float                # |actual - predicted|
    pct_error: Optional[float]      # 100 * error / predicted; None if predicted <= 0
    direction: str                  # "over" / "under" / "exact"


@dataclass
class LiveCalibration:
    """Aggregated calibration statistics over a completed prediction history."""
    n_obs: int
    har_mae: float
    persistence_mae: float
    har_beats_persistence: bool
    mean_bias: float
    breakout_count: int
    breakout_rate: float
    worst_ratio: Optional[float]    # max actual/predicted where predicted > 0
    best_ratio: Optional[float]     # min actual/predicted where predicted > 0
    recent_mae_7d: Optional[float]  # MAE of newest 168 rows (None if < 24)
    recent_mae_30d: Optional[float]  # MAE of newest 720 rows (None if < 24)
    is_degrading: bool              # recent_mae_7d > har_mae * 1.5


def check_breakout(
    actual_range: float,
    predicted_range: float,
    threshold: float = BREAKOUT_THRESHOLD,
) -> BreakoutResult:
    """Classify one realized range against its HAR prediction.

    ``predicted_range <= 0`` (honest unclamped OLS output) can never be a
    breakout: ratio is ``None`` and the function returns immediately, so a
    division by zero is impossible.

    Severity bands are fixed (2.0 / 3.0 / 5.0) regardless of ``threshold``.
    """
    a = float(actual_range)
    p = float(predicted_range)
    if p <= 0.0:
        return BreakoutResult(False, a, p, None, threshold, "none")

    ratio = a / p
    is_breakout = ratio >= threshold
    if ratio >= 5.0:
        severity = "extreme"
    elif ratio >= 3.0:
        severity = "severe"
    elif ratio >= 2.0:
        severity = "moderate"
    else:
        severity = "none"
    return BreakoutResult(is_breakout, a, p, ratio, threshold, severity)


def compute_prediction_error(
    actual_range: float,
    predicted_range: float,
) -> PredictionError:
    """Quantify the error of one prediction (pure arithmetic, never raises)."""
    a = float(actual_range)
    p = float(predicted_range)
    error = a - p
    pct_error = (100.0 * error / p) if p > 0.0 else None
    if a > p:
        direction = "over"
    elif a < p:
        direction = "under"
    else:
        direction = "exact"
    return PredictionError(a, p, error, abs(error), pct_error, direction)


def _valid_series(
    history: List[Dict[str, Any]],
) -> tuple:
    """Split a history into (actuals, predicted) arrays in DESC order.

    Defensively skips rows with missing/non-finite values so one corrupt row
    cannot poison the aggregate statistics.
    """
    actuals: List[float] = []
    predicted: List[float] = []
    for row in history:
        a = row.get("actual_range")
        p = row.get("har_predicted_range")
        if (isinstance(a, (int, float)) and isinstance(p, (int, float))
                and math.isfinite(float(a)) and math.isfinite(float(p))):
            actuals.append(float(a))
            predicted.append(float(p))
    return actuals, predicted


def _mae(actuals: List[float], predicted: List[float]) -> Optional[float]:
    """Mean absolute error over equally-sized series; None when empty."""
    if not actuals or len(actuals) != len(predicted):
        return None
    return sum(abs(a - p) for a, p in zip(actuals, predicted)) / len(actuals)


def _recent_mae(actuals: List[float], predicted: List[float],
                window: int) -> Optional[float]:
    """MAE over the newest ``window`` rows (history is DESC).

    Returns None when fewer than ``_RECENT_MIN_OBS`` rows are in the window.
    """
    slice_actuals = actuals[:window]
    slice_predicted = predicted[:window]
    if len(slice_actuals) < _RECENT_MIN_OBS:
        return None
    return _mae(slice_actuals, slice_predicted)


def _is_degrading(recent_mae_7d: Optional[float], har_mae: float) -> bool:
    """True when the recent 7-day MAE exceeds the overall MAE by 1.5x."""
    if recent_mae_7d is None:
        return False
    return recent_mae_7d > har_mae * DEGRADATION_FACTOR


def get_live_calibration(
    history: List[Dict[str, Any]],
) -> Optional[LiveCalibration]:
    """Calibration statistics over a completed prediction history.

    ``history`` is the output of ``prediction_logger.get_prediction_history``
    (DESC order, newest first, actual_range filled). Only completed rows can
    be in that list, so no future data can leak in here.

    Returns None when fewer than ``MIN_CALIBRATION_OBS`` (24) rows exist.

    * ``persistence_mae``: for row ``i`` (DESC), the persistence prediction is
      the HAR prediction of row ``i + 1`` (the next older row); the oldest row
      has no older neighbor and is skipped - same lag-1 definition as the
      Step 2 logger's calibration summary.
    * ``worst_ratio`` / ``best_ratio``: max/min of ``actual / predicted`` over
      rows with ``predicted > 0``; None when no such row exists.
    * ``recent_mae_7d`` / ``recent_mae_30d``: MAE of the newest 168/720 rows
      (None when fewer than 24 rows are in the window).
    * ``is_degrading``: ``recent_mae_7d > har_mae * 1.5``; False when the
      7-day MAE is unavailable.
    """
    if not history or len(history) < MIN_CALIBRATION_OBS:
        return None
    actuals, predicted = _valid_series(history)
    if not actuals:
        logger.warning("get_live_calibration: no valid rows in history")
        return None
    n_obs = len(actuals)

    har_errors = [a - p for a, p in zip(actuals, predicted)]
    har_mae = sum(abs(e) for e in har_errors) / n_obs
    mean_bias = sum(har_errors) / n_obs

    # Lag-1 persistence: the previous (older) row's HAR prediction predicts
    # this row's actual. Row i+1 in DESC order is older.
    persistence_errors = [
        actuals[i] - predicted[i + 1] for i in range(n_obs - 1)
    ]
    persistence_mae = sum(abs(e) for e in persistence_errors) / len(persistence_errors)

    breakout_count = sum(
        1 for row in history if row.get("breakout_flag")
    )

    ratios = [
        actuals[i] / predicted[i]
        for i in range(n_obs) if predicted[i] > 0.0
    ]
    worst_ratio = max(ratios) if ratios else None
    best_ratio = min(ratios) if ratios else None

    recent_mae_7d = _recent_mae(actuals, predicted, RECENT_7D_ROWS)
    recent_mae_30d = _recent_mae(actuals, predicted, RECENT_30D_ROWS)

    return LiveCalibration(
        n_obs=n_obs,
        har_mae=har_mae,
        persistence_mae=persistence_mae,
        har_beats_persistence=har_mae < persistence_mae,
        mean_bias=mean_bias,
        breakout_count=breakout_count,
        breakout_rate=breakout_count / n_obs,
        worst_ratio=worst_ratio,
        best_ratio=best_ratio,
        recent_mae_7d=recent_mae_7d,
        recent_mae_30d=recent_mae_30d,
        is_degrading=_is_degrading(recent_mae_7d, har_mae),
    )


_HEADER_BAR = "━" * 20


def format_breakout_message(
    asset: str,
    timeframe: str,
    result: BreakoutResult,
    timestamp: str,
) -> str:
    """Telegram text for a breakout result; ``""`` when it is not a breakout.

    Callers decide whether to send (Step 4). Renders the moderate/severe/
    extreme templates exactly as specified - message text only, no Telegram.
    """
    if not result.is_breakout or result.ratio is None:
        return ""
    if result.severity == "extreme":
        header = f"🔴 EXTREME VOLATILITY — {asset} {timeframe}"
        severity_line = "Severity: EXTREME ⚠️⚠️⚠️"
    elif result.severity == "severe":
        header = f"🚨 VOLATILITY BREAKOUT — {asset} {timeframe}"
        severity_line = "Severity: SEVERE"
    else:  # moderate (or a custom-threshold breakout with severity "none")
        header = f"⚠️ VOLATILITY SPIKE — {asset} {timeframe}"
        severity_line = f"Severity: {result.severity.upper()}"
    return "\n".join([
        header,
        _HEADER_BAR,
        f"HAR predicted: ${result.predicted_range:.2f}",
        f"Actual range:  ${result.actual_range:.2f}",
        f"Ratio: {result.ratio:.2f}× expected",
        severity_line,
        f"Time: {timestamp}",
    ])


def format_calibration_message(
    asset: str,
    timeframe: str,
    cal: LiveCalibration,
) -> str:
    """Telegram text for a periodic calibration report (message text only)."""
    worst = f"{cal.worst_ratio:.2f}×" if cal.worst_ratio is not None else "N/A"
    best = f"{cal.best_ratio:.2f}×" if cal.best_ratio is not None else "N/A"
    mae_7d = f"{cal.recent_mae_7d:.4f}" if cal.recent_mae_7d is not None else "N/A"
    mae_30d = f"{cal.recent_mae_30d:.4f}" if cal.recent_mae_30d is not None else "N/A"
    return "\n".join([
        "📊 HAR CALIBRATION REPORT",
        _HEADER_BAR,
        f"Asset: {asset} {timeframe}",
        f"Observations: {cal.n_obs}",
        "",
        "Accuracy:",
        f"  HAR MAE:         {cal.har_mae:.4f}",
        f"  Persistence MAE: {cal.persistence_mae:.4f}",
        f"  HAR beats naive: {'✅ YES' if cal.har_beats_persistence else '❌ NO'}",
        f"  Mean bias:       {cal.mean_bias:+.4f}",
        "",
        "Volatility events:",
        f"  Breakouts: {cal.breakout_count} ({cal.breakout_rate:.1%})",
        f"  Worst ratio: {worst}",
        f"  Best ratio:  {best}",
        "",
        "Recent performance:",
        f"  7-day MAE:  {mae_7d}",
        f"  30-day MAE: {mae_30d}",
        f"  Degrading:  {'⚠️ YES' if cal.is_degrading else '✅ NO'}",
    ])
