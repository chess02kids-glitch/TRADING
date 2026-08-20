"""PostgreSQL (Supabase) persistence for HAR predictions.

Every HAR prediction the bot makes is logged *before* the predicted bar
closes (``actual_range`` NULL), then completed *after* it closes via
``update_actual`` - this ordering is the architectural guarantee that
calibration statistics never see future data.

Leakage rules enforced by design:

* ``log_prediction`` writes a row with ``actual_range = NULL`` - it is always
  called before the candle closes.
* ``update_actual`` only fills rows that still have ``actual_range IS NULL``;
  a completed row is never overwritten.
* ``get_prediction_history`` / ``get_calibration_summary`` use only rows with
  ``actual_range IS NOT NULL``.
* The persistence MAE in the calibration summary shifts the HAR prediction
  by one row chronologically (the previous hour's prediction predicts the
  current hour's actual) - never a future value.
"""
from __future__ import annotations

import os
import logging
import math
import re
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from kronos_trading.alerts.har_forecaster import HarForecast

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = ""              # Backwards compatibility
DEFAULT_HISTORY_LIMIT = 720       # 30 days of 1h bars
MIN_CALIBRATION_OBS = 24          # minimum completed rows for a calibration summary
BREAKOUT_THRESHOLD = 2.0          # actual > 2.0 x predicted -> breakout flag
VALID_REGIMES = ("low", "medium", "high")

# ISO8601 UTC, e.g. "2024-01-15T14:00:00Z" (optional fractional seconds).
_ISO8601_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")

_COLUMNS = (
    "id", '"timestamp"', "asset", "timeframe", "har_predicted_range",
    "coef_b0", "coef_b1", "coef_b2", "coef_b3", "n_obs", "regime",
    "actual_range", "prediction_error", "abs_prediction_error",
    "breakout_flag", "created_at",
)
_SELECT_ALL = f"SELECT {', '.join(_COLUMNS)} FROM har_predictions"


def _connect(db_path: str = None) -> psycopg.Connection:
    """Open one connection with the project's row factory.
    
    The db_path argument is kept for backwards compatibility with test suites
    and existing function signatures, but SUPABASE_DB_URL env var is used.
    """
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise ValueError("SUPABASE_DB_URL environment variable is not set")
    return psycopg.connect(url, row_factory=dict_row)


def _normalize(asset: str, timeframe: str) -> tuple:
    """Upper-case the symbol and lower-case the timeframe for a stable key."""
    return asset.strip().upper(), timeframe.strip().lower()


def _now_iso() -> str:
    """Current UTC time as an ISO8601 string, e.g. '2024-01-15T14:00:00Z'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def initialize_db(db_path: str = None) -> None:
    """No-op for Supabase. Table should be created via SQL migrations."""
    pass


def log_prediction(
    db_path: str,
    timestamp: str,
    asset: str,
    timeframe: str,
    forecast: HarForecast,
    regime: Optional[str] = None,
) -> int:
    """Log one HAR prediction for a bar that has not closed yet."""
    asset, timeframe = _normalize(asset, timeframe)
    if not isinstance(timestamp, str) or not _ISO8601_UTC_RE.match(timestamp):
        logger.warning("Timestamp %r is not ISO8601 UTC (expected e.g. "
                       "'2024-01-15T14:00:00Z'); storing as-is", timestamp)
    if regime is None:
        regime = getattr(forecast, "regime", None)
    if regime is not None:
        regime = str(regime).strip().lower()
        if regime not in VALID_REGIMES:
            logger.warning("Unknown regime %r (expected low/medium/high); "
                           "storing as-is", regime)

    coefs = tuple(forecast.coefficients)
    if len(coefs) != 4:
        logger.warning("Expected 4 HAR coefficients, got %d; padding with zeros",
                       len(coefs))
        coefs = (coefs + (0.0,) * 4)[:4]
    b0, b1, b2, b3 = coefs

    with closing(_connect(db_path)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""INSERT INTO har_predictions
                    ("timestamp", asset, timeframe, har_predicted_range,
                     coef_b0, coef_b1, coef_b2, coef_b3, n_obs, regime, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT ("timestamp", asset, timeframe) DO NOTHING
                    RETURNING id""",
                (timestamp, asset, timeframe,
                 float(forecast.predicted_range), b0, b1, b2, b3,
                 int(forecast.n_obs), regime, _now_iso()),
            )
            row = cur.fetchone()
            if not row:
                logger.info("Duplicate prediction for %s %s @ %s - returning "
                            "existing row id", asset, timeframe, timestamp)
                cur.execute(
                    f'SELECT id FROM har_predictions WHERE "timestamp" = %s '
                    f"AND asset = %s AND timeframe = %s",
                    (timestamp, asset, timeframe),
                )
                row = cur.fetchone()
            conn.commit()
            
    if row is None:
        logger.critical("Insert succeeded but row lookup failed for %s %s @ %s",
                        asset, timeframe, timestamp)
        return -1
    return int(row["id"])


def update_actual(
    db_path: str,
    timestamp: str,
    asset: str,
    timeframe: str,
    actual_range: float,
) -> bool:
    """Fill in the realized range of a predicted bar (after it closes)."""
    asset, timeframe = _normalize(asset, timeframe)
    if not isinstance(actual_range, (int, float)) or not math.isfinite(actual_range):
        logger.warning("Non-finite actual_range %r for %s %s @ %s - skipping",
                       actual_range, asset, timeframe, timestamp)
        return False
    if actual_range < 0:
        logger.warning("Negative actual_range %r for %s %s @ %s (corrupt "
                       "candle?) - storing as-is", actual_range, asset,
                       timeframe, timestamp)

    with closing(_connect(db_path)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f'SELECT har_predicted_range FROM har_predictions '
                f'WHERE "timestamp" = %s AND asset = %s AND timeframe = %s',
                (timestamp, asset, timeframe),
            )
            row = cur.fetchone()
            if row is None:
                logger.warning("update_actual: no prediction row for %s %s @ %s",
                               asset, timeframe, timestamp)
                return False
                
            predicted = float(row["har_predicted_range"])
            error = float(actual_range) - predicted
            breakout = 1 if (predicted > 0.0
                             and float(actual_range) > BREAKOUT_THRESHOLD * predicted) else 0
                             
            cur.execute(
                f"""UPDATE har_predictions
                    SET actual_range = %s, prediction_error = %s,
                        abs_prediction_error = %s, breakout_flag = %s
                    WHERE "timestamp" = %s AND asset = %s AND timeframe = %s
                      AND actual_range IS NULL""",
                (float(actual_range), error, abs(error), breakout,
                 timestamp, asset, timeframe),
            )
            if cur.rowcount == 0:
                logger.warning("update_actual: row for %s %s @ %s already has an "
                               "actual; keeping the first value", asset, timeframe, timestamp)
            conn.commit()
    return True


def get_prediction_history(
    db_path: str,
    asset: str,
    timeframe: str,
    n: int = DEFAULT_HISTORY_LIMIT,
) -> List[Dict[str, Any]]:
    """Last ``n`` *completed* predictions (actual_range filled), newest first."""
    asset, timeframe = _normalize(asset, timeframe)
    if n < 1:
        logger.warning("get_prediction_history: n=%r < 1, returning []", n)
        return []
    with closing(_connect(db_path)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""{_SELECT_ALL}
                    WHERE asset = %s AND timeframe = %s AND actual_range IS NOT NULL
                    ORDER BY "timestamp" DESC
                    LIMIT %s""",
                (asset, timeframe, int(n)),
            )
            rows = cur.fetchall()
    return rows


def get_pending_predictions(
    db_path: str,
    asset: str,
    timeframe: str,
) -> List[Dict[str, Any]]:
    """Predictions still awaiting their realized range (actual_range NULL)."""
    asset, timeframe = _normalize(asset, timeframe)
    with closing(_connect(db_path)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""{_SELECT_ALL}
                    WHERE asset = %s AND timeframe = %s AND actual_range IS NULL
                    ORDER BY "timestamp" ASC""",
                (asset, timeframe),
            )
            rows = cur.fetchall()
    return rows


def get_calibration_summary(
    db_path: str,
    asset: str,
    timeframe: str,
    n: int = DEFAULT_HISTORY_LIMIT,
) -> Optional[Dict[str, Any]]:
    """Calibration statistics over the last ``n`` completed predictions."""
    rows = get_prediction_history(db_path, asset, timeframe, n)
    if len(rows) < MIN_CALIBRATION_OBS:
        return None
    chrono = list(reversed(rows))

    n_obs = len(chrono)
    har_errors = [float(r["actual_range"]) - float(r["har_predicted_range"])
                  for r in chrono]
    har_mae = sum(abs(e) for e in har_errors) / n_obs
    mean_prediction_error = sum(har_errors) / n_obs

    persistence_errors = [
        float(chrono[i]["actual_range"]) - float(chrono[i - 1]["har_predicted_range"])
        for i in range(1, n_obs)
    ]
    persistence_mae = sum(abs(e) for e in persistence_errors) / len(persistence_errors)

    breakout_count = sum(1 for r in chrono if r["breakout_flag"])
    regime_counts = {"low": 0, "medium": 0, "high": 0}
    for r in chrono:
        if r["regime"] in regime_counts:
            regime_counts[r["regime"]] += 1

    return {
        "n_obs": n_obs,
        "har_mae": har_mae,
        "persistence_mae": persistence_mae,
        "har_beats_persistence": har_mae < persistence_mae,
        "mean_prediction_error": mean_prediction_error,
        "breakout_count": breakout_count,
        "breakout_rate": breakout_count / n_obs,
        "regime_counts": regime_counts,
    }
