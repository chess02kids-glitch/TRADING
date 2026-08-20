"""SQLite persistence for HAR predictions (Step 2).

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

Conventions:

* sqlite3 stdlib only - no ORM.
* Every function opens its own connection via a context manager
  (``check_same_thread=False`` + WAL so the scheduler can run from any
  thread/process without locking issues) and closes it on exit.
* ``initialize_db`` is idempotent and is invoked defensively by every
  function, so a missing table can never crash a run.
* Unexpected conditions are logged, not raised; the only exceptions that
  propagate are real DB-level errors (corrupt file), which the scheduler
  treats as fatal for that cycle.
"""
from __future__ import annotations

import logging
import math
import os
import re
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg

from kronos_trading.alerts.har_forecaster import HarForecast

logger = logging.getLogger(__name__)

# Default DB fallback is not strictly needed for cloud, but we keep the variable for compatibility
DEFAULT_DB_PATH = str(Path(__file__).resolve().parents[2] / "data" / "db" / "har_predictions.db")

DEFAULT_HISTORY_LIMIT = 720       # 30 days of 1h bars
MIN_CALIBRATION_OBS = 24          # minimum completed rows for a calibration summary
BREAKOUT_THRESHOLD = 2.0          # actual > 2.0 x predicted -> breakout flag
VALID_REGIMES = ("low", "medium", "high")

# ISO8601 UTC, e.g. "2024-01-15T14:00:00Z" (optional fractional seconds).
_ISO8601_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS har_predictions (
    id SERIAL PRIMARY KEY,
    "timestamp" TEXT NOT NULL,
    asset TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    har_predicted_range REAL NOT NULL,
    coef_b0 REAL NOT NULL,
    coef_b1 REAL NOT NULL,
    coef_b2 REAL NOT NULL,
    coef_b3 REAL NOT NULL,
    n_obs INTEGER NOT NULL,
    regime TEXT,
    actual_range REAL,
    prediction_error REAL,
    abs_prediction_error REAL,
    breakout_flag INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE ("timestamp", asset, timeframe)
)
"""

_COLUMNS = (
    "id", '"timestamp"', "asset", "timeframe", "har_predicted_range",
    "coef_b0", "coef_b1", "coef_b2", "coef_b3", "n_obs", "regime",
    "actual_range", "prediction_error", "abs_prediction_error",
    "breakout_flag", "created_at",
)
_SELECT_ALL = f"SELECT {', '.join(_COLUMNS)} FROM har_predictions"


def _connect(db_path: str):
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError("SUPABASE_DB_URL missing")
    return psycopg.connect(url, row_factory=dict_row, autocommit=True)


def _normalize(asset: str, timeframe: str) -> tuple:
    """Upper-case the symbol and lower-case the timeframe for a stable key."""
    return asset.strip().upper(), timeframe.strip().lower()


def _now_iso() -> str:
    """Current UTC time as an ISO8601 string, e.g. '2024-01-15T14:00:00Z'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def initialize_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create the SQLite database and ``har_predictions`` table if missing.

    Idempotent: safe to call any number of times. Also creates the parent
    directory. The unique constraint on (timestamp, asset, timeframe) makes
    duplicate predictions impossible.
    """
    path = Path(str(db_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(path)) as conn:
        with conn:
            conn.execute(_SCHEMA)


def log_prediction(
    db_path: str,
    timestamp: str,
    asset: str,
    timeframe: str,
    forecast: HarForecast,
    regime: Optional[str] = None,
) -> int:
    """Log one HAR prediction for a bar that has not closed yet.

    Args:
        db_path: SQLite database path.
        timestamp: ISO8601 UTC open time of the predicted bar, e.g.
            ``"2024-01-15T14:00:00Z"``. Part of the unique key.
        asset: e.g. ``'BTC/USDT'`` (case-insensitive).
        timeframe: e.g. ``'1h'`` (case-insensitive).
        forecast: ``HarForecast`` from ``har_forecaster.predict_next_range``;
            its coefficients (B0..B3) are persisted for calibration analysis.
        regime: optional ``'low'``/``'medium'``/``'high'`` from
            ``classify_regime``. Falls back to ``forecast.regime`` when the
            kwarg is omitted (forward-compatible with a regime-carrying
            forecast dataclass).

    Returns:
        Row id of the inserted prediction. On a duplicate
        (timestamp, asset, timeframe) the insert is skipped silently and the
        existing row id is returned; the stored values are never overwritten.
    """
    initialize_db(db_path)
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
        with conn:
            cur = conn.execute(
                f"""INSERT INTO har_predictions
                    ("timestamp", asset, timeframe, har_predicted_range,
                     coef_b0, coef_b1, coef_b2, coef_b3, n_obs, regime, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT("timestamp", asset, timeframe) DO NOTHING""",
                (timestamp, asset, timeframe,
                 float(forecast.predicted_range), b0, b1, b2, b3,
                 int(forecast.n_obs), regime, _now_iso()),
            )
            if cur.rowcount == 0:
                logger.info("Duplicate prediction for %s %s @ %s - returning "
                            "existing row id", asset, timeframe, timestamp)
        row = conn.execute(
            f'SELECT id FROM har_predictions WHERE "timestamp" = %s '
            f"AND asset = %s AND timeframe = %s",
            (timestamp, asset, timeframe),
        ).fetchone()
    if row is None:  # pragma: no cover - impossible unless the DB is corrupt
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
    """Fill in the realized range of a predicted bar (after it closes).

    Computes and stores:
        prediction_error      = actual_range - har_predicted_range
        abs_prediction_error  = |prediction_error|
        breakout_flag         = 1 if actual_range > 2.0 x predicted_range
                                (and predicted_range > 0) else 0

    A row is only updated while ``actual_range IS NULL``: a completed row is
    never overwritten by a second call (protects calibration integrity).

    Returns:
        True if the row was found (and updated, or already completed);
        False if no prediction exists for this (timestamp, asset, timeframe).
    """
    initialize_db(db_path)
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
        with conn:
            row = conn.execute(
                f'SELECT har_predicted_range FROM har_predictions '
                f'WHERE "timestamp" = %s AND asset = %s AND timeframe = %s',
                (timestamp, asset, timeframe),
            ).fetchone()
            if row is None:
                logger.warning("update_actual: no prediction row for %s %s @ %s",
                               asset, timeframe, timestamp)
                return False
            predicted = float(row["har_predicted_range"])
            error = float(actual_range) - predicted
            breakout = 1 if (predicted > 0.0
                             and float(actual_range) > BREAKOUT_THRESHOLD * predicted) else 0
            cur = conn.execute(
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
    return True


def get_prediction_history(
    db_path: str,
    asset: str,
    timeframe: str,
    n: int = DEFAULT_HISTORY_LIMIT,
) -> List[Dict[str, Any]]:
    """Last ``n`` *completed* predictions (actual_range filled), newest first.

    Each dict contains every schema column. Returns ``[]`` when there are no
    completed rows. Timestamps are ISO8601 UTC strings, so DESC ordering is
    chronological-newest-first.
    """
    initialize_db(db_path)
    asset, timeframe = _normalize(asset, timeframe)
    if n < 1:
        logger.warning("get_prediction_history: n=%r < 1, returning []", n)
        return []
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            f"""{_SELECT_ALL}
                WHERE asset = ? AND timeframe = ? AND actual_range IS NOT NULL
                ORDER BY "timestamp" DESC
                LIMIT ?""",
            (asset, timeframe, int(n)),
        ).fetchall()
    return [dict(r) for r in rows]


def get_pending_predictions(
    db_path: str,
    asset: str,
    timeframe: str,
) -> List[Dict[str, Any]]:
    """Predictions still awaiting their realized range (actual_range NULL).

    Ordered oldest first, so the caller completes them in candle order.
    """
    initialize_db(db_path)
    asset, timeframe = _normalize(asset, timeframe)
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            f"""{_SELECT_ALL}
                WHERE asset = ? AND timeframe = ? AND actual_range IS NULL
                ORDER BY "timestamp" ASC""",
            (asset, timeframe),
        ).fetchall()
    return [dict(r) for r in rows]


def get_calibration_summary(
    db_path: str,
    asset: str,
    timeframe: str,
    n: int = DEFAULT_HISTORY_LIMIT,
) -> Optional[Dict[str, Any]]:
    """Calibration statistics over the last ``n`` completed predictions.

    Uses ``get_prediction_history`` internally, so only rows with a filled
    actual range are ever considered.

    Returns None when fewer than ``MIN_CALIBRATION_OBS`` (24) observations.

    Returned dict:
        n_obs                    number of completed rows used
        har_mae                  mean |actual - HAR prediction|
        persistence_mae          mean |actual - previous row's HAR prediction|
                                 (lag-1 persistence; the first chronological
                                 row is skipped - it has no previous row)
        har_beats_persistence    har_mae < persistence_mae
        mean_prediction_error    signed mean error (bias)
        breakout_count           rows with breakout_flag = 1
        breakout_rate            breakout_count / n_obs
        regime_counts            {"low": n, "medium": n, "high": n}
                                 (rows with NULL regime are excluded)
    """
    rows = get_prediction_history(db_path, asset, timeframe, n)  # newest first
    if len(rows) < MIN_CALIBRATION_OBS:
        return None
    chrono = list(reversed(rows))  # oldest -> newest (ISO strings sort correctly)

    n_obs = len(chrono)
    har_errors = [float(r["actual_range"]) - float(r["har_predicted_range"])
                  for r in chrono]
    har_mae = sum(abs(e) for e in har_errors) / n_obs
    mean_prediction_error = sum(har_errors) / n_obs

    # Persistence: the previous hour's HAR prediction predicts this hour's
    # actual. The first chronological row has no previous row - skipped.
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
