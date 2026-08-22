"""
Supabase data access layer.
All database queries live here.
Read-only access to har_predictions table.
No writes. No inserts. No deletes.
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


def get_db_url() -> Optional[str]:
    """
    Get Supabase connection URL from
    environment variable.
    Returns None if not set.
    """
    return os.environ.get("SUPABASE_DB_URL")


def fetch_predictions(
    asset: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 720,
    completed_only: bool = False,
) -> pd.DataFrame:
    """
    Fetch recent predictions from Supabase.

    Parameters
    ----------
    asset : str
        Trading pair e.g. "BTC/USDT"
    timeframe : str
        Candle timeframe e.g. "1h"
    limit : int
        Maximum rows to return
    completed_only : bool
        If True, only rows where
        actual_range IS NOT NULL

    Returns
    -------
    pd.DataFrame with columns matching schema.
    Empty DataFrame on error.
    """
    try:
        import psycopg
        from psycopg.rows import dict_row

        url = get_db_url()
        if not url:
            logger.warning(
                "SUPABASE_DB_URL not set")
            return pd.DataFrame()

        where_clause = (
            "WHERE asset = %s "
            "AND timeframe = %s"
        )
        if completed_only:
            where_clause += (
                " AND actual_range IS NOT NULL")

        query = f"""
            SELECT *
            FROM har_predictions
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT %s
        """

        with psycopg.connect(
            url,
            row_factory=dict_row
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (asset, timeframe, limit))
                rows = cur.fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Parse timestamps
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], utc=True,
            errors="coerce")
        df["created_at"] = pd.to_datetime(
            df["created_at"], utc=True,
            errors="coerce")

        # Sort ascending for charts
        df = df.sort_values(
            "timestamp").reset_index(drop=True)

        return df

    except ImportError:
        logger.error("psycopg not installed")
        return pd.DataFrame()

    except Exception as e:
        logger.error(
            f"Failed to fetch predictions "
            f"for {asset}: {e}")
        return pd.DataFrame()


def fetch_all_assets(
    timeframe: str = "1h",
    limit: int = 720,
) -> dict[str, pd.DataFrame]:
    """
    Fetch predictions for all assets.

    Returns
    -------
    dict mapping asset name to DataFrame.
    e.g. {"BTC/USDT": df, "ETH/USDT": df}
    """
    from dashboard.config import ASSETS
    return {
        asset: fetch_predictions(
            asset=asset,
            timeframe=timeframe,
            limit=limit,
        )
        for asset in ASSETS
    }


def fetch_calibration_summary(
    asset: str = "BTC/USDT",
    timeframe: str = "1h",
) -> dict:
    """
    Compute calibration summary statistics.

    Returns dict with:
      total_predictions: int
      completed: int
      pending: int
      har_mae: float | None
      persistence_mae: float | None
      har_beats: bool | None
      breakout_count: int
      breakout_rate: float
      regime_counts: dict
      calibration_day: int
      days_remaining: int
      mean_bias: float | None
      first_prediction_ts: str | None
    """
    from dashboard.config import (
        CALIBRATION_TOTAL_DAYS,
        MIN_PREDICTIONS_FOR_STATS,
    )

    df = fetch_predictions(
        asset=asset,
        timeframe=timeframe,
        limit=99999,
    )

    if df.empty:
        return {
            "total_predictions": 0,
            "completed": 0,
            "pending": 0,
            "har_mae": None,
            "persistence_mae": None,
            "har_beats": None,
            "breakout_count": 0,
            "breakout_rate": 0.0,
            "regime_counts": {},
            "calibration_day": 1,
            "days_remaining": CALIBRATION_TOTAL_DAYS,
            "mean_bias": None,
            "first_prediction_ts": None,
        }

    completed = df[
        df["actual_range"].notna()]
    pending = df[
        df["actual_range"].isna()]

    # Calibration day
    if not df.empty and "created_at" in df:
        first_ts = df["created_at"].min()
        if pd.notna(first_ts):
            now = pd.Timestamp.now(tz="UTC")
            days = max(1, int(
                (now - first_ts).total_seconds()
                / 86400) + 1)
            days_remaining = max(
                0, CALIBRATION_TOTAL_DAYS - days)
            first_ts_str = str(first_ts)[:19]
        else:
            days = 1
            days_remaining = CALIBRATION_TOTAL_DAYS
            first_ts_str = None
    else:
        days = 1
        days_remaining = CALIBRATION_TOTAL_DAYS
        first_ts_str = None

    # MAE stats
    har_mae = None
    persistence_mae = None
    har_beats = None
    mean_bias = None

    if len(completed) >= MIN_PREDICTIONS_FOR_STATS:
        har_mae = float(
            completed["abs_prediction_error"].mean())
        mean_bias = float(
            completed["prediction_error"].mean())

        # Persistence: lag-1 prediction
        prev_pred = completed[
            "har_predicted_range"].shift(1)
        persist_errors = (
            completed["actual_range"] - prev_pred
        ).abs().dropna()
        if len(persist_errors) > 0:
            persistence_mae = float(
                persist_errors.mean())
            har_beats = har_mae < persistence_mae

    # Breakouts
    breakout_count = int(
        completed["breakout_flag"].sum()
    ) if not completed.empty else 0
    breakout_rate = (
        breakout_count / len(completed)
        if len(completed) > 0 else 0.0
    )

    # Regime distribution
    regime_counts = {}
    if not df.empty and "regime" in df.columns:
        counts = df["regime"].value_counts()
        regime_counts = counts.to_dict()

    return {
        "total_predictions": len(df),
        "completed": len(completed),
        "pending": len(pending),
        "har_mae": har_mae,
        "persistence_mae": persistence_mae,
        "har_beats": har_beats,
        "breakout_count": breakout_count,
        "breakout_rate": breakout_rate,
        "regime_counts": regime_counts,
        "calibration_day": days,
        "days_remaining": days_remaining,
        "mean_bias": mean_bias,
        "first_prediction_ts": first_ts_str,
    }


def fetch_breakouts(
    limit: int = 20,
) -> pd.DataFrame:
    """
    Fetch all breakout events across
    all assets.

    Returns DataFrame ordered by
    timestamp DESC.
    """
    try:
        import psycopg
        from psycopg.rows import dict_row

        url = get_db_url()
        if not url:
            return pd.DataFrame()

        query = """
            SELECT
                timestamp,
                asset,
                timeframe,
                har_predicted_range,
                actual_range,
                ROUND(
                    (actual_range /
                     NULLIF(har_predicted_range, 0)
                    )::numeric, 2
                ) AS ratio,
                created_at
            FROM har_predictions
            WHERE breakout_flag = 1
              AND actual_range IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT %s
        """

        with psycopg.connect(
            url,
            row_factory=dict_row
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (limit,))
                rows = cur.fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], utc=True,
            errors="coerce")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch breakouts: {e}")
        return pd.DataFrame()
