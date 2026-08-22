"""
Data loading for NautilusTrader backtest.

Loads:
1. OHLCV data from existing feather files
   (freqtrade_har/user_data/data/kucoin/)
   OR from CCXT if not available

2. HAR predictions from Supabase
   (historical predictions since bot started)
   OR computes walk-forward HAR from OHLCV

The key innovation:
   We use REAL HAR predictions from Supabase
   where they exist (days 1-30 of calibration)
   We compute walk-forward HAR for historical
   data before the bot started.
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def load_ohlcv(
    asset: str = "BTC/USDT",
    timeframe: str = "1h",
    start: str = "2024-01-01",
    end: str = "2026-01-01",
) -> pd.DataFrame:
    """
    Load OHLCV data.

    Tries in order:
    1. freqtrade_har feather files
    2. vectorbt_har if available
    3. CCXT KuCoin as fallback

    Returns DataFrame with columns:
      timestamp (DatetimeIndex UTC)
      open, high, low, close, volume

    Filtered to [start, end] range.
    """
    from vectorbt_har.data_loader import load_ohlcv_from_feather, filter_timerange
    
    # We leverage the existing reliable feather loader
    df = load_ohlcv_from_feather(asset, timeframe)
    return filter_timerange(df, start, end)


def load_har_predictions_from_supabase(
    asset: str = "BTC/USDT",
    timeframe: str = "1h",
) -> pd.Series:
    """
    Load real HAR predictions from Supabase.
    These are the predictions the live bot
    has been making since 2026-08-20.

    Returns pd.Series indexed by timestamp (UTC)
    with har_predicted_range values.

    Returns empty Series if not available.
    Never raises.
    """
    try:
        import psycopg
        from psycopg.rows import dict_row

        url = os.environ.get("SUPABASE_DB_URL")
        if not url:
            logger.info(
                "No SUPABASE_DB_URL — "
                "skipping live predictions")
            return pd.Series(dtype=float)

        with psycopg.connect(
            url, row_factory=dict_row
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT timestamp,
                           har_predicted_range
                    FROM har_predictions
                    WHERE asset = %s
                      AND timeframe = %s
                      AND har_predicted_range
                          IS NOT NULL
                    ORDER BY timestamp ASC
                """, (asset, timeframe))
                rows = cur.fetchall()

        if not rows:
            return pd.Series(dtype=float)

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        return df["har_predicted_range"]

    except Exception as e:
        logger.warning(
            f"Supabase fetch failed: {e}")
        return pd.Series(dtype=float)


def compute_har_walkforward(
    ohlcv_df: pd.DataFrame,
) -> pd.Series:
    """
    Compute walk-forward HAR predictions
    for the full OHLCV dataset.

    Uses kronos_trading.volatility_baselines
    har_forecast() function.

    This is the same computation used
    in vectorbt_har but applied here
    for historical data.

    Returns pd.Series of predicted ranges
    indexed by timestamp.
    """
    from kronos_trading.volatility_baselines \
        import har_forecast, HAR_MIN_TRAIN

    ranges = ohlcv_df["high"] - ohlcv_df["low"]
    predictions = pd.Series(
        np.nan,
        index=ranges.index,
        dtype=float)

    n = len(ranges)
    for i in range(HAR_MIN_TRAIN, n):
        try:
            pred = har_forecast(ranges.iloc[:i])
            if np.isfinite(pred) and pred > 0:
                predictions.iloc[i] = pred
        except Exception:
            pass

    return predictions


def get_combined_har_predictions(
    ohlcv_df: pd.DataFrame,
    asset: str = "BTC/USDT",
    timeframe: str = "1h",
) -> pd.Series:
    """
    Get HAR predictions combining:
    1. Historical walk-forward (computed)
    2. Live Supabase predictions (real)

    Live predictions override historical
    where they overlap.

    This is the key feature of this approach:
    Real predictions used when available.
    """
    # Compute historical walk-forward
    historical = compute_har_walkforward(
        ohlcv_df)

    # Get live predictions from Supabase
    live = load_har_predictions_from_supabase(
        asset, timeframe)

    if live.empty:
        logger.info(
            "No live predictions available. "
            "Using walk-forward only.")
        return historical

    # Combine: live overrides historical
    combined = historical.copy()
    live_aligned = live.reindex(
        historical.index, method="nearest",
        tolerance=pd.Timedelta("1h"))
    combined.update(live_aligned.dropna())

    live_count = live_aligned.dropna().count()
    logger.info(
        f"Combined predictions: "
        f"{historical.notna().sum()} historical + "
        f"{live_count} live for {asset}")

    return combined
