"""
HAR Regime Filter for Freqtrade Strategies

Reads the latest HAR volatility prediction
from Supabase and returns regime classification.

HAR model: validated OOS DM p ≈ 2.15e-26
Predicts: next-bar candle range magnitude
Does NOT predict: price direction

This module is a FILTER only.
It tells strategies WHEN to trade.
It never determines WHAT direction to trade.

Paper trading only. No live orders.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Regime constants
REGIME_LOW = "low"
REGIME_MEDIUM = "medium"
REGIME_HIGH = "high"
REGIME_UNKNOWN = "unknown"

# Max age before prediction considered stale
MAX_PREDICTION_AGE_HOURS = 2


@dataclass
class HARPrediction:
    """
    Single HAR prediction fetched from Supabase.
    """
    timestamp: str
    asset: str
    timeframe: str
    har_predicted_range: float
    regime: str
    age_hours: float
    is_stale: bool


def get_latest_har_prediction(
    asset: str = "BTC/USDT",
    timeframe: str = "1h",
    db_url: Optional[str] = None,
) -> Optional[HARPrediction]:
    """
    Fetch latest HAR prediction from Supabase.

    Returns None on any failure.
    Returns None if prediction is stale.
    Never raises.

    Parameters
    ----------
    asset : str
        Trading pair e.g. "BTC/USDT"
    timeframe : str
        Candle timeframe e.g. "1h"
    db_url : str, optional
        Supabase connection URL.
        If None, reads SUPABASE_DB_URL env var.

    Returns
    -------
    HARPrediction or None
    """
    try:
        import psycopg
        from psycopg.rows import dict_row

        url = db_url or os.environ.get(
            "SUPABASE_DB_URL")

        if not url:
            logger.warning(
                "SUPABASE_DB_URL not set. "
                "HAR regime filter disabled.")
            return None

        with psycopg.connect(
            url,
            row_factory=dict_row
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        timestamp,
                        asset,
                        timeframe,
                        har_predicted_range,
                        regime,
                        created_at
                    FROM har_predictions
                    WHERE asset = %s
                      AND timeframe = %s
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (asset, timeframe)
                )
                row = cur.fetchone()

        if row is None:
            logger.info(
                f"No predictions found: "
                f"{asset} {timeframe}")
            return None

        # Compute prediction age
        created_raw = row["created_at"]
        if isinstance(created_raw, str):
            created_at = datetime.fromisoformat(
                created_raw.replace("Z", "+00:00"))
        else:
            created_at = created_raw

        if created_at.tzinfo is None:
            created_at = created_at.replace(
                tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age_hours = (
            (now - created_at).total_seconds()
            / 3600
        )
        is_stale = (
            age_hours > MAX_PREDICTION_AGE_HOURS)

        if is_stale:
            logger.warning(
                f"HAR prediction stale: "
                f"{age_hours:.1f}h old for "
                f"{asset}. Filter disabled.")

        regime = (
            row.get("regime") or REGIME_UNKNOWN)

        return HARPrediction(
            timestamp=str(row["timestamp"]),
            asset=asset,
            timeframe=timeframe,
            har_predicted_range=float(
                row["har_predicted_range"]),
            regime=regime,
            age_hours=round(age_hours, 2),
            is_stale=is_stale,
        )

    except ImportError:
        logger.warning(
            "psycopg not installed. "
            "HAR regime filter disabled.")
        return None

    except Exception as e:
        logger.warning(
            f"HAR fetch failed for "
            f"{asset}: {e}. "
            f"Strategy continues without filter.")
        return None


def get_regime(
    asset: str = "BTC/USDT",
    timeframe: str = "1h",
    db_url: Optional[str] = None,
) -> str:
    """
    Get current HAR regime for an asset.

    Returns
    -------
    str: "low" / "medium" / "high" / "unknown"

    Never raises.
    Returns "unknown" on any failure.
    Strategies treat "unknown" as "medium".
    """
    pred = get_latest_har_prediction(
        asset, timeframe, db_url)

    if pred is None:
        return REGIME_UNKNOWN
    if pred.is_stale:
        return REGIME_UNKNOWN
    return pred.regime


def is_tradeable_regime(
    asset: str = "BTC/USDT",
    timeframe: str = "1h",
    db_url: Optional[str] = None,
    allow_regimes: Optional[list] = None,
) -> bool:
    """
    Check if current regime allows trading.

    Default: allow low and medium only.
    Skips high volatility by default.

    Parameters
    ----------
    allow_regimes : list, optional
        Regimes that permit trading.
        Default: ["low", "medium"]

    Returns
    -------
    bool: True if trading is allowed
    """
    if allow_regimes is None:
        allow_regimes = [REGIME_LOW, REGIME_MEDIUM]

    regime = get_regime(asset, timeframe, db_url)

    # Unknown → conservative fallback
    if regime == REGIME_UNKNOWN:
        return REGIME_MEDIUM in allow_regimes

    return regime in allow_regimes


def is_high_volatility(
    asset: str = "BTC/USDT",
    timeframe: str = "1h",
    db_url: Optional[str] = None,
) -> bool:
    """Returns True if HAR regime is high."""
    return get_regime(
        asset, timeframe, db_url) == REGIME_HIGH


def is_low_volatility(
    asset: str = "BTC/USDT",
    timeframe: str = "1h",
    db_url: Optional[str] = None,
) -> bool:
    """Returns True if HAR regime is low."""
    return get_regime(
        asset, timeframe, db_url) == REGIME_LOW
