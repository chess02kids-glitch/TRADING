"""Walk-forward HAR predictions on full datasets."""

import numpy as np
import pandas as pd
from kronos_trading.volatility_baselines import har_forecast, HAR_MIN_TRAIN

def compute_har_predictions(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Compute walk-forward HAR predictions
    for every bar in the DataFrame.

    Uses only past data for each prediction.
    Returns Series with same index as df.
    First HAR_MIN_TRAIN bars are NaN.
    """
    ranges = df["high"] - df["low"]
    n = len(ranges)
    predictions = pd.Series(
        np.nan,
        index=ranges.index,
        dtype=float)

    # Pre-calculate rolling means
    r1 = ranges.shift(1)
    r5 = ranges.rolling(5).mean().shift(1)
    r22 = ranges.rolling(22).mean().shift(1)
    
    X_full = np.column_stack((np.ones(n), r1.values, r5.values, r22.values))
    y_full = ranges.values
    
    # We can only start when X has no NaNs, which is at index 22
    for i in range(22 + HAR_MIN_TRAIN, n):
        X_train = X_full[22:i]
        y_train = y_full[22:i]
        
        try:
            beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
            pred = float(X_full[i] @ beta)
            if np.isfinite(pred) and pred > 0:
                predictions.iloc[i] = pred
        except np.linalg.LinAlgError:
            pass

    return predictions


def validate_predictions(
    predictions: pd.Series,
    max_nan_rate: float = 0.05,
) -> dict:
    """
    Validate HAR prediction quality.

    Returns dict:
      total_bars: int
      valid_predictions: int
      nan_count: int
      nan_rate: float
      passes_threshold: bool
      min_prediction: float
      max_prediction: float
      mean_prediction: float
      median_prediction: float
    """
    total_bars = len(predictions)
    nan_count = int(predictions.isna().sum())
    valid_predictions = total_bars - nan_count
    
    nan_rate = nan_count / total_bars if total_bars > 0 else 1.0
    passes_threshold = nan_rate < max_nan_rate
    
    valid_mask = predictions.notna()
    if valid_mask.any():
        valid_series = predictions[valid_mask]
        min_prediction = float(valid_series.min())
        max_prediction = float(valid_series.max())
        mean_prediction = float(valid_series.mean())
        median_prediction = float(valid_series.median())
    else:
        min_prediction = float('nan')
        max_prediction = float('nan')
        mean_prediction = float('nan')
        median_prediction = float('nan')
        
    return {
        "total_bars": total_bars,
        "valid_predictions": valid_predictions,
        "nan_count": nan_count,
        "nan_rate": nan_rate,
        "passes_threshold": passes_threshold,
        "min_prediction": min_prediction,
        "max_prediction": max_prediction,
        "mean_prediction": mean_prediction,
        "median_prediction": median_prediction,
    }


def compute_stop_distances(
    predictions: pd.Series,
    close: pd.Series,
    multiplier: float,
    min_stop: float = 0.005,
    max_stop: float = 0.20,
    fallback_stop: float = 0.05,
) -> pd.Series:
    """
    Compute dynamic stop distances.

    stop = multiplier * prediction / close
    Clipped to [min_stop, max_stop].
    NaN predictions -> fallback_stop.

    Returns Series of stop distances
    (as positive fractions, e.g. 0.05 = 5%).
    """
    stop_distance = (multiplier * predictions / close)
    stop_distance = stop_distance.clip(min_stop, max_stop)
    stop_distance = stop_distance.fillna(fallback_stop)
    return stop_distance
