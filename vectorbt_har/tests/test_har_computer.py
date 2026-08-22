"""Tests for har_computer.py."""

import numpy as np
import pandas as pd
from vectorbt_har.har_computer import (
    compute_har_predictions, 
    validate_predictions, 
    compute_stop_distances
)
from kronos_trading.volatility_baselines import HAR_MIN_TRAIN

def test_compute_har_predictions_length():
    df = pd.DataFrame({"high": np.arange(100), "low": np.arange(100)-1})
    preds = compute_har_predictions(df)
    assert len(preds) == len(df)

def test_compute_har_first_bars_nan():
    df = pd.DataFrame({"high": np.arange(100), "low": np.arange(100)-1})
    preds = compute_har_predictions(df)
    assert preds.iloc[:HAR_MIN_TRAIN].isna().all()

def test_compute_har_no_lookahead():
    df1 = pd.DataFrame({"high": np.arange(100.0), "low": np.arange(100.0)-1.0})
    df2 = df1.copy()
    df2.loc[50:, "high"] = 999.0
    
    p1 = compute_har_predictions(df1)
    p2 = compute_har_predictions(df2)
    # prediction at 49 uses data up to 48.
    # changing data at 50 shouldn't affect predictions up to 50
    assert p1.iloc[:51].equals(p2.iloc[:51])
    # However, prediction at 51 will use data at 50, so it will differ.
    assert p1.iloc[52] != p2.iloc[52]

def test_compute_har_all_valid_after_warmup():
    # Make a long enough deterministic series
    ranges = np.sin(np.arange(1000) * 0.1) + 2.0
    df = pd.DataFrame({"high": ranges, "low": np.zeros(1000)})
    preds = compute_har_predictions(df)
    # Exclude first 30 bars (warmup + safe margin)
    valid_rate = preds.iloc[30:].notna().mean()
    assert valid_rate > 0.95

def test_validate_predictions_pass():
    preds = pd.Series([1.0] * 96 + [np.nan] * 4)
    report = validate_predictions(preds, max_nan_rate=0.05)
    assert report["passes_threshold"] == True
    assert report["nan_rate"] == 0.04

def test_validate_predictions_fail():
    preds = pd.Series([1.0] * 90 + [np.nan] * 10)
    report = validate_predictions(preds, max_nan_rate=0.05)
    assert report["passes_threshold"] == False
    assert report["nan_rate"] == 0.10

def test_compute_stop_distances_formula():
    preds = pd.Series([1000.0])
    close = pd.Series([50000.0])
    stop = compute_stop_distances(preds, close, 1.5, min_stop=0.001)
    assert stop.iloc[0] == 0.03

def test_compute_stop_distances_clipped_min():
    preds = pd.Series([1.0])
    close = pd.Series([50000.0])
    stop = compute_stop_distances(preds, close, 1.0, min_stop=0.005)
    assert stop.iloc[0] == 0.005

def test_compute_stop_distances_clipped_max():
    preds = pd.Series([20000.0])
    close = pd.Series([50000.0])
    stop = compute_stop_distances(preds, close, 1.0, max_stop=0.20)
    assert stop.iloc[0] == 0.20

def test_compute_stop_distances_nan_fallback():
    preds = pd.Series([np.nan])
    close = pd.Series([50000.0])
    stop = compute_stop_distances(preds, close, 1.0, fallback_stop=0.05)
    assert stop.iloc[0] == 0.05
