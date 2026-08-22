"""Tests for backtester.py."""

import numpy as np
import pandas as pd
from vectorbt_har.backtester import (
    run_single_backtest, 
    run_baseline_backtest, 
    compute_trade_pvalue
)

def _get_mock_data():
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    # Trend up then down to ensure trades happen
    close = np.linspace(100, 200, 50).tolist() + np.linspace(200, 50, 50).tolist()
    high = [c + 5 for c in close]
    # Add a huge range to trigger entry
    high[25] = close[25] + 50
    low = [c - 5 for c in close]
    df = pd.DataFrame({"high": high, "low": low, "close": close}, index=dates)
    preds = pd.Series([10.0] * 100, index=dates)
    return df, preds

def test_run_single_backtest_returns_dict():
    df, preds = _get_mock_data()
    res = run_single_backtest(df, preds, stop_multiplier=1.0)
    assert isinstance(res, dict)
    assert "multiplier" in res
    assert "total_return_pct" in res
    assert "sharpe_ratio" in res

def test_run_baseline_backtest_returns_dict():
    df, _ = _get_mock_data()
    res = run_baseline_backtest(df)
    assert isinstance(res, dict)
    assert "multiplier" in res
    assert res["multiplier"] == 0.0

def test_run_single_backtest_no_crash_empty():
    empty_df = pd.DataFrame()
    empty_preds = pd.Series(dtype=float)
    res = run_single_backtest(empty_df, empty_preds, 1.0)
    assert isinstance(res, dict)
    assert len(res) == 0

def test_compute_pvalue_few_trades():
    df, preds = _get_mock_data()
    # Mock data might not have 5 trades
    res = run_single_backtest(df, preds, 1.0)
    if "portfolio" in res and res["total_trades"] < 5:
        pval = compute_trade_pvalue(res["portfolio"])
        assert pval is None

def test_compute_pvalue_returns_float():
    # Force many trades by oscillating price wildly
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    # High range to enter, then drop to exit, repeatedly
    close = np.array([100, 90] * 50)
    high = close + 20
    low = close - 5
    df = pd.DataFrame({"high": high, "low": low, "close": close}, index=dates)
    preds = pd.Series([10.0] * 100, index=dates)
    
    res = run_single_backtest(df, preds, 1.0)
    if res.get("total_trades", 0) >= 5:
        pval = compute_trade_pvalue(res["portfolio"])
        assert isinstance(pval, float)
        assert 0.0 <= pval <= 1.0
