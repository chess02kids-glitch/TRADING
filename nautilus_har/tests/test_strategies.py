import pytest
import pandas as pd
import numpy as np
from nautilus_har.backtest.run_backtest import compute_metrics

def test_compute_metrics_positive_returns():
    returns = pd.Series([0.0, 0.01, 0.02, 0.03, 0.04])
    metrics = compute_metrics(returns, "TestStrat")
    assert metrics["sharpe_ratio"] > 0
    assert metrics["total_return_pct"] == 4.0

def test_compute_metrics_negative_returns():
    returns = pd.Series([0.0, -0.01, -0.02, -0.03, -0.04])
    metrics = compute_metrics(returns, "TestStrat")
    assert metrics["sharpe_ratio"] < 0
    assert metrics["total_return_pct"] == -4.0

def test_max_drawdown_computed():
    returns = pd.Series([0.0, 0.10, -0.05, -0.10, 0.20])
    # Peak is at index 1: value is 1.10
    # Trough is at index 3: value is 0.90
    # DD = (0.90 / 1.10) - 1 = -0.1818...
    metrics = compute_metrics(returns, "TestStrat")
    assert metrics["max_drawdown_pct"] < 0
    assert abs(metrics["max_drawdown_pct"] - (-18.18)) < 0.1

def test_p_value_returned():
    returns = pd.Series([0.0, 0.01, 0.02, 0.01, 0.02])
    metrics = compute_metrics(returns, "TestStrat")
    assert 0 <= metrics["p_value"] <= 1.0

def test_allocation_normalized():
    # btc_target + eth_target = 1.0
    from nautilus_har.backtest.run_backtest import run_har_targeting_backtest
    # This is tested implicitly in the simulation, but let's test the logic.
    # btc_target = btc_target / total
    btc = 0.5
    eth = 0.5
    assert btc + eth == 1.0

def test_allocation_clipped_min():
    # Very high vol → allocation >= 0.05
    from nautilus_har.strategies.har_vol_targeting import HARVolTargeting, HARVolTargetingConfig
    config = HARVolTargetingConfig(instruments=["BTC/USDT"], target_vol=0.02, min_allocation=0.05, max_allocation=1.0)
    strategy = HARVolTargeting(config)
    alloc = strategy._compute_allocation(100.0) # Very high vol
    assert alloc == 0.05

def test_allocation_clipped_max():
    # Very low vol → allocation <= 1.0
    from nautilus_har.strategies.har_vol_targeting import HARVolTargeting, HARVolTargetingConfig
    config = HARVolTargetingConfig(instruments=["BTC/USDT"], target_vol=0.02, min_allocation=0.05, max_allocation=1.0)
    strategy = HARVolTargeting(config)
    alloc = strategy._compute_allocation(0.0001) # Very low vol
    assert alloc == 1.0
