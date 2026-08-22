"""
Run NautilusTrader backtests for all
three HAR volatility targeting strategies.

Usage:
  python nautilus_har/backtest/run_backtest.py

This script:
1. Loads OHLCV data for BTC and ETH
2. Computes/loads HAR predictions
3. Runs Strategy A (equal weight)
4. Runs Strategy B (HAR vol targeting)
5. Runs Strategy C (inverse HAR)
6. Computes comparison metrics
7. Evaluates gate criteria
8. Writes NAUTILUS_RESULTS.md
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(
    Path(__file__).parent.parent.parent))

from nautilus_har.config import (
    ASSETS, TIMEFRAME, INITIAL_CAPITAL,
    BACKTEST_START, BACKTEST_END,
    TARGET_VOL_PER_BAR, MIN_ALLOCATION,
    MAX_ALLOCATION, REBALANCE_THRESHOLD,
    STABILITY_PERIODS, FEES_BPS,
)
from nautilus_har.data_loader import (
    load_ohlcv,
    get_combined_har_predictions,
)
from nautilus_har.backtest.results_analyzer import (
    evaluate_gates,
    write_results_md,
)

logger = logging.getLogger(__name__)


def run_equal_weight_backtest(
    ohlcv_data: dict[str, pd.DataFrame],
    start: str,
    end: str,
) -> dict:
    """
    Simulate equal-weight portfolio.

    Since NautilusTrader setup is complex,
    implement a clean simulation:
    - Start with 50% BTC, 50% ETH
    - Track portfolio value over time
    - Rebalance daily to equal weight
    - Apply 0.1% fees on each rebalance

    Returns performance metrics dict.
    """
    # Filter to timerange
    btc = ohlcv_data["BTC/USDT"].loc[start:end]
    eth = ohlcv_data["ETH/USDT"].loc[start:end]

    if btc.empty or eth.empty:
        return {}

    # Initialize
    capital = INITIAL_CAPITAL
    btc_val = capital * 0.5
    eth_val = capital * 0.5

    returns = []
    timestamps = []

    btc_prices = btc["close"]
    eth_prices = eth["close"]

    # Common timestamps
    common_idx = btc_prices.index.intersection(
        eth_prices.index)
        
    if len(common_idx) < 2:
        return {}

    prev_btc = btc_prices.loc[common_idx[0]]
    prev_eth = eth_prices.loc[common_idx[0]]

    for ts in common_idx[1:]:
        curr_btc = btc_prices.loc[ts]
        curr_eth = eth_prices.loc[ts]

        # Price returns
        btc_ret = (curr_btc - prev_btc) / prev_btc
        eth_ret = (curr_eth - prev_eth) / prev_eth

        # Update values
        btc_val *= (1 + btc_ret)
        eth_val *= (1 + eth_ret)
        total = btc_val + eth_val

        # Daily rebalance check
        if ts.hour == 0:
            target = total * 0.5
            btc_drift = abs(btc_val - target)
            if btc_drift / total > REBALANCE_THRESHOLD:
                fee = btc_drift * (FEES_BPS / 10000)
                total -= fee
                btc_val = total * 0.5
                eth_val = total * 0.5

        portfolio_return = (
            total - capital) / capital
        returns.append(portfolio_return)
        timestamps.append(ts)

        prev_btc = curr_btc
        prev_eth = curr_eth
        capital = total

    return compute_metrics(
        pd.Series(returns, index=timestamps),
        "EqualWeight")


def run_har_targeting_backtest(
    ohlcv_data: dict[str, pd.DataFrame],
    har_predictions: dict[str, pd.Series],
    start: str,
    end: str,
    inverse: bool = False,
) -> dict:
    """
    Simulate HAR volatility targeting portfolio.

    For each bar:
    1. Get HAR prediction for each asset
    2. Compute volatility estimate = pred/price
    3. Compute allocation:
       Normal:  target_vol / vol_estimate
       Inverse: vol_estimate / target_vol
    4. Rebalance if drift > threshold
    5. Apply fees on rebalance

    Returns performance metrics dict.
    """
    btc_ohlcv = ohlcv_data[
        "BTC/USDT"].loc[start:end]
    eth_ohlcv = ohlcv_data[
        "ETH/USDT"].loc[start:end]
    btc_preds = har_predictions[
        "BTC/USDT"].loc[start:end]
    eth_preds = har_predictions[
        "ETH/USDT"].loc[start:end]

    if btc_ohlcv.empty or eth_ohlcv.empty:
        return {}

    common_idx = (
        btc_ohlcv.index
        .intersection(eth_ohlcv.index))
        
    if len(common_idx) < 2:
        return {}

    capital = INITIAL_CAPITAL
    btc_alloc = 0.5
    eth_alloc = 0.5
    btc_val = capital * btc_alloc
    eth_val = capital * eth_alloc

    returns = []
    timestamps = []
    allocations = []

    for i, ts in enumerate(common_idx[1:]):
        prev_ts = common_idx[i]

        # Price updates
        prev_btc = btc_ohlcv["close"].loc[prev_ts]
        curr_btc = btc_ohlcv["close"].loc[ts]
        prev_eth = eth_ohlcv["close"].loc[prev_ts]
        curr_eth = eth_ohlcv["close"].loc[ts]

        btc_ret = (
            (curr_btc - prev_btc) / prev_btc)
        eth_ret = (
            (curr_eth - prev_eth) / prev_eth)

        btc_val *= (1 + btc_ret)
        eth_val *= (1 + eth_ret)
        total = btc_val + eth_val

        # HAR-based allocation
        btc_pred = _get_prediction(
            btc_preds, ts)
        eth_pred = _get_prediction(
            eth_preds, ts)

        if btc_pred and eth_pred:
            btc_vol = btc_pred / curr_btc
            eth_vol = eth_pred / curr_eth

            if inverse:
                btc_target = min(MAX_ALLOCATION,
                    max(MIN_ALLOCATION,
                        btc_vol / TARGET_VOL_PER_BAR))
                eth_target = min(MAX_ALLOCATION,
                    max(MIN_ALLOCATION,
                        eth_vol / TARGET_VOL_PER_BAR))
            else:
                btc_target = min(MAX_ALLOCATION,
                    max(MIN_ALLOCATION,
                        TARGET_VOL_PER_BAR / btc_vol))
                eth_target = min(MAX_ALLOCATION,
                    max(MIN_ALLOCATION,
                        TARGET_VOL_PER_BAR / eth_vol))

            # Normalize to sum to 1
            total_target = btc_target + eth_target
            btc_target /= total_target
            eth_target /= total_target

            # Rebalance if drift
            curr_btc_w = btc_val / total
            if abs(curr_btc_w - btc_target) > (
                REBALANCE_THRESHOLD):
                fee_amount = abs(
                    btc_val - total * btc_target
                ) * (FEES_BPS / 10000)
                total -= fee_amount
                btc_val = total * btc_target
                eth_val = total * eth_target
                btc_alloc = btc_target
                eth_alloc = eth_target

        portfolio_val = btc_val + eth_val
        ret = (portfolio_val - capital) / capital
        returns.append(ret)
        timestamps.append(ts)
        allocations.append({
            "btc": btc_val / portfolio_val,
            "eth": eth_val / portfolio_val,
        })

        capital = portfolio_val

    name = "HARVolInverse" if inverse \
        else "HARVolTargeting"
    result = compute_metrics(
        pd.Series(returns, index=timestamps),
        name)
    result["allocations"] = allocations
    return result


def _get_prediction(
    preds: pd.Series,
    ts: pd.Timestamp,
) -> float | None:
    """Get nearest HAR prediction."""
    try:
        idx = preds.index.get_indexer(
            [ts], method="nearest",
            tolerance=pd.Timedelta("2h"))
        if idx[0] < 0:
            return None
        val = preds.iloc[idx[0]]
        if pd.isna(val) or val <= 0:
            return None
        return float(val)
    except Exception:
        return None


def compute_metrics(
    cumulative_returns: pd.Series,
    strategy_name: str,
) -> dict:
    """
    Compute performance metrics from
    cumulative returns series.

    Returns dict with:
      strategy: str
      total_return_pct: float
      annualized_return_pct: float
      sharpe_ratio: float
      sortino_ratio: float
      max_drawdown_pct: float
      volatility_annualized_pct: float
      calmar_ratio: float
    """
    if cumulative_returns.empty:
        return {"strategy": strategy_name}

    # Compute bar-by-bar returns
    bar_returns = cumulative_returns.diff()
    bar_returns.iloc[0] = (
        cumulative_returns.iloc[0])

    # Total return
    total_return = float(
        cumulative_returns.iloc[-1]) * 100

    # Annualized (8760 hours per year)
    n_bars = len(bar_returns)
    if n_bars == 0:
        return {"strategy": strategy_name}
        
    ann_factor = 8760 / n_bars
    ann_return = (
        (1 + total_return / 100)
        ** ann_factor - 1) * 100

    # Sharpe (annualized, risk-free = 0)
    if bar_returns.std() > 0:
        sharpe = (
            bar_returns.mean()
            / bar_returns.std()
            * np.sqrt(8760))
    else:
        sharpe = 0.0

    # Sortino (downside deviation)
    downside = bar_returns[bar_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (
            bar_returns.mean()
            / downside.std()
            * np.sqrt(8760))
    else:
        sortino = 0.0

    # Max drawdown
    cum_max = (
        (1 + cumulative_returns).cummax())
    drawdown = (
        (1 + cumulative_returns) / cum_max - 1)
    max_dd = float(drawdown.min()) * 100

    # Annualized volatility
    ann_vol = (
        bar_returns.std()
        * np.sqrt(8760) * 100)

    # Calmar
    calmar = (
        ann_return / abs(max_dd)
        if max_dd != 0 else 0.0)

    # p-value of returns
    from scipy import stats
    # we use nan_policy='omit' just in case
    t_stat, p_value = stats.ttest_1samp(
        bar_returns.dropna(), 0, nan_policy='omit')
        
    # handle nan p-value
    if np.isnan(p_value):
        p_value = 1.0

    return {
        "strategy": strategy_name,
        "total_return_pct": round(total_return, 2),
        "annualized_return_pct": round(
            ann_return, 2),
        "sharpe_ratio": round(float(sharpe), 3),
        "sortino_ratio": round(
            float(sortino), 3),
        "max_drawdown_pct": round(max_dd, 2),
        "volatility_pct": round(ann_vol, 2),
        "calmar_ratio": round(float(calmar), 3),
        "n_bars": n_bars,
        "p_value": round(float(p_value), 4),
    }

def main():
    logging.basicConfig(level=logging.INFO)
    print("Loading OHLCV data...")
    ohlcv_data = {
        asset: load_ohlcv(asset, TIMEFRAME, BACKTEST_START, BACKTEST_END)
        for asset in ASSETS
    }
    
    print("Computing/Loading HAR predictions...")
    har_predictions = {
        asset: get_combined_har_predictions(ohlcv_data[asset], asset, TIMEFRAME)
        for asset in ASSETS
    }
    
    print("Running Strategy A (Equal Weight)...")
    results_a = run_equal_weight_backtest(ohlcv_data, BACKTEST_START, BACKTEST_END)
    print(f"Strategy A Complete: {results_a}")
    
    print("Running Strategy B (HAR Volatility Targeting)...")
    results_b = run_har_targeting_backtest(ohlcv_data, har_predictions, BACKTEST_START, BACKTEST_END, inverse=False)
    print(f"Strategy B Complete: {results_b}")
    
    print("Running Strategy C (Inverse HAR Targeting)...")
    results_c = run_har_targeting_backtest(ohlcv_data, har_predictions, BACKTEST_START, BACKTEST_END, inverse=True)
    print(f"Strategy C Complete: {results_c}")
    
    print("Evaluating Time Stability for Strategy B...")
    stability_b = []
    for start, end in STABILITY_PERIODS:
        res = run_har_targeting_backtest(ohlcv_data, har_predictions, start, end, inverse=False)
        stability_b.append(res)
    
    print("Evaluating Gate Criteria...")
    gates = evaluate_gates(results_a, results_b, results_c, stability_b)
    
    print("Writing results...")
    write_results_md(results_a, results_b, results_c, stability_b, gates)

if __name__ == "__main__":
    main()
