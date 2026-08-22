"""Vectorbt integration for HAR stop-loss testing."""

import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy.stats import ttest_1samp
from vectorbt_har.signals import compute_entry_signals, compute_exit_signals
from vectorbt_har.har_computer import compute_stop_distances


def run_single_backtest(
    ohlcv_df: pd.DataFrame,
    har_predictions: pd.Series,
    stop_multiplier: float,
    fees: float = 0.001,
    initial_cash: float = 10000.0,
) -> dict:
    """Run one backtest with vectorbt."""
    try:
        if len(ohlcv_df) == 0:
            return {}
            
        close = ohlcv_df["close"]
        high = ohlcv_df["high"]
        low = ohlcv_df["low"]
        
        entries = compute_entry_signals(high, low)
        exits = compute_exit_signals(close)
        
        # Stop-loss
        stop_distance = compute_stop_distances(
            har_predictions, close, stop_multiplier
        )
        
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            sl_stop=stop_distance,
            fees=fees,
            init_cash=initial_cash,
            freq="1h",
        )
        
        trade_count = int(pf.trades.count())
        
        return {
            "multiplier": float(stop_multiplier),
            "total_return_pct": float(pf.total_return() * 100),
            "sharpe_ratio": float(pf.sharpe_ratio()) if np.isfinite(pf.sharpe_ratio()) else 0.0,
            "max_drawdown_pct": float(pf.max_drawdown() * 100) if pf.max_drawdown() is not None else 0.0,
            "total_trades": trade_count,
            "win_rate": float(pf.trades.win_rate() * 100) if trade_count > 0 else 0.0,
            "profit_factor": float(pf.trades.profit_factor()) if trade_count > 0 and np.isfinite(pf.trades.profit_factor()) else 0.0,
            "avg_trade_pct": float(pf.trades.returns.mean() * 100) if trade_count > 0 else 0.0,
            "portfolio": pf,  # keep the portfolio for p-value calc
        }
    except Exception as e:
        print(f"Error in single backtest (mult {stop_multiplier}): {e}")
        return {}


def run_baseline_backtest(
    ohlcv_df: pd.DataFrame,
    fixed_stop: float = 0.05,
    fees: float = 0.001,
    initial_cash: float = 10000.0,
) -> dict:
    """Run baseline backtest with fixed stop."""
    try:
        if len(ohlcv_df) == 0:
            return {}
            
        close = ohlcv_df["close"]
        high = ohlcv_df["high"]
        low = ohlcv_df["low"]
        
        entries = compute_entry_signals(high, low)
        exits = compute_exit_signals(close)
        
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            sl_stop=fixed_stop,
            fees=fees,
            init_cash=initial_cash,
            freq="1h",
        )
        
        trade_count = int(pf.trades.count())
        
        return {
            "multiplier": 0.0,
            "total_return_pct": float(pf.total_return() * 100),
            "sharpe_ratio": float(pf.sharpe_ratio()) if np.isfinite(pf.sharpe_ratio()) else 0.0,
            "max_drawdown_pct": float(pf.max_drawdown() * 100) if pf.max_drawdown() is not None else 0.0,
            "total_trades": trade_count,
            "win_rate": float(pf.trades.win_rate() * 100) if trade_count > 0 else 0.0,
            "profit_factor": float(pf.trades.profit_factor()) if trade_count > 0 and np.isfinite(pf.trades.profit_factor()) else 0.0,
            "avg_trade_pct": float(pf.trades.returns.mean() * 100) if trade_count > 0 else 0.0,
            "portfolio": pf,
        }
    except Exception as e:
        print(f"Error in baseline backtest: {e}")
        return {}


def compute_trade_pvalue(
    pf,
) -> float | None:
    """
    Compute p-value for trade returns.
    H0: mean return = 0
    """
    if pf is None or pf.trades.count() < 5:
        return None
        
    returns = pf.trades.returns
    if len(returns) < 5:
        return None
        
    t_stat, p_val = ttest_1samp(returns.values, popmean=0.0)
    
    # If all returns are exactly the same (e.g. 0 variance), ttest can return nan
    if np.isnan(p_val):
        return None
        
    return float(p_val)
