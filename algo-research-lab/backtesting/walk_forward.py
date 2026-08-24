import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Tuple

def run_walk_forward(
    entries: pd.Series, 
    exits: pd.Series, 
    sizes: pd.Series,
    close: pd.Series, 
    window_days: int = 180, 
    step_days: int = 60
) -> Tuple[bool, Dict[str, float]]:
    """
    Performs Walk-Forward validation by rolling through the data.
    Evaluates strategy using a composite WFO quality score.
    
    Returns (Passed, MetricsDict)
    """
    start_idx = 0
    total_bars = len(close)
    bars_per_day = 24 # Assuming 1h timeframe
    
    window_bars = window_days * bars_per_day
    step_bars = step_days * bars_per_day
    
    oos_returns = []
    oos_sharpes = []
    oos_drawdowns = []
    
    while start_idx + window_bars < total_bars:
        train_start = start_idx
        train_end = start_idx + window_bars
        test_start = train_end
        test_end = min(test_start + step_bars, total_bars)
        
        test_close = close.iloc[test_start:test_end]
        test_entries = entries.iloc[test_start:test_end]
        test_exits = exits.iloc[test_start:test_end]
        test_sizes = sizes.iloc[test_start:test_end]
        
        if test_entries.sum() > 0:
            portfolio = vbt.Portfolio.from_signals(
                test_close,
                test_entries,
                test_exits,
                size=test_sizes,
                size_type="percent",
                freq="1h",
                init_cash=10000.0
            )
            stats = portfolio.stats()
            ret = float(stats.get("Total Return [%]", 0.0))
            sharpe = float(stats.get("Sharpe Ratio", 0.0))
            dd = float(stats.get("Max Drawdown [%]", 0.0))
            
            # handle NaNs
            if np.isnan(sharpe): sharpe = 0.0
            if np.isnan(dd): dd = 0.0
                
            oos_returns.append(ret)
            oos_sharpes.append(sharpe)
            oos_drawdowns.append(dd)
        else:
            oos_returns.append(0.0)
            oos_sharpes.append(0.0)
            oos_drawdowns.append(0.0)
            
        start_idx += step_bars
        
    if not oos_returns:
        return False, {"oos_sharpe": 0.0, "win_rate_windows": 0.0, "wfo_quality_score": 0.0}
        
    returns_arr = np.array(oos_returns)
    sharpes_arr = np.array(oos_sharpes)
    dd_arr = np.array(oos_drawdowns)
    
    win_windows = np.sum(returns_arr > 0) / len(returns_arr)
    mean_ret = np.mean(returns_arr)
    median_ret = np.median(returns_arr)
    worst_window = np.min(returns_arr)
    worst_dd = np.min(dd_arr)
    
    oos_sharpe = 0.0
    if np.std(returns_arr) > 0:
        oos_sharpe = mean_ret / np.std(returns_arr)
        
    # WFO Quality Composite Score (0 to 10 scale approx)
    # Consistency (up to 3 pts)
    score_consistency = min(3.0, (win_windows - 0.3) * 6) if win_windows > 0.3 else 0
    # Positive Expectancy (up to 3 pts)
    score_expectancy = min(3.0, max(0, mean_ret / 5.0))
    # Risk-Adjusted (up to 2 pts)
    score_risk_adj = min(2.0, max(0, np.mean(sharpes_arr)))
    # Drawdown Control (up to 2 pts)
    score_dd = min(2.0, max(0, (20.0 + worst_dd) / 10.0)) # worst_dd is negative
    
    # Catastrophe penalty
    catastrophe_penalty = 5.0 if worst_window < -25.0 else 0.0
    
    wfo_quality = (score_consistency + score_expectancy + score_risk_adj + score_dd) - catastrophe_penalty
    wfo_quality = max(0.0, float(wfo_quality))
    
    metrics = {
        "oos_sharpe": float(oos_sharpe),
        "mean_oos_return": float(mean_ret),
        "median_oos_return": float(median_ret),
        "worst_window_return": float(worst_window),
        "worst_oos_drawdown": float(worst_dd),
        "win_rate_windows": float(win_windows),
        "wfo_quality_score": float(wfo_quality)
    }
    
    passed = wfo_quality >= 3.0 and win_windows >= 0.40
    
    return passed, metrics
