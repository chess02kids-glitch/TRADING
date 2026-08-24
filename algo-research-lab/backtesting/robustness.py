import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Tuple, List, Callable
import copy

def run_cost_stress_test(
    entries: pd.Series, 
    exits: pd.Series, 
    sizes: pd.Series,
    close: pd.Series,
    baseline_fees: float = 0.001,
    baseline_slippage: float = 0.001
) -> Tuple[bool, float, List[Dict]]:
    """
    Evaluates strategy under progressively worse execution assumptions.
    Returns (Passed, cost_survival_limit, List of curve data points)
    """
    multipliers = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5]
    
    curve_data = []
    cost_survival_limit = 0.0
    passed = False
    
    for mult in multipliers:
        f = baseline_fees * mult
        s = baseline_slippage * mult
        
        portfolio = vbt.Portfolio.from_signals(
            close,
            entries,
            exits,
            size=sizes,
            size_type="percent",
            freq="1h",
            fees=f,
            slippage=s,
            init_cash=10000.0
        )
        
        stats = portfolio.stats()
        ret = float(stats.get("Total Return [%]", 0.0))
        sharpe = float(stats.get("Sharpe Ratio", 0.0))
        
        if np.isnan(ret): ret = 0.0
        if np.isnan(sharpe): sharpe = 0.0
            
        curve_data.append({
            "multiplier": mult,
            "return_pct": ret,
            "sharpe": sharpe,
            "sortino": float(stats.get("Sortino Ratio", 0.0)) if not pd.isna(stats.get("Sortino Ratio")) else 0.0,
            "max_drawdown_pct": float(stats.get("Max Drawdown [%]", 0.0)) if not pd.isna(stats.get("Max Drawdown [%]")) else 0.0,
            "profit_factor": float(stats.get("Profit Factor", 0.0)) if not pd.isna(stats.get("Profit Factor")) else 0.0,
            "trade_count": int(stats.get("Total Closed Trades", 0)) if not pd.isna(stats.get("Total Closed Trades")) else 0
        })
        
        if ret > 0:
            cost_survival_limit = mult
            
    # We require survival up to at least 1.5x
    passed = cost_survival_limit >= 1.5
            
    return passed, cost_survival_limit, curve_data

def run_parameter_stability_test(
    genome: dict,
    close: pd.Series,
    generator_func: Callable
) -> Tuple[bool, str, float, List[Dict]]:
    """
    Perturbs parameters to ensure the chosen config is on a stable plateau.
    Returns (Passed, Flag, StabilityScore, List of run data points)
    """
    params = genome.get("direction", {}).get("params", {})
    if not params:
        # Strategy has no parameters (e.g. basic price action), perfectly stable
        return True, "STABLE", 10.0, []
        
    runs_data = []
    sharpes = []
    
    for key, base_val in params.items():
        if isinstance(base_val, (int, float)):
            # Perturb by -20%, -10%, +10%, +20%
            deltas = [-0.2, -0.1, 0.1, 0.2]
            for delta in deltas:
                test_val = base_val * (1.0 + delta)
                if isinstance(base_val, int):
                    test_val = int(round(test_val))
                if test_val == base_val:
                    continue # skipped redundant
                    
                test_genome = copy.deepcopy(genome)
                test_genome["direction"]["params"][key] = test_val
                
                # Mock a df for the generator (only close is strictly required for this test generally)
                df_mock = pd.DataFrame({"close": close, "high": close, "low": close})
                entries, exits, sizes = generator_func(df_mock, test_genome)
                
                if entries.sum() == 0:
                    continue
                    
                portfolio = vbt.Portfolio.from_signals(
                    close, entries, exits, size=sizes, size_type="percent", freq="1h", init_cash=10000.0, fees=0.001, slippage=0.001
                )
                
                stats = portfolio.stats()
                sharpe = float(stats.get("Sharpe Ratio", 0.0))
                if np.isnan(sharpe): sharpe = 0.0
                
                runs_data.append({
                    "parameter_name": key,
                    "parameter_value": {"value": test_val},
                    "oos_return_pct": float(stats.get("Total Return [%]", 0.0)),
                    "oos_sharpe": sharpe,
                    "max_drawdown_pct": float(stats.get("Max Drawdown [%]", 0.0)),
                    "profit_factor": float(stats.get("Profit Factor", 0.0)),
                    "trade_count": int(stats.get("Total Closed Trades", 0))
                })
                sharpes.append(sharpe)
                
    if not sharpes:
        return True, "STABLE", 10.0, runs_data
        
    sharpes_arr = np.array(sharpes)
    sharpe_std = np.std(sharpes_arr)
    sharpe_mean = np.mean(sharpes_arr)
    
    stability_score = 10.0 - (sharpe_std * 5.0) # penalty for high variance
    stability_score = max(0.0, min(10.0, stability_score))
    
    flag = "STABLE"
    if stability_score < 7.0:
        flag = "MODERATELY_STABLE"
    if stability_score < 4.0 or sharpe_mean < 0:
        flag = "FRAGILE"
        
    passed = flag != "FRAGILE"
    
    return passed, flag, float(stability_score), runs_data
