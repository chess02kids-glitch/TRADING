import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Tuple

def run_fast_screen(
    entries: pd.Series, 
    exits: pd.Series, 
    sizes: pd.Series,
    close: pd.Series, 
    fees: float = 0.001,
    slippage: float = 0.001
) -> dict:
    """
    Runs a fast VectorBT portfolio simulation.
    Takes entry/exit boolean arrays and close price.
    """
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        size=sizes,
        size_type="percent",
        fees=fees,
        slippage=slippage,
        freq="1h",
        init_cash=10000.0,
        reject_prob=0.0
    )
    
    stats = portfolio.stats()
    
    # Calculate additional metrics
    # Convert index to native python types for JSON serialization
    res = {
        "return_pct": float(stats.get("Total Return [%]", 0.0)),
        "sharpe": float(stats.get("Sharpe Ratio", 0.0)),
        "sortino": float(stats.get("Sortino Ratio", 0.0)),
        "max_drawdown_pct": float(stats.get("Max Drawdown [%]", 0.0)),
        "calmar": float(stats.get("Calmar Ratio", 0.0)),
        "profit_factor": float(stats.get("Profit Factor", 0.0)),
        "win_rate": float(stats.get("Win Rate [%]", 0.0)),
        "trade_count": int(stats.get("Total Closed Trades", 0)),
        "average_trade_pct": float(stats.get("Avg Winning Trade [%]", 0.0)) if "Avg Winning Trade [%]" in stats else 0.0,
    }
    
    # NaN replacement
    for k, v in res.items():
        if pd.isna(v) or np.isnan(v):
            res[k] = 0.0
            
    return res

def filter_base_metrics(metrics: dict, min_trades: int = 50, min_profit_factor: float = 1.05) -> Tuple[bool, str]:
    """
    Instantly rejects clearly bad strategies.
    Returns (Passed, RejectionReason)
    """
    if metrics["trade_count"] < min_trades:
        print(f"DEBUG SCREENER: trade_count={metrics['trade_count']} < {min_trades}")
        return False, "LOW_TRADE_COUNT"
        
    if metrics["profit_factor"] < min_profit_factor:
        return False, "LOW_PROFIT_FACTOR"
        
    if metrics["max_drawdown_pct"] < -50.0:
        return False, "CATASTROPHIC_DRAWDOWN"
        
    if metrics["return_pct"] < 0:
        return False, "NEGATIVE_RETURN"
        
    return True, ""
