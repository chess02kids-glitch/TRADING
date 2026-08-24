import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Tuple

def run_concentration_analysis(
    entries: pd.Series, 
    exits: pd.Series, 
    sizes: pd.Series,
    df: pd.DataFrame
) -> Tuple[bool, Dict[str, float], str]:
    """
    Analyzes the portfolio's reliance on outlier trades.
    Returns (Passed, Metrics, Flag)
    """
    portfolio = vbt.Portfolio.from_signals(
        df["close"],
        entries,
        exits,
        size=sizes,
        size_type="percent",
        freq="1h",
        fees=0.001,
        slippage=0.001,
        init_cash=10000.0
    , upon_opposite_entry="Ignore")
    
    trades = portfolio.trades.records_readable
    if len(trades) == 0:
        return False, {
            "best_trade_pct": None,
            "top_5_trades_contribution_pct": None,
            "top_10_trades_contribution_pct": None,
            "best_month_pct": None,
            "best_quarter_pct": None
        }, "HIGH_CONCENTRATION"
        
    returns = trades["Return"]
    total_return = returns.sum()
    
    if total_return <= 0:
        return False, {
            "best_trade_pct": None,
            "top_5_trades_contribution_pct": None,
            "top_10_trades_contribution_pct": None,
            "best_month_pct": None,
            "best_quarter_pct": None
        }, "NEGATIVE_RETURN"
        
    sorted_returns = returns.sort_values(ascending=False)
    
    best_trade_pct = sorted_returns.iloc[0] / total_return if len(sorted_returns) > 0 else 0
    top_5_pct = sorted_returns.head(5).sum() / total_return if len(sorted_returns) >= 5 else 1.0
    top_10_pct = sorted_returns.head(10).sum() / total_return if len(sorted_returns) >= 10 else 1.0
    
    # Analyze best month
    monthly_returns = portfolio.returns().resample("ME").sum()
    best_month_pct = 0.0
    if len(monthly_returns) > 0:
        best_month_ret = monthly_returns.max()
        portfolio_total_ret = portfolio.returns().sum()
        if portfolio_total_ret > 0:
            best_month_pct = best_month_ret / portfolio_total_ret
            
    best_quarter_pct = 0.0
    quarterly_returns = portfolio.returns().resample("QE").sum()
    if len(quarterly_returns) > 0:
        best_q_ret = quarterly_returns.max()
        if portfolio_total_ret > 0:
            best_quarter_pct = best_q_ret / portfolio_total_ret
            
    # Determine flag
    flag = "LOW_CONCENTRATION"
    if top_5_pct > 0.40 or best_quarter_pct > 0.50:
        flag = "MODERATE_CONCENTRATION"
    if top_5_pct > 0.60 or best_trade_pct > 0.30 or best_month_pct > 0.50:
        flag = "HIGH_CONCENTRATION"
        
    # We reject if the strategy is highly concentrated (i.e. a few lucky trades)
    passed = flag != "HIGH_CONCENTRATION"
    
    metrics = {
        "best_trade_pct": float(best_trade_pct),
        "top_5_trades_contribution_pct": float(top_5_pct),
        "top_10_trades_contribution_pct": float(top_10_pct),
        "best_month_pct": float(best_month_pct),
        "best_quarter_pct": float(best_quarter_pct)
    }
    
    return passed, metrics, flag
