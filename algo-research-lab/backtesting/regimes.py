import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, List

def classify_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifies market regimes for each bar WITHOUT lookahead bias.
    Uses rolling windows to define local volatility and trend context.
    """
    close = df["close"]
    
    # Calculate True Range proxy (high - low / close) for volatility
    tr = (df["high"] - df["low"]) / df["close"]
    # 7-day (168h) rolling average of True Range
    rolling_vol = tr.rolling(window=168, min_periods=24).mean()
    
    # Context window: 90 days (2160h)
    # We rank the current rolling_vol against the past 90 days
    # This prevents lookahead and uses only what the agent knows at that exact timestamp.
    # Note: Using rolling.quantile is slow but robust. For speed, we approximate with mean + std.
    vol_mean_90d = rolling_vol.rolling(window=2160, min_periods=168).mean()
    vol_std_90d = rolling_vol.rolling(window=2160, min_periods=168).std()
    
    z_vol = (rolling_vol - vol_mean_90d) / vol_std_90d
    
    vol_regime = pd.Series("NORMAL_VOL", index=df.index)
    vol_regime[z_vol < -1.0] = "LOW_VOL"
    vol_regime[z_vol > 1.0] = "HIGH_VOL"
    vol_regime[z_vol > 2.5] = "EXTREME_VOL"
    
    # Trend classification (using ADX proxy or simple SMA slope)
    sma20 = close.rolling(20 * 24).mean()
    sma50 = close.rolling(50 * 24).mean()
    # If SMAs are fanned out and moving, it's trending. Otherwise ranging.
    diff_pct = abs(sma20 - sma50) / close
    diff_mean = diff_pct.rolling(window=2160, min_periods=168).mean()
    
    trend_regime = pd.Series("RANGING", index=df.index)
    trend_regime[diff_pct > diff_mean] = "TRENDING"
    
    return pd.DataFrame({
        "volatility_regime": vol_regime,
        "trend_regime": trend_regime
    })


def run_regime_analysis(
    entries: pd.Series, 
    exits: pd.Series, 
    sizes: pd.Series,
    df: pd.DataFrame
) -> List[Dict]:
    """
    Evaluates strategy performance sliced by market regime.
    """
    regimes_df = classify_regimes(df)
    
    # We need to simulate the strategy first to get the trade history or bar-by-bar returns
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
    )
    
    returns = portfolio.returns()
    
    results = []
    
    # Evaluate Volatility Regimes
    for r in ["LOW_VOL", "NORMAL_VOL", "HIGH_VOL", "EXTREME_VOL"]:
        mask = regimes_df["volatility_regime"] == r
        if mask.sum() > 0:
            regime_rets = returns[mask]
            
            # Simple metrics for the slice
            ret_pct = float(regime_rets.sum() * 100) # approximate
            sharpe = 0.0
            if regime_rets.std() > 0:
                # Annualized sharpe approximation (24*365 = 8760)
                sharpe = float((regime_rets.mean() / regime_rets.std()) * np.sqrt(8760))
                
            results.append({
                "regime_type": r,
                "return_pct": ret_pct,
                "sharpe": sharpe,
                "sortino": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_trade_pct": 0.0,
                "trade_count": 0
            })
            
    # Evaluate Trend Regimes
    for r in ["TRENDING", "RANGING"]:
        mask = regimes_df["trend_regime"] == r
        if mask.sum() > 0:
            regime_rets = returns[mask]
            
            ret_pct = float(regime_rets.sum() * 100)
            sharpe = 0.0
            if regime_rets.std() > 0:
                sharpe = float((regime_rets.mean() / regime_rets.std()) * np.sqrt(8760))
                
            results.append({
                "regime_type": r,
                "return_pct": ret_pct,
                "sharpe": sharpe,
                "sortino": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_trade_pct": 0.0,
                "trade_count": 0
            })
            
    return results
