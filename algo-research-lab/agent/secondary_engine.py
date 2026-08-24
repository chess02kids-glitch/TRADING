import pandas as pd
from typing import Tuple

def run_secondary_engine(
    close: pd.Series, 
    entries: pd.Series, 
    exits: pd.Series, 
    fees: float = 0.0, 
    slippage: float = 0.0,
    init_cash: float = 10000.0
) -> Tuple[float, int]:
    """
    Minimal event-driven backtester to cross-verify VectorBT.
    Assumes standard from_signals execution:
    - Execute at the close price of the SAME bar the signal is generated.
    - No accumulation (binary positions).
    - 100% of cash deployed on entry.
    """
    cash = init_cash
    position = 0.0
    trades = 0
    
    # Pre-align arrays for speed
    c = close.values
    ent = entries.values
    ext = exits.values
    
    for i in range(len(c)):
        if position == 0 and ent[i]:
            # Entry
            exec_price = c[i] * (1 + slippage)
            # Deploy all cash
            position = cash / exec_price
            cash = 0
            
            # deduct fees
            trade_val = position * exec_price
            fee_amount = trade_val * fees
            
            # To pay fees, we actually have to reduce position or deduct from cash.
            # In VectorBT default, fees are deducted from cash, which might go negative, 
            # or deducted from the position.
            # We'll just approximate it by reducing the position equivalent to the fee.
            position -= (fee_amount / exec_price)
            
        elif position > 0 and ext[i]:
            # Exit
            exec_price = c[i] * (1 - slippage)
            trade_val = position * exec_price
            
            fee_amount = trade_val * fees
            cash = trade_val - fee_amount
            position = 0
            trades += 1
            
    # Mark to market at the end
    if position > 0:
        final_value = cash + (position * c[-1] * (1 - slippage) * (1 - fees))
    else:
        final_value = cash
        
    return final_value, trades

if __name__ == "__main__":
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    prices = pd.Series([100.0, 102.0] * 50, index=dates)
    entries = prices == 100
    exits = prices == 102
    
    val, t = run_secondary_engine(prices, entries, exits)
    print(f"Secondary Engine -> Trades: {t}, Final Value: {val:.2f}")
