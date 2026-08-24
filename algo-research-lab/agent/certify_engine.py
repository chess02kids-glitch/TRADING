import pandas as pd
import numpy as np
import vectorbt as vbt

def generate_synthetic_data():
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    # Base synthetic trend
    trend_prices = np.linspace(100, 199, 100)
    # Synthetic MR (oscillating)
    mr_prices = np.array([100.0, 102.0] * 50)
    return dates, trend_prices, mr_prices

def test_1_buy_and_hold():
    dates, prices, _ = generate_synthetic_data()
    close = pd.Series(prices, index=dates)
    entries = pd.Series(False, index=dates)
    exits = pd.Series(False, index=dates)
    entries.iloc[0] = True  # Buy on first bar
    
    pf = vbt.Portfolio.from_signals(close, entries, exits, freq="1h", init_cash=10000.0, fees=0.001, slippage=0.001)
    
    # Hand calculation:
    # cash = 10000
    # price = 100
    # slippage = 0.001 -> exec price = 100 * 1.001 = 100.1
    # max units = 10000 / 100.1 = 99.9000999...
    # fees = 0.001
    
    trade_count = len(pf.trades)
    final_val = pf.value().iloc[-1]
    
    print(f"Test 1 (Buy & Hold) -> Trades: {trade_count}, Final Value: {final_val:.2f}")
    assert trade_count == 1, "Should have 1 trade"
    return pf

def test_2_always_flat():
    dates, prices, _ = generate_synthetic_data()
    close = pd.Series(prices, index=dates)
    entries = pd.Series(False, index=dates)
    exits = pd.Series(False, index=dates)
    
    pf = vbt.Portfolio.from_signals(close, entries, exits, freq="1h", init_cash=10000.0, fees=0.001, slippage=0.001)
    
    trade_count = len(pf.trades)
    final_val = pf.value().iloc[-1]
    
    print(f"Test 2 (Always Flat) -> Trades: {trade_count}, Final Value: {final_val:.2f}")
    assert trade_count == 0, "Should have 0 trades"
    assert final_val == 10000.0, "Value should remain init_cash"
    return pf

def test_3_alternating_signal():
    dates, prices, _ = generate_synthetic_data()
    close = pd.Series(prices, index=dates)
    entries = pd.Series(False, index=dates)
    exits = pd.Series(False, index=dates)
    
    for i in range(10, 90, 20):
        entries.iloc[i] = True
        exits.iloc[i+10] = True
        
    pf = vbt.Portfolio.from_signals(close, entries, exits, freq="1h", init_cash=10000.0, fees=0.0, slippage=0.0)
    
    trade_count = len(pf.trades)
    print(f"Test 3 (Alternating) -> Trades: {trade_count}, Final Value: {pf.value().iloc[-1]:.2f}")
    # 4 trades expected (10-20, 30-40, 50-60, 70-80)
    assert trade_count == 4, f"Expected 4 trades, got {trade_count}"
    return pf

def test_4_synthetic_trend():
    dates, prices, _ = generate_synthetic_data()
    close = pd.Series(prices, index=dates)
    
    # Strategy: SMA crossover
    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    # It will never cross if it's a straight line, so just use >
    entries = sma10 > sma20
    exits = sma10 < sma20
    
    pf = vbt.Portfolio.from_signals(close, entries, exits, freq="1h", init_cash=10000.0, fees=0.0, slippage=0.0)
    ret = pf.returns().sum()
    print(f"Test 4 (Synthetic Trend) -> Trades: {len(pf.trades)}, Return: {ret*100:.2f}%")
    assert ret > 0, "Trend following on perfect trend should be positive"
    return pf

def test_5_synthetic_mr():
    dates, _, mr_prices = generate_synthetic_data()
    close = pd.Series(mr_prices, index=dates)
    
    # Strategy: buy at 100, sell at 102
    entries = close == 100
    exits = close == 102
    
    pf = vbt.Portfolio.from_signals(close, entries, exits, freq="1h", init_cash=10000.0, fees=0.0, slippage=0.0)
    ret = pf.returns().sum()
    final_val = pf.value().iloc[-1]
    print(f"Test 5 (Synthetic MR) -> Trades: {len(pf.trades)}, Final Value: {final_val:.2f}, Return: {ret*100:.2f}%")
    assert ret > 0, "MR on perfect oscillation should be positive"
    return pf

def test_signal_alignment():
    dates, prices, _ = generate_synthetic_data()
    close = pd.Series(prices, index=dates)
    entries = pd.Series(False, index=dates)
    exits = pd.Series(False, index=dates)
    
    entries.iloc[5] = True
    exits.iloc[10] = True
    
    pf = vbt.Portfolio.from_signals(close, entries, exits, freq="1h", fees=0.0, slippage=0.0)
    records = pf.trades.records_readable
    
    entry_idx = records["Entry Timestamp"].iloc[0]
    entry_price = records["Avg Entry Price"].iloc[0]
    
    print(f"Test 6 (Alignment) -> Signal at index 5. Executed at index {entry_idx} with price {entry_price}")
    # VectorBT `from_signals` defaults to executing at the exact same index where signal is True, using the provided `price` (which defaults to `close`).
    # Thus signal at index 5 executes at index 5. 
    # If the signal was calculated using close[5], this implies trading exactly AT the close of the bar.
    # We must document this behavior.
    assert entry_idx == dates[5], "Execution should match signal index exactly in default VBT"
    assert entry_price == close.iloc[5], "Execution price should match close price of signal bar"

def test_continuous_sizing():
    dates, prices, _ = generate_synthetic_data()
    close = pd.Series(prices, index=dates)
    entries = pd.Series(False, index=dates)
    exits = pd.Series(False, index=dates)
    sizes = pd.Series(0.0, index=dates)
    
    entries.iloc[5] = True
    sizes.iloc[5] = 0.5 # 50% entry
    
    entries.iloc[10] = True
    sizes.iloc[10] = 1.0 # increase to 100%
    
    exits.iloc[15] = True
    sizes.iloc[15] = 0.5 # decrease to 50%
    
    exits.iloc[20] = True
    sizes.iloc[20] = 0.0 # flat
    
    pf = vbt.Portfolio.from_signals(
        close, entries, exits, size=sizes, size_type="percent", 
        freq="1h", init_cash=10000.0, fees=0.0, slippage=0.0
    )
    
    print("Test 7 (Sizing) -> Trades:")
    print(pf.trades.records_readable[["Entry Timestamp", "Exit Timestamp", "Size", "Avg Entry Price", "Avg Exit Price"]])
    
def run_all_tests():
    print("--- RUNNING ENGINE CERTIFICATION TESTS ---")
    test_1_buy_and_hold()
    test_2_always_flat()
    test_3_alternating_signal()
    test_4_synthetic_trend()
    test_5_synthetic_mr()
    test_signal_alignment()
    test_continuous_sizing()
    print("--- CERTIFICATION TESTS COMPLETE ---")

if __name__ == "__main__":
    run_all_tests()
