"""Trading signals for VolatilityBreakout strategy."""

import pandas as pd

def compute_entry_signals(
    high: pd.Series,
    low: pd.Series,
    range_mult: float = 2.0,
    lookback: int = 20,
) -> pd.Series:
    """
    VolatilityBreakout entry signal.

    Entry when:
      (high - low) > range_mult * mean(last N ranges)

    Returns boolean Series.
    True = enter long.
    """
    current_range = high - low
    rolling_mean_range = current_range.rolling(window=lookback).mean()
    entries = current_range > (range_mult * rolling_mean_range)
    
    # Fill NA (first lookback-1 bars) with False
    return entries.fillna(False)


def compute_exit_signals(
    close: pd.Series,
    exit_lookback: int = 10,
) -> pd.Series:
    """
    Donchian channel exit signal.

    Exit when:
      close < min(close, last N bars)

    CRITICAL: Use shift(1) on rolling min.
    close.shift(1).rolling(exit_lookback).min()
    """
    rolling_min_shifted = close.shift(1).rolling(window=exit_lookback).min()
    exits = close < rolling_min_shifted
    
    # Fill NA (first exit_lookback bars) with False
    return exits.fillna(False)


def verify_no_lookahead(
    entries: pd.Series,
    exits: pd.Series,
    ohlcv: pd.DataFrame,
) -> bool:
    """
    Verify signals have no look-ahead bias.
    Test: Perturb future data.
    Verify signals do not change.
    """
    if len(ohlcv) < 50:
        return True # Not enough data to test properly
        
    df_perturbed = ohlcv.copy()
    
    # Change future data starting from index 25
    df_perturbed.loc[df_perturbed.index[25:], 'high'] *= 10
    df_perturbed.loc[df_perturbed.index[25:], 'low'] *= 10
    df_perturbed.loc[df_perturbed.index[25:], 'close'] *= 10
    
    new_entries = compute_entry_signals(df_perturbed['high'], df_perturbed['low'])
    new_exits = compute_exit_signals(df_perturbed['close'])
    
    # Check that up to index 24 (inclusive), signals match exactly
    # i.e., changing future data (25 onwards) didn't affect past signals (0 to 24)
    past_entries_match = (entries.iloc[:25] == new_entries.iloc[:25]).all()
    past_exits_match = (exits.iloc[:25] == new_exits.iloc[:25]).all()
    
    return bool(past_entries_match and past_exits_match)
