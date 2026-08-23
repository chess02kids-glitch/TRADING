"""Momentum pattern tests: exact placement, shift(1) timing, forward returns."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sandbox.pattern_research.patterns.momentum import (
    compute_forward_return,
    detect_higher_high_higher_low,
    detect_lower_low_lower_high,
    detect_momentum_combined,
)
from .conftest import make_candles


def _staircase_up(n=8):
    # each bar's high and low strictly above the previous bar's
    return make_candles([(10 + i, 11 + i, 9 + i, 10.5 + i, 100.0) for i in range(n)])


def _staircase_down(n=8):
    return make_candles([(30 - i, 31 - i, 29 - i, 29.5 - i, 100.0) for i in range(n)])


def test_hh_hl_fires_only_after_three_rising_bars():
    candles = _staircase_up(8)
    sig = detect_higher_high_higher_low(candles, lookback=3)
    # raw condition first true at bar 3 (needs bars 0..3); shift(1) -> bar 4
    assert list(sig.values) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert sig.index.equals(candles.index)
    assert set(np.unique(sig.values)) <= {0, 1}


def test_ll_lh_fires_only_after_three_falling_bars():
    candles = _staircase_down(8)
    sig = detect_lower_low_lower_high(candles, lookback=3)
    assert list(sig.values) == [0, 0, 0, 0, -1, -1, -1, -1]


def test_no_signal_on_flat_market():
    candles = make_candles([(10, 11, 9, 10, 100.0) for _ in range(10)])
    assert detect_higher_high_higher_low(candles).abs().sum() == 0
    assert detect_lower_low_lower_high(candles).abs().sum() == 0


def test_mixed_structure_produces_no_signal():
    # highs rise but lows fall -> not HH/HL, not LL/LH
    rows = [(10, 11 + i, 9 - i, 10, 100.0) for i in range(6)]
    candles = make_candles(rows)
    assert detect_higher_high_higher_low(candles).abs().sum() == 0
    assert detect_lower_low_lower_high(candles).abs().sum() == 0


def test_combined_is_sum_and_mutually_exclusive(synthetic_candles):
    up = detect_higher_high_higher_low(synthetic_candles)
    down = detect_lower_low_lower_high(synthetic_candles)
    combined = detect_momentum_combined(synthetic_candles)
    assert ((up != 0) & (down != 0)).sum() == 0
    assert combined.equals((up + down).rename("momentum"))


def test_no_lookahead_signal_depends_only_on_past(synthetic_candles):
    """Truncating the future must not change any earlier signal value."""
    full = detect_momentum_combined(synthetic_candles)
    cut = 1500
    partial = detect_momentum_combined(synthetic_candles.iloc[:cut])
    assert list(partial.values) == list(full.iloc[:cut].values)


def test_mutating_future_bars_cannot_change_past_signals(synthetic_candles):
    tampered = synthetic_candles.copy()
    tampered.iloc[2000:] *= 1.5
    base = detect_momentum_combined(synthetic_candles).iloc[:2000]
    assert list(detect_momentum_combined(tampered).iloc[:2000].values) == list(base.values)


def test_lookback_validation():
    with pytest.raises(ValueError):
        detect_higher_high_higher_low(_staircase_up(5), lookback=0)
    with pytest.raises(ValueError):
        detect_lower_low_lower_high(_staircase_down(5), lookback=0)


def test_missing_columns_raise():
    df = pd.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(ValueError):
        detect_higher_high_higher_low(df)


def test_empty_candles_return_empty_series():
    from sandbox.pattern_research.data_loader import empty_candles
    sig = detect_higher_high_higher_low(empty_candles())
    assert len(sig) == 0


# --- compute_forward_return -------------------------------------------------
def test_forward_return_math_and_correct_flag():
    closes = [100.0, 110.0, 99.0, 99.0]
    candles = make_candles([(c, c + 1, c - 1, c, 10.0) for c in closes])
    signal = pd.Series([1, -1, 0, 1], index=candles.index)
    out = compute_forward_return(candles, signal, horizon=1)
    # bar 3 has no t+1 -> dropped; bar 2 has signal 0 -> dropped
    assert list(out.index) == list(candles.index[:2])
    assert out.loc[candles.index[0], "forward_return"] == pytest.approx(0.10)
    assert out.loc[candles.index[0], "correct"] == 1          # +1 signal, up move
    assert out.loc[candles.index[1], "forward_return"] == pytest.approx(-0.1)
    assert out.loc[candles.index[1], "correct"] == 1          # -1 signal, down move


def test_forward_return_horizon_2_and_3():
    closes = [100.0, 101.0, 104.0, 90.0, 90.0]
    candles = make_candles([(c, c + 1, c - 1, c, 10.0) for c in closes])
    signal = pd.Series([1] * len(closes), index=candles.index)
    h2 = compute_forward_return(candles, signal, horizon=2)
    assert len(h2) == 3
    assert h2.iloc[0]["forward_return"] == pytest.approx(0.04)
    h3 = compute_forward_return(candles, signal, horizon=3)
    assert len(h3) == 2
    assert h3.iloc[0]["forward_return"] == pytest.approx(-0.10)
    assert h3.iloc[0]["correct"] == 0


def test_forward_return_include_flat_keeps_zero_signals():
    candles = make_candles([(10, 11, 9, 10 + i, 5.0) for i in range(5)])
    signal = pd.Series([0, 0, 1, 0, 0], index=candles.index)
    kept = compute_forward_return(candles, signal, horizon=1, include_flat=True)
    assert len(kept) == 4 and (kept["signal"] == 0).sum() == 3


def test_forward_return_zero_move_is_scored_incorrect():
    candles = make_candles([(10, 11, 9, 10.0, 5.0), (10, 11, 9, 10.0, 5.0)])
    signal = pd.Series([1, 0], index=candles.index)
    out = compute_forward_return(candles, signal)
    assert out.iloc[0]["forward_return"] == 0.0
    assert out.iloc[0]["correct"] == 0


def test_forward_return_rejects_bad_horizon():
    candles = make_candles([(10, 11, 9, 10, 5.0)] * 3)
    with pytest.raises(ValueError):
        compute_forward_return(candles, pd.Series([1, 1, 1], index=candles.index), horizon=0)
