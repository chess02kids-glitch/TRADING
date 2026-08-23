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
    detect_momentum_fade_combined,
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


# --- fade (mean-reversion) reading ------------------------------------------
def test_momentum_fade_is_inverse_of_momentum(synthetic_candles):
    """The fade series is the exact negation, on the same bars, with the same
    no-look-ahead guarantees (inherited, not reimplemented)."""
    momentum = detect_momentum_combined(synthetic_candles)
    fade = detect_momentum_fade_combined(synthetic_candles)

    assert fade.name == "momentum_fade"
    assert fade.index.equals(momentum.index)
    assert list(fade.values) == [-v for v in momentum.values]
    assert set(np.unique(fade.values)) <= {-1, 0, 1}
    # same support: exactly the same bars fire
    assert list((fade != 0).values) == list((momentum != 0).values)
    # HH/HL bars are sold, LL/LH bars are bought
    assert (fade[momentum == 1] == -1).all()
    assert (fade[momentum == -1] == 1).all()

    # no new look-ahead: truncating the future leaves earlier values identical
    cut = 1500
    partial = detect_momentum_fade_combined(synthetic_candles.iloc[:cut])
    assert list(partial.values) == list(fade.iloc[:cut].values)

    # mutating bars from 2000 onward cannot change the first 2000 signals
    tampered = synthetic_candles.copy()
    tampered.iloc[2000:] *= 1.5
    again = detect_momentum_fade_combined(tampered)
    assert list(again.iloc[:2000].values) == list(fade.iloc[:2000].values)


def test_momentum_fade_hit_rate_is_the_complement(synthetic_candles):
    """Same events as momentum; 'correct' flips except on flat forward moves,
    which both readings score 0 (a flat move is 'wrong' for ±1 either way)."""
    momentum = detect_momentum_combined(synthetic_candles)
    fade = detect_momentum_fade_combined(synthetic_candles)
    m = compute_forward_return(synthetic_candles, momentum, horizon=1)
    f = compute_forward_return(synthetic_candles, fade, horizon=1)

    assert list(m.index) == list(f.index)            # identical event sets
    flat = m["forward_return"] == 0.0
    assert (m.loc[flat, "correct"] == 0).all() and (f.loc[flat, "correct"] == 0).all()
    nonflat = ~flat
    assert (f.loc[nonflat, "correct"].to_numpy()
            == 1 - m.loc[nonflat, "correct"].to_numpy()).all()
    # hence hit_fade + hit_continuation = fraction of non-flat events (~1.0
    # on this data, where exactly-zero forward returns never occur)
    assert f["correct"].mean() + m["correct"].mean() == pytest.approx(1.0 - flat.mean())


def test_fade_on_a_staircase_sells_the_uptrend():
    assert list(detect_momentum_fade_combined(_staircase_up(8)).values) == \
        [0, 0, 0, 0, -1, -1, -1, -1]
    assert list(detect_momentum_fade_combined(_staircase_down(8)).values) == \
        [0, 0, 0, 0, 1, 1, 1, 1]


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
