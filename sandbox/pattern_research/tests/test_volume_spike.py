"""Volume-spike pattern tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sandbox.pattern_research.patterns.volume_spike import (
    compute_volume_ratio,
    detect_volume_spike,
)
from .conftest import make_candles


def _base_rows(n, volume=100.0, bullish=True):
    if bullish:
        return [(10.0, 10.5, 9.5, 10.2, volume) for _ in range(n)]
    return [(10.0, 10.5, 9.5, 9.8, volume) for _ in range(n)]


def test_volume_ratio_is_one_on_constant_volume():
    candles = make_candles(_base_rows(30))
    ratio = compute_volume_ratio(candles, window=20)
    assert ratio.iloc[:19].isna().all()          # window not full yet
    assert ratio.iloc[19:].round(9).eq(1.0).all()


def test_volume_ratio_reacts_to_a_spike():
    rows = _base_rows(25)
    rows[24] = (10.0, 10.5, 9.5, 10.2, 1000.0)
    ratio = compute_volume_ratio(make_candles(rows), window=20)
    # mean of 19x100 + 1000 = 145 -> 1000/145
    assert ratio.iloc[24] == pytest.approx(1000.0 / 145.0)


def test_spike_direction_and_shift():
    rows = _base_rows(26)
    rows[24] = (10.0, 11.0, 9.5, 10.9, 1000.0)     # bullish spike
    sig = detect_volume_spike(make_candles(rows), threshold=2.0, window=20)
    assert sig.iloc[24] == 0     # not visible on the spike bar itself (shift(1))
    assert sig.iloc[25] == 1     # emitted on the next bar

    rows[24] = (11.0, 11.0, 9.5, 9.6, 1000.0)      # bearish spike
    sig = detect_volume_spike(make_candles(rows), threshold=2.0, window=20)
    assert sig.iloc[25] == -1


def test_no_spike_below_threshold():
    rows = _base_rows(26)
    rows[24] = (10.0, 10.5, 9.5, 10.2, 120.0)
    assert detect_volume_spike(make_candles(rows), threshold=2.0, window=20).abs().sum() == 0


def test_zero_body_spike_is_not_signalled():
    rows = _base_rows(26)
    rows[24] = (10.0, 10.5, 9.5, 10.0, 1000.0)   # doji body -> sign 0
    assert detect_volume_spike(make_candles(rows), threshold=2.0, window=20).abs().sum() == 0


def test_zero_mean_volume_does_not_divide_by_zero():
    rows = [(10.0, 10.5, 9.5, 10.2, 0.0) for _ in range(25)]
    ratio = compute_volume_ratio(make_candles(rows), window=20)
    assert ratio.isna().all()
    assert detect_volume_spike(make_candles(rows)).abs().sum() == 0


def test_no_lookahead(synthetic_candles):
    cut = 900
    full = detect_volume_spike(synthetic_candles).iloc[:cut]
    partial = detect_volume_spike(synthetic_candles.iloc[:cut])
    assert list(partial.values) == list(full.values)


def test_parameter_validation():
    candles = make_candles(_base_rows(25))
    with pytest.raises(ValueError):
        compute_volume_ratio(candles, window=1)
    with pytest.raises(ValueError):
        detect_volume_spike(candles, threshold=0.0)
    with pytest.raises(ValueError):
        compute_volume_ratio(pd.DataFrame({"close": [1.0]}))
