"""Candlestick pattern tests — hand-built bars with known geometry."""
from __future__ import annotations

import pandas as pd
import pytest

from sandbox.pattern_research.patterns.candlestick import (
    detect_bearish_engulfing,
    detect_bullish_engulfing,
    detect_doji,
    detect_hammer,
)
from .conftest import make_candles


def test_bullish_engulfing_detected_and_shifted():
    rows = [
        (10.0, 10.5, 9.5, 10.0, 100.0),   # filler
        (10.0, 10.2, 9.0, 9.2, 100.0),    # bearish body 10.0 -> 9.2
        (9.0, 11.0, 8.9, 10.5, 100.0),    # bullish body 9.0 -> 10.5 engulfs
        (10.5, 10.6, 10.4, 10.5, 100.0),  # filler
    ]
    sig = detect_bullish_engulfing(make_candles(rows))
    assert list(sig.values) == [0, 0, 0, 1]  # completed at bar 2, emitted at bar 3


def test_bearish_engulfing_detected():
    rows = [
        (10.0, 10.5, 9.5, 10.0, 100.0),
        (9.0, 10.3, 8.9, 10.0, 100.0),    # bullish body 9.0 -> 10.0
        (10.2, 10.4, 8.5, 8.8, 100.0),    # bearish body 10.2 -> 8.8 engulfs
        (8.8, 8.9, 8.7, 8.8, 100.0),
    ]
    sig = detect_bearish_engulfing(make_candles(rows))
    assert list(sig.values) == [0, 0, 0, -1]


def test_engulfing_requires_opposite_prior_colour():
    # both candles bullish -> no bullish engulfing even if body is bigger
    rows = [
        (9.5, 10.0, 9.4, 9.8, 100.0),
        (9.0, 11.0, 8.9, 10.5, 100.0),
        (10.5, 10.6, 10.4, 10.5, 100.0),
    ]
    assert detect_bullish_engulfing(make_candles(rows)).abs().sum() == 0


def test_engulfing_requires_strict_growth():
    # identical bodies (mirrored) do not engulf
    rows = [
        (10.0, 10.5, 9.5, 9.0, 100.0),   # bearish 10 -> 9
        (9.0, 10.5, 8.5, 10.0, 100.0),   # bullish 9 -> 10, exactly equal body
        (10.0, 10.1, 9.9, 10.0, 100.0),
    ]
    assert detect_bullish_engulfing(make_candles(rows)).abs().sum() == 0


def test_doji_threshold_rule():
    rows = [
        (100.0, 101.0, 99.0, 100.05, 10.0),  # |c-o|/range = 0.05/2 = 0.025 < 0.1 -> doji
        (100.0, 101.0, 99.0, 100.9, 10.0),   # 0.9/2 = 0.45 -> not a doji
        (100.0, 100.0, 100.0, 100.0, 10.0),  # zero range -> excluded (no div by zero)
        (100.0, 101.0, 99.0, 100.0, 10.0),
    ]
    sig = detect_doji(make_candles(rows))
    assert list(sig.values) == [0, 1, 0, 0]


def test_doji_threshold_is_configurable():
    rows = [(100.0, 101.0, 99.0, 100.3, 10.0), (100.0, 101.0, 99.0, 100.0, 10.0)]
    candles = make_candles(rows)
    assert detect_doji(candles, threshold=0.1).sum() == 0   # 0.15 not < 0.1
    assert detect_doji(candles, threshold=0.2).sum() == 1   # 0.15 < 0.2
    with pytest.raises(ValueError):
        detect_doji(candles, threshold=0.0)


def test_hammer_geometry():
    # range 10 (90..100), body 100-98=2 in the upper 30% (>= 97),
    # lower shadow 8 > 2*2, upper shadow 0 < 0.5*2
    rows = [
        (98.0, 100.0, 90.0, 100.0, 10.0),
        (100.0, 100.5, 99.5, 100.0, 10.0),
    ]
    sig = detect_hammer(make_candles(rows))
    assert list(sig.values) == [0, 1]


def test_hammer_rejects_long_upper_shadow_and_low_body():
    rows = [
        (98.0, 110.0, 90.0, 100.0, 10.0),   # big upper shadow -> reject
        (90.5, 100.0, 90.0, 91.0, 10.0),    # body at the bottom of the range -> reject
        (100.0, 100.0, 100.0, 100.0, 10.0),  # zero range/body -> reject
        (100.0, 100.5, 99.5, 100.2, 10.0),
    ]
    assert detect_hammer(make_candles(rows)).abs().sum() == 0


def test_all_detectors_have_no_lookahead(synthetic_candles):
    cut = 1200
    for fn in (detect_bullish_engulfing, detect_bearish_engulfing,
               detect_doji, detect_hammer):
        full = fn(synthetic_candles).iloc[:cut]
        partial = fn(synthetic_candles.iloc[:cut])
        assert list(partial.values) == list(full.values), fn.__name__


def test_detectors_reject_missing_columns():
    df = pd.DataFrame({"close": [1.0, 2.0]})
    for fn in (detect_bullish_engulfing, detect_bearish_engulfing,
               detect_doji, detect_hammer):
        with pytest.raises(ValueError):
            fn(df)
