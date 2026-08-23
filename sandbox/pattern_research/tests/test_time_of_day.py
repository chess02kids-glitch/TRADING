"""Time-of-day / day-of-week bias tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sandbox.pattern_research.patterns.time_of_day import (
    build_hour_signal,
    compute_daily_bias,
    compute_hourly_bias,
    find_best_hours,
)
from .conftest import make_candles


def _candles_with_hourly_pattern(days=30):
    """Up move only during UTC hour 5, flat-ish otherwise."""
    n = days * 24
    idx = pd.date_range("2024-01-01T00:00:00Z", periods=n, freq="1h", tz="UTC",
                        name="timestamp")
    rng = np.random.default_rng(3)
    steps = rng.normal(0.0, 0.001, size=n)
    steps[idx.hour == 5] = 0.01
    close = 100.0 * np.exp(np.cumsum(steps))
    df = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999,
                       "close": close, "volume": 1.0}, index=idx)
    return df


def test_hourly_bias_schema_and_sorting():
    candles = _candles_with_hourly_pattern()
    hourly = compute_hourly_bias(candles)
    assert list(hourly.columns) == ["hour", "mean_return", "win_rate", "n_observations"]
    assert set(hourly.index) == set(range(24))
    wr = hourly["win_rate"].dropna().tolist()
    assert wr == sorted(wr, reverse=True)
    assert hourly.iloc[0]["hour"] == 5          # the injected hour ranks first
    assert hourly.loc[5, "win_rate"] == 1.0
    assert int(hourly["n_observations"].sum()) == len(candles) - 1


def test_daily_bias_schema():
    candles = _candles_with_hourly_pattern()
    daily = compute_daily_bias(candles)
    assert list(daily.index) == list(range(7))
    assert daily.loc[0, "day_name"] == "Mon" and daily.loc[6, "day_name"] == "Sun"
    assert int(daily["n_observations"].sum()) == len(candles) - 1


def test_find_best_hours_applies_both_filters():
    hourly = pd.DataFrame({
        "hour": [0, 1, 2, 3],
        "mean_return": [0.1, 0.1, 0.1, 0.1],
        "win_rate": [0.60, 0.70, 0.54, 0.90],
        "n_observations": [500, 500, 500, 50],   # hour 3 has too few obs
    }, index=[0, 1, 2, 3])
    assert find_best_hours(hourly, min_win_rate=0.55) == [1, 0]
    assert find_best_hours(hourly, min_win_rate=0.65) == [1]
    assert find_best_hours(pd.DataFrame()) == []


def test_build_hour_signal_fires_on_the_preceding_bar():
    candles = _candles_with_hourly_pattern(days=2)
    sig = build_hour_signal(candles, [5])
    fired = candles.index[sig == 1]
    assert set(fired.hour) == {4}
    assert len(fired) == 2
    assert build_hour_signal(candles, []).abs().sum() == 0


def test_hour_signal_captures_the_biased_hour():
    from sandbox.pattern_research.patterns.momentum import compute_forward_return
    candles = _candles_with_hourly_pattern(days=40)
    events = compute_forward_return(candles, build_hour_signal(candles, [5]), horizon=1)
    assert len(events) == 40
    assert events["correct"].mean() == 1.0   # the injected edge is fully captured


def test_bias_tables_handle_empty_and_bad_input():
    from sandbox.pattern_research.data_loader import empty_candles
    hourly = compute_hourly_bias(empty_candles())
    assert len(hourly) == 24 and hourly["n_observations"].sum() == 0
    assert compute_daily_bias(empty_candles())["n_observations"].sum() == 0
    with pytest.raises(TypeError):
        compute_hourly_bias(pd.DataFrame({"close": [1.0, 2.0]}))
