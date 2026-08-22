"""Tests for phase9a.direction_calculator (hit rates + temporal split)."""
from __future__ import annotations

import pandas as pd
import pytest

from phase9a.direction_calculator import (
    compute_hit_rate,
    split_temporal_windows,
)


def make_df(events, base="2024-01-01T00:00:00Z"):
    """events: list of (asset, breakout_direction, forward_direction, horizon)."""
    rows = []
    b = pd.Timestamp(base)
    for i, (asset, bdir, fdir, h) in enumerate(events):
        ts = (b + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append({
            "breakout_timestamp": ts, "asset": asset,
            "breakout_direction": bdir, "horizon": h,
            "forward_direction": fdir, "forward_return": 0.01 * fdir,
            "breakout_close_price": 100.0,
        })
    return pd.DataFrame(rows)


class TestComputeHitRate:

    def test_basic_hit_rate(self):
        df = make_df([
            ("BTC/USDT", 1, 1, 1), ("BTC/USDT", 1, 1, 1),
            ("BTC/USDT", -1, -1, 1), ("BTC/USDT", 1, -1, 1),
        ])
        hr = compute_hit_rate(df, horizon=1)
        assert hr["n_events"] == 4
        assert hr["n_correct"] == 3
        assert hr["overall_hit_rate"] == pytest.approx(0.75)

    def test_by_asset_and_by_direction(self):
        df = make_df([
            ("BTC/USDT", 1, 1, 1), ("ETH/USDT", 1, -1, 1),
            ("BTC/USDT", -1, -1, 1), ("ETH/USDT", -1, -1, 1),
        ])
        hr = compute_hit_rate(df, horizon=1)
        assert hr["by_asset"]["BTC/USDT"] == pytest.approx(1.0)
        assert hr["by_asset"]["ETH/USDT"] == pytest.approx(0.5)
        assert hr["by_direction"][1] == pytest.approx(0.5)
        assert hr["by_direction"][-1] == pytest.approx(1.0)

    def test_horizon_filter(self):
        df = make_df([("BTC/USDT", 1, 1, 1), ("BTC/USDT", 1, -1, 2)])
        hr1 = compute_hit_rate(df, horizon=1)
        hr2 = compute_hit_rate(df, horizon=2)
        assert hr1["n_events"] == 1 and hr1["overall_hit_rate"] == pytest.approx(1.0)
        assert hr2["n_events"] == 1 and hr2["overall_hit_rate"] == pytest.approx(0.0)

    def test_empty(self):
        df = pd.DataFrame([{
            "breakout_timestamp": "2024-01-01T00:00:00Z", "asset": "BTC/USDT",
            "breakout_direction": 1, "horizon": 1, "forward_direction": None,
            "forward_return": 0.0, "breakout_close_price": 100.0,
        }])
        hr = compute_hit_rate(df, horizon=1)
        assert hr["n_events"] == 0
        assert hr["overall_hit_rate"] == 0.0


class TestSplitTemporalWindows:

    def test_three_thirds(self):
        df = make_df([("BTC/USDT", 1, 1, 1) for _ in range(9)])
        older, middle, recent = split_temporal_windows(df)
        assert len(older) == 3 and len(middle) == 3 and len(recent) == 3
        # older holds the earliest timestamps
        assert older["breakout_timestamp"].iloc[0] < recent["breakout_timestamp"].iloc[-1]

    def test_small_input(self):
        df = make_df([("BTC/USDT", 1, 1, 1)])
        older, middle, recent = split_temporal_windows(df)
        assert len(older) + len(middle) + len(recent) == 1
