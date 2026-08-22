"""Unit tests for phase9a.direction_calculator (past-only direction + returns).

All synthetic data — no DB, no network. Covers: direction sign & columns,
forward-return arithmetic, the "skip missing t+N bar" rule, multi-asset
matching via an ``asset`` column, timestamp normalization (ISO string vs
epoch-ms), and empty-input edge cases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phase9a.direction_calculator import (
    compute_breakout_direction,
    compute_forward_returns,
)


def _candles(prices, start="2024-01-15T00:00:00Z"):
    """Build an OHLCV frame from a sequence of close prices (1h bars).

    Each bar: open=prev close (first open = price), high/low straddle by a
    fixed spread so high-low range is positive, close=given price.
    """
    rows = []
    prev = prices[0]
    for i, p in enumerate(prices):
        o = prev
        c = p
        hi = max(o, c) + 1.0
        lo = min(o, c) - 1.0
        ts = pd.Timestamp(start) + pd.Timedelta(hours=i)
        rows.append({
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": float(o), "high": float(hi), "low": float(lo),
            "close": float(c), "volume": 1000.0,
        })
        prev = c
    return pd.DataFrame(rows)


def _breakout_row(ts, asset="BTC/USDT", predicted=10.0, actual=30.0, regime="high"):
    return {
        "timestamp": ts, "asset": asset, "timeframe": "1h",
        "har_predicted_range": predicted, "actual_range": actual,
        "regime": regime, "breakout_flag": 1,
    }


# ---------------------------------------------------------------------------

class TestComputeBreakoutDirection:

    def test_up_and_down_direction(self):
        # Bar at index 2 closes higher than open -> +1; bar at index 5 closes lower -> -1.
        prices = [100, 100, 110, 110, 110, 90, 90]
        candles = _candles(prices)
        ts0 = candles["timestamp"].iloc[0]
        br = pd.DataFrame([
            _breakout_row(candles["timestamp"].iloc[2]),  # open=100, close=110 -> UP
            _breakout_row(candles["timestamp"].iloc[5]),  # open=110, close=90  -> DOWN
        ])
        out = compute_breakout_direction(candles, br)
        assert list(out.columns) == [
            "timestamp", "asset", "breakout_direction", "close_at_breakout",
            "open_at_breakout", "actual_range_at_breakout",
            "har_predicted_at_breakout", "regime",
        ]
        assert len(out) == 2
        dirs = dict(zip(out["timestamp"], out["breakout_direction"]))
        assert dirs[candles["timestamp"].iloc[2]] == 1
        assert dirs[candles["timestamp"].iloc[5]] == -1
        # close/open carried through
        row0 = out.iloc[0]
        assert row0["close_at_breakout"] == 110.0
        assert row0["open_at_breakout"] == 100.0
        assert row0["har_predicted_at_breakout"] == 10.0

    def test_close_equal_open_is_up(self):
        # Spec: close >= open -> +1.
        rows = [{"timestamp": "2024-01-15T00:00:00Z", "open": 100.0, "high": 101.0,
                 "low": 99.0, "close": 100.0, "volume": 1.0}]
        candles = pd.DataFrame(rows)
        br = pd.DataFrame([_breakout_row("2024-01-15T00:00:00Z")])
        out = compute_breakout_direction(candles, br)
        assert out["breakout_direction"].iloc[0] == 1

    def test_unmatched_breakout_dropped(self):
        candles = _candles([100, 101, 102])
        br = pd.DataFrame([_breakout_row("2099-01-01T00:00:00Z")])  # not in candles
        out = compute_breakout_direction(candles, br)
        assert len(out) == 0

    def test_empty_breakout_rows(self):
        candles = _candles([100, 101, 102])
        out = compute_breakout_direction(candles, pd.DataFrame())
        assert out.empty

    def test_epoch_ms_timestamp_supported(self):
        # Candle timestamp as epoch-ms int should match an ISO breakout timestamp.
        base_ms = 1705276800000  # 2024-01-15T00:00:00Z
        candles = pd.DataFrame([
            {"timestamp": base_ms, "open": 100.0, "high": 112.0, "low": 99.0,
             "close": 110.0, "volume": 1.0},
        ])
        br = pd.DataFrame([_breakout_row("2024-01-15T00:00:00Z")])
        out = compute_breakout_direction(candles, br)
        assert len(out) == 1
        assert out["breakout_direction"].iloc[0] == 1


class TestComputeForwardReturns:

    def test_forward_return_arithmetic_and_sign(self):
        prices = [100, 110, 99, 105]  # breakout at index 0 (close 100)
        candles = _candles(prices)
        br = pd.DataFrame([_breakout_row(candles["timestamp"].iloc[0])])
        out = compute_forward_returns(candles, br, horizons=[1, 2, 3])
        # close0=100, close1=110 -> +10% (+1), close2=99 -> -1% (-1), close3=105 -> +5% (+1)
        by_h = {int(r["horizon"]): r for _, r in out.iterrows()}
        assert by_h[1]["forward_return"] == pytest.approx(110 / 100 - 1)
        assert by_h[1]["forward_direction"] == 1
        assert by_h[2]["forward_return"] == pytest.approx(99 / 100 - 1)
        assert by_h[2]["forward_direction"] == -1
        assert by_h[3]["forward_return"] == pytest.approx(105 / 100 - 1)
        assert by_h[3]["forward_direction"] == 1

    def test_missing_tN_bar_skipped(self):
        # Breakout at the last bar -> no forward bars -> empty result.
        prices = [100, 110, 120]
        candles = _candles(prices)
        br = pd.DataFrame([_breakout_row(candles["timestamp"].iloc[-1])])
        out = compute_forward_returns(candles, br, horizons=[1, 2, 3])
        assert out.empty  # no t+N bar exists

    def test_partial_horizons_near_end(self):
        # Breakout at second-to-last bar: only horizon 1 exists.
        prices = [100, 110, 120]
        candles = _candles(prices)
        br = pd.DataFrame([_breakout_row(candles["timestamp"].iloc[-2])])
        out = compute_forward_returns(candles, br, horizons=[1, 2, 3])
        assert set(out["horizon"]) == {1}

    def test_no_future_data_used(self):
        # Forward return at t+1 must use close_{t+1}, not any later bar.
        prices = [100, 150, 999, 999]
        candles = _candles(prices)
        br = pd.DataFrame([_breakout_row(candles["timestamp"].iloc[0])])
        out = compute_forward_returns(candles, br, horizons=[1])
        row = out.iloc[0]
        assert row["forward_return"] == pytest.approx(150 / 100 - 1)  # not 999

    def test_empty_inputs(self):
        assert compute_forward_returns(_candles([100, 101]), pd.DataFrame()).empty


class TestMultiAsset:

    def test_asset_column_filters_correctly(self):
        # Two assets interleaved in one frame; each breakout matches its own series.
        base = pd.Timestamp("2024-01-15T00:00:00Z")
        rows = []
        for i in range(5):
            ts = (base + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            rows.append({"timestamp": ts, "asset": "BTC/USDT", "open": 100.0,
                         "high": 105.0, "low": 99.0, "close": 102.0, "volume": 1.0})
            rows.append({"timestamp": ts, "asset": "ETH/USDT", "open": 50.0,
                         "high": 52.0, "low": 48.0, "close": 51.0, "volume": 1.0})
        candles = pd.DataFrame(rows)
        br = pd.DataFrame([
            _breakout_row(base.strftime("%Y-%m-%dT%H:%M:%SZ"), asset="BTC/USDT"),
            _breakout_row(base.strftime("%Y-%m-%dT%H:%M:%SZ"), asset="ETH/USDT"),
        ])
        out = compute_breakout_direction(candles, br)
        assert len(out) == 2
        assert set(out["asset"]) == {"BTC/USDT", "ETH/USDT"}

    def test_forward_returns_stay_within_asset(self):
        # ETH forward bars must not borrow BTC candles.
        base = pd.Timestamp("2024-01-15T00:00:00Z")
        rows = []
        for i in range(4):
            ts = (base + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            rows.append({"timestamp": ts, "asset": "BTC/USDT", "open": 100.0,
                         "high": 105.0, "low": 99.0, "close": 100.0 + i, "volume": 1.0})
            rows.append({"timestamp": ts, "asset": "ETH/USDT", "open": 50.0,
                         "high": 52.0, "low": 48.0, "close": 50.0 + i, "volume": 1.0})
        candles = pd.DataFrame(rows)
        br = pd.DataFrame([_breakout_row(
            base.strftime("%Y-%m-%dT%H:%M:%SZ"), asset="ETH/USDT")])
        out = compute_forward_returns(candles, br, horizons=[1])
        # ETH t+1 close = 51, base = 50 -> +2%
        assert out.iloc[0]["forward_return"] == pytest.approx(51 / 50 - 1)
        assert out.iloc[0]["asset"] == "ETH/USDT"
