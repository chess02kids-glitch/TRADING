"""Tests for phase9a.continuation_tester (temporal stability + G1-G6 gates)."""
from __future__ import annotations

import pandas as pd
import pytest

from phase9a.continuation_tester import (
    compute_temporal_stability,
    run_all_gate_checks,
)


def make_df(events, base="2024-01-01T00:00:00Z"):
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


def _both_assets(n_each, hit=True, horizon=1):
    events = []
    for _ in range(n_each):
        events.append(("BTC/USDT", 1, 1 if hit else -1, horizon))
        events.append(("ETH/USDT", 1, 1 if hit else -1, horizon))
    return make_df(events)


class TestTemporalStability:

    def test_stable_when_all_hit(self):
        df = _both_assets(20)  # 40 events, all hits
        stab = compute_temporal_stability(df, horizon=1)
        assert stab["older"] == pytest.approx(1.0)
        assert stab["is_stable"] is True
        assert stab["degrading"] is False

    def test_degrading_when_recent_collapses(self):
        # First 24 events hit, last 24 miss.
        rows = []
        b = pd.Timestamp("2024-01-01T00:00:00Z")
        for i in range(48):
            hit = i < 24
            ts = (b + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            rows.append({"breakout_timestamp": ts, "asset": "BTC/USDT",
                         "breakout_direction": 1, "horizon": 1,
                         "forward_direction": 1 if hit else -1,
                         "forward_return": 0.01, "breakout_close_price": 100.0})
        df = pd.DataFrame(rows)
        stab = compute_temporal_stability(df, horizon=1)
        assert stab["older"] > 0.5
        assert stab["recent"] < 0.5
        assert stab["is_stable"] is False
        assert stab["degrading"] is True


class TestRunAllGateChecks:

    def test_signal_found_when_all_pass(self):
        df = _both_assets(35, hit=True)  # 35 events each asset (>= 30)
        gates = run_all_gate_checks(df, horizon=1)
        assert gates["G1"] is True
        assert gates["G2"] is True
        assert gates["G3"] is True
        assert gates["G4"] is True
        assert gates["G5"] is True
        assert gates["G6"] is True
        assert gates["all_pass"] is True
        assert gates["verdict"] == "SIGNAL FOUND"
        assert "hit_rate" in gates["details"]
        assert "dm" in gates["details"]

    def test_closed_when_all_miss(self):
        df = _both_assets(35, hit=False)
        gates = run_all_gate_checks(df, horizon=1)
        assert gates["G1"] is False
        assert gates["verdict"] == "CLOSED"

    def test_g6_fails_with_few_events(self):
        df = _both_assets(10, hit=True)  # 20 each < 30
        gates = run_all_gate_checks(df, horizon=1)
        assert gates["G6"] is False
        assert gates["verdict"] == "CLOSED"

    def test_g3_fails_single_asset(self):
        df = make_df([("BTC/USDT", 1, 1, 1) for _ in range(40)])
        gates = run_all_gate_checks(df, horizon=1)
        assert gates["G3"] is False  # ETH missing
