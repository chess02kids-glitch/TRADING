"""Unit tests for phase9a.continuation_tester (hit rates, stability, gates)."""
from __future__ import annotations

import pandas as pd
import pytest

from phase9a.continuation_tester import (
    BOTH_ASSETS,
    compute_degradation_flag,
    compute_hit_rate,
    compute_temporal_stability,
    merge_direction_returns,
    run_gate_checks,
)


def _build(events, horizon=1):
    """events: list of dicts with keys ts, asset, bdir, fdir, regime.

    Builds matching direction_df and returns_df at the given horizon.
    """
    d_rows, r_rows = [], []
    for i, e in enumerate(events):
        d_rows.append({
            "timestamp": e["ts"], "asset": e["asset"],
            "breakout_direction": e["bdir"], "close_at_breakout": 100.0,
            "open_at_breakout": 100.0, "actual_range_at_breakout": 30.0,
            "har_predicted_at_breakout": 10.0, "regime": e.get("regime"),
        })
        r_rows.append({
            "timestamp": e["ts"], "asset": e["asset"], "horizon": horizon,
            "forward_return": 0.01 * e["fdir"], "forward_direction": e["fdir"],
        })
    return pd.DataFrame(d_rows), pd.DataFrame(r_rows)


def _ts(i):
    return (pd.Timestamp("2024-01-15T00:00:00Z") + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------

class TestHitRate:

    def test_basic_hit_rate(self):
        # 3 correct, 1 wrong -> 75%
        events = [
            {"ts": _ts(0), "asset": "BTC/USDT", "bdir": 1, "fdir": 1, "regime": "high"},
            {"ts": _ts(1), "asset": "BTC/USDT", "bdir": 1, "fdir": 1, "regime": "low"},
            {"ts": _ts(2), "asset": "BTC/USDT", "bdir": -1, "fdir": -1, "regime": "high"},
            {"ts": _ts(3), "asset": "BTC/USDT", "bdir": 1, "fdir": -1, "regime": "low"},
        ]
        d, r = _build(events)
        hr = compute_hit_rate(d, r, horizon=1)
        assert hr["n_events"] == 4
        assert hr["n_correct"] == 3
        assert hr["hit_rate"] == pytest.approx(0.75)
        assert hr["by_asset"]["BTC/USDT"] == pytest.approx(0.75)
        assert hr["by_asset_n"]["BTC/USDT"] == 4
        assert hr["by_regime"]["high"] == pytest.approx(1.0)
        assert hr["by_regime"]["low"] == pytest.approx(0.5)

    def test_by_asset_split(self):
        events = [
            {"ts": _ts(0), "asset": "BTC/USDT", "bdir": 1, "fdir": 1, "regime": "high"},
            {"ts": _ts(1), "asset": "ETH/USDT", "bdir": 1, "fdir": -1, "regime": "high"},
        ]
        d, r = _build(events)
        hr = compute_hit_rate(d, r, horizon=1)
        assert hr["by_asset"]["BTC/USDT"] == pytest.approx(1.0)
        assert hr["by_asset"]["ETH/USDT"] == pytest.approx(0.0)
        assert hr["hit_rate"] == pytest.approx(0.5)

    def test_zero_events(self):
        d = pd.DataFrame(columns=["timestamp", "asset", "breakout_direction", "regime"])
        r = pd.DataFrame(columns=["timestamp", "asset", "horizon", "forward_direction"])
        hr = compute_hit_rate(d, r, horizon=1)
        assert hr["n_events"] == 0
        assert hr["hit_rate"] == 0.0

    def test_horizon_filtering(self):
        events = [{"ts": _ts(0), "asset": "BTC/USDT", "bdir": 1, "fdir": 1, "regime": "high"}]
        d, r = _build(events, horizon=1)
        # Returns only exist at horizon 1; asking for horizon 2 yields 0 events.
        hr2 = compute_hit_rate(d, r, horizon=2)
        assert hr2["n_events"] == 0

    def test_zero_forward_return_is_a_miss(self):
        events = [{"ts": _ts(0), "asset": "BTC/USDT", "bdir": 1, "fdir": 0, "regime": "high"}]
        d, r = _build(events)
        hr = compute_hit_rate(d, r, horizon=1)
        assert hr["n_correct"] == 0


class TestTemporalStability:

    def test_thirds_split(self):
        # 9 events: first 3 all hit, middle 3 all hit, last 3 all miss.
        events = []
        for i in range(9):
            fdir = 1 if i < 6 else -1
            events.append({"ts": _ts(i), "asset": "BTC/USDT", "bdir": 1,
                           "fdir": fdir, "regime": "high"})
        d, r = _build(events)
        stab = compute_temporal_stability(d, r, horizon=1)
        assert stab["older_hit_rate"] == pytest.approx(1.0)
        assert stab["middle_hit_rate"] == pytest.approx(1.0)
        assert stab["recent_hit_rate"] == pytest.approx(0.0)
        assert stab["is_stable"] is False

    def test_all_stable(self):
        events = [{"ts": _ts(i), "asset": "BTC/USDT", "bdir": 1, "fdir": 1, "regime": "high"}
                  for i in range(9)]
        d, r = _build(events)
        stab = compute_temporal_stability(d, r, horizon=1)
        assert stab["is_stable"] is True


class TestDegradation:

    def test_degradation_flag_true(self):
        events = []
        for i in range(9):
            fdir = 1 if i < 6 else -1   # recent collapses
            events.append({"ts": _ts(i), "asset": "BTC/USDT", "bdir": 1,
                           "fdir": fdir, "regime": "high"})
        d, r = _build(events)
        assert compute_degradation_flag(d, r, horizon=1) is True

    def test_degradation_flag_false_when_stable(self):
        events = [{"ts": _ts(i), "asset": "BTC/USDT", "bdir": 1, "fdir": 1, "regime": "high"}
                  for i in range(9)]
        d, r = _build(events)
        assert compute_degradation_flag(d, r, horizon=1) is False


class TestGateChecks:

    def _hr(self, overall, by_asset, by_asset_n, by_regime=None):
        return {"hit_rate": overall, "n_events": sum(by_asset_n.values()),
                "n_correct": 0, "by_asset": by_asset,
                "by_asset_n": by_asset_n, "by_regime": by_regime or {}}

    def _temporal(self, o=0.6, m=0.6, r=0.6):
        return {"older_hit_rate": o, "middle_hit_rate": m, "recent_hit_rate": r,
                "is_stable": all(x > 0.5 for x in (o, m, r))}

    def test_all_pass_signal_found(self):
        hr = self._hr(0.6, {"BTC/USDT": 0.6, "ETH/USDT": 0.6},
                      {"BTC/USDT": 40, "ETH/USDT": 35})
        temp = self._temporal()
        dm = {"p_value": 0.01}
        gates = run_gate_checks(hr, temp, 75, dm_dict=dm)
        assert gates["G1"] is True
        assert gates["G2"] is True
        assert gates["G3"] is True
        assert gates["G4"] is True
        assert gates["G5"] is True
        assert gates["G6"] is True
        assert gates["all_pass"] is True
        assert gates["verdict"] == "SIGNAL FOUND"

    def test_g1_fails_below_55(self):
        hr = self._hr(0.52, {"BTC/USDT": 0.6, "ETH/USDT": 0.52},
                      {"BTC/USDT": 40, "ETH/USDT": 40})
        gates = run_gate_checks(hr, self._temporal(), 80, dm_dict={"p_value": 0.01})
        assert gates["G1"] is False
        assert gates["verdict"] == "CLOSED"

    def test_g2_placeholder_when_no_dm(self):
        hr = self._hr(0.6, {"BTC/USDT": 0.6, "ETH/USDT": 0.6},
                      {"BTC/USDT": 40, "ETH/USDT": 40})
        gates = run_gate_checks(hr, self._temporal(), 80, dm_dict=None)
        assert gates["G2"] is False
        assert gates["verdict"] == "CLOSED"

    def test_g2_fails_high_p(self):
        hr = self._hr(0.6, {"BTC/USDT": 0.6, "ETH/USDT": 0.6},
                      {"BTC/USDT": 40, "ETH/USDT": 40})
        gates = run_gate_checks(hr, self._temporal(), 80, dm_dict={"p_value": 0.5})
        assert gates["G2"] is False

    def test_g3_fails_single_asset(self):
        hr = self._hr(0.6, {"BTC/USDT": 0.6}, {"BTC/USDT": 40})
        gates = run_gate_checks(hr, self._temporal(), 40, dm_dict={"p_value": 0.01})
        assert gates["G3"] is False

    def test_g3_fails_one_asset_below_50(self):
        hr = self._hr(0.6, {"BTC/USDT": 0.6, "ETH/USDT": 0.45},
                      {"BTC/USDT": 40, "ETH/USDT": 40})
        gates = run_gate_checks(hr, self._temporal(), 80, dm_dict={"p_value": 0.01})
        assert gates["G3"] is False

    def test_g4_fails_unstable_window(self):
        hr = self._hr(0.6, {"BTC/USDT": 0.6, "ETH/USDT": 0.6},
                      {"BTC/USDT": 40, "ETH/USDT": 40})
        gates = run_gate_checks(hr, self._temporal(o=0.4), 80, dm_dict={"p_value": 0.01})
        assert gates["G4"] is False

    def test_g5_fails_on_degradation(self):
        hr = self._hr(0.6, {"BTC/USDT": 0.6, "ETH/USDT": 0.6},
                      {"BTC/USDT": 40, "ETH/USDT": 40})
        gates = run_gate_checks(hr, self._temporal(o=0.7, r=0.5), 80,
                                dm_dict={"p_value": 0.01})
        # recent (0.5) is 20pp below older (0.7) -> degradation -> G5 False
        assert gates["G5"] is False

    def test_g6_fails_few_events(self):
        hr = self._hr(0.6, {"BTC/USDT": 0.6, "ETH/USDT": 0.6},
                      {"BTC/USDT": 40, "ETH/USDT": 20})
        gates = run_gate_checks(hr, self._temporal(), 60, dm_dict={"p_value": 0.01})
        assert gates["G6"] is False

    def test_g6_fallback_to_total(self):
        # No per-asset counts -> fall back to total n_events.
        hr = {"hit_rate": 0.6, "by_asset": {"BTC/USDT": 0.6, "ETH/USDT": 0.6},
              "by_asset_n": {}}
        gates = run_gate_checks(hr, self._temporal(), 50, dm_dict={"p_value": 0.01})
        assert gates["G6"] is True


class TestMerge:

    def test_merge_columns_and_hit(self):
        events = [{"ts": _ts(0), "asset": "BTC/USDT", "bdir": 1, "fdir": 1, "regime": "high"},
                  {"ts": _ts(1), "asset": "BTC/USDT", "bdir": 1, "fdir": -1, "regime": "high"}]
        d, r = _build(events)
        m = merge_direction_returns(d, r, horizon=1)
        assert list(m["hit"]) == [1, 0]
        assert "breakout_direction" in m.columns
        assert "forward_direction" in m.columns
