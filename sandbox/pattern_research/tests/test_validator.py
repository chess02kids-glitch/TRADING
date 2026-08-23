"""Validator tests: DM parity with Phase 9A, gate logic, walk-forward."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sandbox.pattern_research import validator
from sandbox.pattern_research.patterns.momentum import detect_momentum_combined
from .conftest import make_candles


# --- DM test ----------------------------------------------------------------
def test_dm_perfect_signal_is_significant():
    actual = np.array([1, -1] * 40)
    res = validator.run_dm_test(actual, actual)
    assert res["hit_rate"] == 1.0
    assert res["p_value"] < 0.05
    assert res["dm_stat"] > 0


def test_dm_always_wrong_signal_is_not_significant():
    actual = np.array([1, -1] * 40)
    res = validator.run_dm_test(actual, -actual)
    assert res["hit_rate"] == 0.0
    assert res["p_value"] > 0.95


def test_dm_coin_flip_has_no_edge():
    rng = np.random.default_rng(11)
    actual = rng.choice([-1, 1], size=4000)
    predicted = rng.choice([-1, 1], size=4000)
    res = validator.run_dm_test(actual, predicted)
    assert 0.45 < res["hit_rate"] < 0.55
    assert res["p_value"] > 0.05


def test_dm_empty_and_length_mismatch():
    assert validator.run_dm_test([], [])["conclusion"] == "NO DATA"
    with pytest.raises(ValueError):
        validator.run_dm_test([1, -1], [1])


def test_dm_matches_phase9a_implementation_exactly():
    """Rule 3: same DM test as Phase 9A — assert numeric parity."""
    phase9a = pytest.importorskip("phase9a.dm_test")
    rng = np.random.default_rng(5)
    actual = rng.choice([-1, 1], size=500)
    predicted = np.where(rng.random(500) < 0.58, actual, -actual)
    mine = validator.run_dm_test(actual, predicted)
    theirs = phase9a.compute_dm_statistic(actual, predicted)
    assert mine["dm_stat"] == pytest.approx(theirs["dm_stat"], rel=1e-12, abs=1e-12)
    assert mine["p_value"] == pytest.approx(theirs["p_value"], rel=1e-12, abs=1e-12)
    assert mine["hit_rate"] == pytest.approx(theirs["hit_rate"])


# --- gates ------------------------------------------------------------------
def _results_frame(n_per_asset=100, hit_rate=0.60, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    ts = pd.Timestamp("2024-01-01T00:00:00Z")
    for asset in validator.BOTH_ASSETS:
        for i in range(n_per_asset):
            signal = 1 if i % 2 == 0 else -1
            correct = rng.random() < hit_rate
            fr = 0.01 * signal * (1 if correct else -1)
            rows.append({"timestamp": ts + pd.Timedelta(hours=i), "asset": asset,
                         "signal": signal, "forward_return": fr,
                         "correct": int(correct)})
    return pd.DataFrame(rows)


def test_gates_pass_on_a_strong_synthetic_signal():
    gates = validator.run_gate_checks(_results_frame(n_per_asset=300, hit_rate=0.75))
    assert all(gates[g] for g in ("G1", "G2", "G3", "G4", "G5", "G6"))
    assert gates["verdict"] == "SIGNAL FOUND"


def test_gates_fail_on_a_coin_flip():
    gates = validator.run_gate_checks(_results_frame(n_per_asset=300, hit_rate=0.50))
    assert gates["verdict"] == "CLOSED"
    assert not gates["G1"] and not gates["G2"]


def test_g6_requires_thirty_events_per_asset():
    gates = validator.run_gate_checks(_results_frame(n_per_asset=20, hit_rate=0.9))
    assert not gates["G6"]
    assert gates["verdict"] == "CLOSED"


def test_single_asset_run_cannot_pass_cross_asset_gates():
    df = _results_frame(n_per_asset=300, hit_rate=0.9)
    df = df[df["asset"] == "BTC/USDT"]
    gates = validator.run_gate_checks(df)
    assert not gates["G1"] and not gates["G3"] and not gates["G6"]
    assert any("Single-asset" in n for n in gates["notes"])


def test_min_occurrences_note_below_fifty():
    gates = validator.run_gate_checks(_results_frame(n_per_asset=10, hit_rate=0.9))
    assert not gates["details"]["enough_occurrences"]
    assert any("occurrences" in n for n in gates["notes"])


def test_gates_reject_missing_columns():
    with pytest.raises(ValueError):
        validator.run_gate_checks(pd.DataFrame({"signal": [1]}))


def test_zero_signal_rows_are_excluded():
    df = _results_frame(n_per_asset=60, hit_rate=0.6)
    flat = df.copy()
    flat["signal"] = 0
    combined = pd.concat([df, flat])
    assert (validator.compute_hit_rate(combined)["n_events"]
            == validator.compute_hit_rate(df)["n_events"])


def test_temporal_stability_detects_degradation():
    n = 300
    ts = pd.date_range("2024-01-01T00:00:00Z", periods=n, freq="1h")
    correct = [1] * (n // 3) + [1] * (n // 3) + [0] * (n - 2 * (n // 3))
    df = pd.DataFrame({"timestamp": ts, "asset": "BTC/USDT", "signal": 1,
                       "forward_return": [0.01 if c else -0.01 for c in correct],
                       "correct": correct})
    temporal = validator.compute_temporal_stability(df)
    assert temporal["older"] == 1.0 and temporal["recent"] == 0.0
    assert temporal["degrading"] and not temporal["is_stable"]


# --- walk-forward -----------------------------------------------------------
def test_walk_forward_shape(synthetic_candles):
    wf = validator.run_walk_forward(synthetic_candles, detect_momentum_combined, n_splits=3)
    assert set(["older", "middle", "recent", "splits"]).issubset(wf)
    assert len(wf["splits"]) == 3
    assert sum(s["n_events"] for s in wf["splits"]) > 0
    for s in wf["splits"]:
        assert 0.0 <= s["hit_rate"] <= 1.0


def test_walk_forward_supports_other_split_counts(synthetic_candles):
    wf = validator.run_walk_forward(synthetic_candles, detect_momentum_combined, n_splits=5)
    assert len(wf["splits"]) == 5
    with pytest.raises(ValueError):
        validator.run_walk_forward(synthetic_candles, detect_momentum_combined, n_splits=1)


def test_walk_forward_on_empty_candles():
    from sandbox.pattern_research.data_loader import empty_candles
    wf = validator.run_walk_forward(empty_candles(), detect_momentum_combined)
    assert wf["splits"] == [] and wf["is_stable"] is False


def test_walk_forward_detects_a_planted_edge():
    """A signal that always predicts the next up-bar should score ~100%."""
    n = 900
    idx = pd.date_range("2024-01-01T00:00:00Z", periods=n, freq="1h", tz="UTC",
                        name="timestamp")
    close = 100.0 * (1.001 ** np.arange(n))       # monotonically rising
    candles = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999,
                            "close": close, "volume": 1.0}, index=idx)
    always_long = lambda c: pd.Series(1, index=c.index)
    wf = validator.run_walk_forward(candles, always_long, n_splits=3)
    assert wf["older"] == wf["middle"] == wf["recent"] == 1.0
    assert wf["is_stable"] and not wf["degrading"]
