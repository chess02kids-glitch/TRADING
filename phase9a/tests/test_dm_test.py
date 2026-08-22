"""Unit tests for phase9a.dm_test (Diebold-Mariano vs random baseline)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from phase9a.dm_test import (
    compute_dm_statistic,
    random_baseline_benchmark,
    _newey_west_se,
)


class TestRandomBaseline:

    def test_shape_and_values(self):
        b = random_baseline_benchmark(50)
        assert b.shape == (50,)
        assert set(np.unique(b)).issubset({-1, 1})

    def test_deterministic_with_seed(self):
        a = random_baseline_benchmark(100, seed=42)
        b = random_baseline_benchmark(100, seed=42)
        c = random_baseline_benchmark(100, seed=7)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_empty(self):
        assert random_baseline_benchmark(0).size == 0


class TestNeweyWestSE:

    def test_constant_series_zero_se(self):
        d = np.zeros(30)
        assert _newey_west_se(d) == 0.0

    def test_positive_se(self):
        rng = np.random.default_rng(0)
        d = rng.normal(0, 1, size=200)
        se = _newey_west_se(d)
        assert se > 0.0
        assert math.isfinite(se)


class TestComputeDmStatistic:

    def test_zero_observations(self):
        r = compute_dm_statistic([], [])
        assert r["n_obs"] == 0
        assert r["p_value"] == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_dm_statistic([1, 1, 1], [1, 1])

    def test_perfect_signal_significant(self):
        # Signal always correct => loss_signal=0, beats random => negative DM => small p.
        rng = np.random.default_rng(1)
        actual = rng.choice([-1, 1], size=200)
        predicted = actual.copy()
        r = compute_dm_statistic(actual, predicted)
        assert r["n_obs"] == 200
        assert r["dm_stat"] < 0
        assert r["p_value"] < 0.05
        assert "SIGNIFICANTLY BETTER" in r["conclusion"]

    def test_anti_signal_not_significant(self):
        # Signal always wrong => worse than random => p large.
        rng = np.random.default_rng(2)
        actual = rng.choice([-1, 1], size=200)
        predicted = -actual
        r = compute_dm_statistic(actual, predicted)
        assert r["dm_stat"] > 0
        assert r["p_value"] > 0.5

    def test_pvalue_in_unit_interval(self):
        rng = np.random.default_rng(3)
        actual = rng.choice([-1, 1], size=60)
        predicted = rng.choice([-1, 1], size=60)
        r = compute_dm_statistic(actual, predicted)
        assert 0.0 <= r["p_value"] <= 1.0
        assert math.isfinite(r["dm_stat"]) or math.isinf(r["dm_stat"])

    def test_explicit_baseline(self):
        actual = np.array([1, 1, -1, -1, 1, -1] * 10)
        predicted = actual.copy()  # perfect
        baseline = np.array([-1, 1, 1, -1, -1, 1] * 10)
        r = compute_dm_statistic(actual, predicted, baseline=baseline)
        assert r["p_value"] < 0.05

    def test_deterministic_output(self):
        rng = np.random.default_rng(4)
        actual = rng.choice([-1, 1], size=80)
        predicted = rng.choice([-1, 1], size=80)
        r1 = compute_dm_statistic(actual, predicted)
        r2 = compute_dm_statistic(actual, predicted)
        assert r1["dm_stat"] == r2["dm_stat"]
        assert r1["p_value"] == r2["p_value"]
