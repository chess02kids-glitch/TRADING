"""Tests for phase9a.dm_test (Diebold-Mariano vs coin-flip baseline)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from phase9a.dm_test import compute_dm_statistic, random_baseline_benchmark


class TestRandomBaseline:

    def test_shape_and_values(self):
        b = random_baseline_benchmark(50)
        assert b.shape == (50,)
        assert set(np.unique(b)).issubset({-1, 1})

    def test_exact_half_split(self):
        b = random_baseline_benchmark(100)
        assert int((b == 1).sum()) == 50
        assert int((b == -1).sum()) == 50

    def test_deterministic(self):
        a = random_baseline_benchmark(80, seed=42)
        c = random_baseline_benchmark(80, seed=42)
        d = random_baseline_benchmark(80, seed=1)
        assert np.array_equal(a, c)
        assert not np.array_equal(a, d)

    def test_empty(self):
        assert random_baseline_benchmark(0).size == 0


class TestComputeDmStatistic:

    def test_perfect_signal_significant(self):
        rng = np.random.default_rng(1)
        actual = rng.choice([-1, 1], size=200)
        predicted = actual.copy()
        r = compute_dm_statistic(actual, predicted)
        assert r["n_obs"] == 200
        assert r["hit_rate"] == pytest.approx(1.0)
        assert r["dm_stat"] > 0            # positive => signal beats random
        assert r["p_value"] < 0.05
        assert "SIGNIFICANTLY BETTER" in r["conclusion"]

    def test_anti_signal_not_significant(self):
        rng = np.random.default_rng(2)
        actual = rng.choice([-1, 1], size=200)
        r = compute_dm_statistic(actual, -actual)
        assert r["dm_stat"] < 0
        assert r["p_value"] > 0.5

    def test_pvalue_in_unit_interval(self):
        rng = np.random.default_rng(3)
        actual = rng.choice([-1, 1], size=60)
        predicted = rng.choice([-1, 1], size=60)
        r = compute_dm_statistic(actual, predicted)
        assert 0.0 <= r["p_value"] <= 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_dm_statistic([1, 1, 1], [1, 1])

    def test_empty(self):
        r = compute_dm_statistic([], [])
        assert r["n_obs"] == 0
        assert r["p_value"] == 1.0
