"""Diebold-Mariano test: is the breakout-direction signal better than a coin flip?

One-sided test of forecast accuracy. Setup (per the Phase 9A spec):

* Forecast = the breakout-bar direction (``predicted``).
* Realised direction = ``actual``.
* Signal loss ``loss_signal = 1{actual != predicted}``.
* Random baseline loss = ``0.5`` (the expected loss of a 50/50 coin flip).
* Loss differential ``d_t = loss_random - loss_signal``  (``> 0`` ⇒ signal
  beats random).
* DM statistic ``= mean(d_t) / (HAC_std(d_t) / sqrt(n))`` with a Newey-West
  HAC estimator using a fixed **3 lags** (Bartlett kernel).
* p-value is **one-sided** (signal better than random), i.e. the right tail:
  ``p = 1 - Φ(DM)``. A small p (< 0.05) ⇒ the directional edge is
  statistically significant.

``random_baseline_benchmark`` is provided as a deterministic 50/50 ±1 array
utility (used for comparison/benchmarking; the DM test itself uses the 0.5
expected-loss baseline above).
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

SIGNIFICANCE_LEVEL = 0.05
HAC_LAGS = 3


def random_baseline_benchmark(n: int, seed: int = 42) -> np.ndarray:
    """Deterministic array of ``n`` values with an exact 50/50 ±1 split.

    Returns an empty ``int8`` array for ``n <= 0``.
    """
    n = int(n)
    if n <= 0:
        return np.empty(0, dtype=np.int8)
    arr = np.array([-1] * (n // 2) + [1] * (n - n // 2), dtype=np.int8)
    rng = np.random.default_rng(seed)
    rng.shuffle(arr)
    return arr


def _normal_cdf(x: float) -> float:
    try:
        from scipy.stats import norm  # type: ignore
        return float(norm.cdf(x))
    except Exception:  # pragma: no cover - scipy is in requirements
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _hac_std(d: np.ndarray, lags: int = HAC_LAGS) -> float:
    """Newey-West (Bartlett) long-run standard deviation of series ``d``."""
    d = np.asarray(d, dtype=float)
    n = d.size
    if n < 2:
        return 0.0
    e = d - d.mean()
    lag = max(1, int(lags))
    gamma0 = float(np.dot(e, e) / n)
    omega = gamma0
    for j in range(1, lag + 1):
        if j >= n:
            break
        weight = 1.0 - j / (lag + 1.0)
        gamma_j = float(np.dot(e[:-j], e[j:]) / n)
        omega += 2.0 * weight * gamma_j
    return float(math.sqrt(omega)) if omega > 0.0 else 0.0


def compute_dm_statistic(actual_directions, predicted_directions) -> Dict[str, object]:
    """One-sided DM test of the signal vs a 50/50 random baseline.

    Args:
        actual_directions: realised directions (``+1``/``-1``).
        predicted_directions: signal directions (``+1``/``-1``); same length.

    Returns ``{"dm_stat", "p_value", "n_obs", "hit_rate", "conclusion"}``.
    A positive DM statistic with a small one-sided p-value means the signal is
    a statistically significant improvement over a coin flip.
    """
    actual = np.sign(np.asarray(actual_directions, dtype=float)).astype(int)
    predicted = np.sign(np.asarray(predicted_directions, dtype=float)).astype(int)
    if actual.size != predicted.size:
        raise ValueError("actual and predicted must have the same length")
    n = int(actual.size)

    if n == 0:
        return {"dm_stat": 0.0, "p_value": 1.0, "n_obs": 0,
                "hit_rate": 0.0, "conclusion": "NO DATA"}

    loss_signal = (predicted != actual).astype(float)
    loss_random = 0.5
    d = loss_random - loss_signal  # > 0 when the signal is better
    d_bar = float(d.mean())
    hit_rate = float((predicted == actual).mean())

    hac_std = _hac_std(d)
    if hac_std == 0.0 or not math.isfinite(hac_std):
        if d_bar > 0.0:
            dm_stat, p_value = float("inf"), 0.0
        elif d_bar < 0.0:
            dm_stat, p_value = float("-inf"), 1.0
        else:
            dm_stat, p_value = 0.0, 0.5
    else:
        dm_stat = d_bar / (hac_std / math.sqrt(n))
        p_value = 1.0 - _normal_cdf(dm_stat)  # one-sided right tail

    if p_value < SIGNIFICANCE_LEVEL:
        conclusion = (
            f"SIGNAL SIGNIFICANTLY BETTER THAN RANDOM (p={p_value:.3g} < "
            f"{SIGNIFICANCE_LEVEL})")
    else:
        conclusion = f"NO SIGNIFICANT EDGE OVER RANDOM (p={p_value:.3g})"

    return {
        "dm_stat": dm_stat,
        "p_value": p_value,
        "n_obs": n,
        "hit_rate": hit_rate,
        "conclusion": conclusion,
    }
