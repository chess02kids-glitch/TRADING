"""Diebold-Mariano test: directional signal vs a 50/50 random baseline.

Phase 9A needs to know whether the breakout-bar direction is *significantly*
better at forecasting the next bar's direction than a coin flip. The
Diebold-Mariano (DM) test compares the loss series of two forecasts using a
HAC (heteroskedasticity- and autocorrelation-consistent) standard error, which
is the correct tool when forecast errors are serially correlated (as hourly
market returns are).

Setup used here:

* Forecast A — the breakout-bar direction (``+1`` / ``-1``).
* Forecast B — a deterministic 50/50 random baseline
  (:func:`random_baseline_benchmark`, fixed ``seed=42`` so the test is
  reproducible).
* Loss — asymmetric / 0-1: ``1`` when the forecast's sign disagrees with the
  realised direction, else ``0``.
* Loss differential ``d_i = loss_A_i - loss_B_i``. The signal is *better* than
  random when ``d_i`` is negative on average (lower loss).
* DM statistic ``= mean(d) / SE_HAC(mean(d))``, with the Newey-West (Bartlett)
  estimator for the long-run variance.
* One-sided test of "signal beats random": ``p = P(Z <= DM)`` (left tail), so
  a strongly negative DM statistic yields a small p-value. This matches the
  pre-registered gate G2 ("DM test p < 0.05 vs random baseline", one-sided).

Edge cases: when the HAC variance is zero (e.g. perfect or perfectly useless
signal) the statistic is sent to ``-inf`` / ``+inf`` and the p-value to ``0``
/ ``1`` accordingly — never a divide-by-zero. ``scipy.stats.norm`` is used for
the CDF when available, with an ``erf``-based fallback so the module works
without scipy installed (scipy is listed in requirements but not hard-required
at import time).
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

SIGNIFICANCE_LEVEL = 0.05  # G2 gate threshold


def random_baseline_benchmark(n: int, seed: int = 42) -> np.ndarray:
    """Deterministic array of ``n`` random ``+1`` / ``-1`` directions.

    A 50/50 coin-flip baseline. Fixed ``seed`` so the DM test is reproducible
    run-to-run (the same baseline is used for every comparison). Returns an
    empty ``int8`` array when ``n <= 0``.
    """
    n = int(n)
    if n <= 0:
        return np.empty(0, dtype=np.int8)
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=n)


def _normal_cdf(x: float) -> float:
    """Standard-normal CDF; scipy when present, ``erf`` fallback otherwise."""
    try:
        from scipy.stats import norm  # type: ignore
        return float(norm.cdf(x))
    except Exception:  # pragma: no cover - scipy is in requirements
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _newey_west_se(d: np.ndarray) -> float:
    """HAC (Newey-West, Bartlett kernel) standard error of ``mean(d)``.

    Lag length ``L = floor(n ** 0.25)`` (a standard default). Returns ``0.0``
    for ``n < 2`` or when the long-run variance is non-positive (caller maps
    that to an infinite statistic).
    """
    d = np.asarray(d, dtype=float)
    n = d.size
    if n < 2:
        return 0.0
    e = d - d.mean()
    lag = max(1, int(math.floor(n ** 0.25)))
    gamma0 = float(np.dot(e, e) / n)
    omega = gamma0
    for j in range(1, lag + 1):
        weight = 1.0 - j / (lag + 1.0)
        gamma_j = float(np.dot(e[:-j], e[j:]) / n)
        omega += 2.0 * weight * gamma_j
    if omega <= 0.0:
        return 0.0
    return float(math.sqrt(omega / n))


def compute_dm_statistic(
    actual_directions,
    predicted_directions,
    baseline: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, object]:
    """One-sided Diebold-Mariano test of the signal vs a random baseline.

    Args:
        actual_directions: realised directions (``+1``/``-1``/``0``) for each
            event — the ``forward_direction`` column at the chosen horizon.
        predicted_directions: the signal's directions (``+1``/``-1``) — the
            ``breakout_direction`` column. Must be the same length as actual.
        baseline: optional explicit random baseline (``+1``/``-1``). When
            ``None`` a deterministic 50/50 baseline is generated via
            :func:`random_baseline_benchmark` with ``seed``.
        seed: seed for the generated baseline (ignored when ``baseline`` given).

    Returns:
        ``{"dm_stat": float, "p_value": float, "n_obs": int, "conclusion": str}``.

        * ``dm_stat`` — negative ⇒ signal better than random.
        * ``p_value`` — one-sided left-tail; ``< 0.05`` ⇒ significant edge.
        * ``conclusion`` — human-readable verdict.
    """
    actual = np.sign(np.asarray(actual_directions, dtype=float)).astype(int)
    predicted = np.sign(np.asarray(predicted_directions, dtype=float)).astype(int)
    n = int(actual.size)

    if n == 0:
        logger.warning("compute_dm_statistic: 0 observations")
        return {"dm_stat": 0.0, "p_value": 1.0, "n_obs": 0,
                "conclusion": "NO DATA"}
    if predicted.size != n:
        raise ValueError(
            f"actual ({n}) and predicted ({predicted.size}) length mismatch")

    if baseline is None:
        baseline_arr = random_baseline_benchmark(n, seed=seed)
    else:
        baseline_arr = np.sign(np.asarray(baseline, dtype=float)).astype(int)
    if baseline_arr.size != n:
        raise ValueError(
            f"baseline ({baseline_arr.size}) must match actual ({n})")

    loss_signal = (predicted != actual).astype(float)
    loss_baseline = (baseline_arr != actual).astype(float)
    d = loss_signal - loss_baseline  # <0 means the signal is better
    d_bar = float(d.mean())

    se = _newey_west_se(d)
    if se == 0.0 or not math.isfinite(se):
        # Degenerate: every differential identical (or constant zero variance).
        if d_bar < 0.0:
            dm_stat, p_value = float("-inf"), 0.0
        elif d_bar > 0.0:
            dm_stat, p_value = float("inf"), 1.0
        else:
            dm_stat, p_value = 0.0, 0.5
    else:
        dm_stat = d_bar / se
        p_value = _normal_cdf(dm_stat)

    if p_value < SIGNIFICANCE_LEVEL:
        conclusion = (
            f"SIGNAL SIGNIFICANTLY BETTER THAN RANDOM (p={p_value:.3g} < "
            f"{SIGNIFICANCE_LEVEL})"
        )
    else:
        conclusion = f"NO SIGNIFICANT EDGE OVER RANDOM (p={p_value:.3g})"

    return {
        "dm_stat": dm_stat,
        "p_value": p_value,
        "n_obs": n,
        "conclusion": conclusion,
    }
