"""Phase 4 robustness - honest paired statistical comparison.

Self-contained (numpy only) so the results are deterministic and reproducible
on the target machine:

* ``bootstrap_mean_ci``        - percentile bootstrap confidence interval on a
  mean (fixed seed).
* ``wilcoxon_signed_rank``     - Wilcoxon signed-rank test via the standard
  normal approximation with tie correction (zero differences dropped).
* ``mcnemar``                  - McNemar test for paired binary outcomes, with
  continuity-corrected chi-square for large discordant counts and the exact
  binomial for small discordant counts.

Every comparison is **paired**: it uses only observations that exist at the
exact same timestamps for both systems, so no metric is ever computed on
mismatched samples.

Statistical significance is NOT the same as trading profitability - the report
carries this note explicitly.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

EPS = 1e-12


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def bootstrap_mean_ci(values, n_boot: int = 10_000, alpha: float = 0.05,
                      seed: int = 0) -> Dict[str, Any]:
    """Percentile bootstrap CI on the mean of ``values`` (fixed seed)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return {'n': 0, 'mean': None, 'ci_low': None, 'ci_high': None,
                'n_boot': n_boot, 'alpha': alpha, 'note': 'no finite values'}
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return {'n': n, 'mean': float(arr.mean()), 'ci_low': lo, 'ci_high': hi,
            'n_boot': n_boot, 'alpha': alpha}


def wilcoxon_signed_rank(diffs, seed: int = 0) -> Dict[str, Any]:
    """Two-sided Wilcoxon signed-rank test on paired differences.

    Uses the normal approximation with tie correction; zero differences are
    dropped (the standard convention). For very small non-zero sample sizes
    (``n < 20``) the normal approximation is only approximate - the returned
    dict carries that caveat.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0.0]
    n = len(d)
    if n == 0:
        return {'statistic': None, 'z': None, 'p_value': None, 'n_nonzero': 0,
                'method': 'wilcoxon_signed_rank_normal_approx',
                'note': 'no non-zero differences'}
    absd = np.abs(d)
    order = np.argsort(absd, kind='mergesort')
    ranks = np.empty(n, dtype=float)
    tie_groups: List[int] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[order[j + 1]] == absd[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0  # 1-based average rank
        ranks[order[i:j + 1]] = avg
        g = j - i + 1
        if g > 1:
            tie_groups.append(g)
        i = j + 1

    w = float(ranks[d > 0].sum())
    mean = n * (n + 1) / 4.0
    tie_corr = sum(g ** 3 - g for g in tie_groups) / 48.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr
    if var <= 0:
        var = EPS
    z = (w - mean) / math.sqrt(var)
    p = 2 * _normal_sf(abs(z))
    note = 'normal approximation with tie correction; zero differences dropped'
    if n < 20:
        note += '; n<20 so the approximation is coarse'
    return {'statistic': w, 'z': float(z), 'p_value': float(p), 'n_nonzero': n,
            'method': 'wilcoxon_signed_rank_normal_approx', 'note': note}


def mcnemar(b: int, c: int) -> Dict[str, Any]:
    """McNemar test for paired binary outcomes.

    ``b`` = count where system A is correct and system B is incorrect;
    ``c`` = count where system A is incorrect and system B is correct.
    """
    b = int(b)
    c = int(c)
    n_disc = b + c
    if n_disc == 0:
        return {'b': b, 'c': c, 'statistic': None, 'p_value': None,
                'method': 'mcnemar', 'note': 'no discordant pairs'}
    if n_disc < 25:
        # exact two-sided binomial
        m = min(b, c)
        p = 0.0
        for k in range(m + 1):
            p += math.comb(n_disc, k) * (0.5 ** n_disc)
        p = min(1.0, 2.0 * p)
        return {'b': b, 'c': c, 'statistic': None, 'p_value': float(p),
                'method': 'mcnemar_exact_binomial',
                'note': 'exact binomial (discordant pairs < 25)'}
    stat = ((abs(b - c) - 1.0) ** 2) / n_disc
    p = _chi2_sf(stat, df=1)
    return {'b': b, 'c': c, 'statistic': float(stat), 'p_value': float(p),
            'method': 'mcnemar_chi2_continuity_corrected',
            'note': 'continuity-corrected chi-square (1 df)'}


def _chi2_sf(x: float, df: int = 1) -> float:
    # Survival function of the chi-square distribution with 1 df.
    return _normal_sf(math.sqrt(x)) * 2.0 if df == 1 else math.nan


def _align(rows_a, rows_b):
    """Align two row lists on ``prediction_timestamp`` (identical observations)."""
    if rows_a is None or rows_b is None:
        return [], []
    b_map = {r.prediction_timestamp: r for r in rows_b}
    a_out, b_out = [], []
    for r in rows_a:
        m = b_map.get(r.prediction_timestamp)
        if m is not None:
            a_out.append(r)
            b_out.append(m)
    return a_out, b_out


def paired_error_comparison(kronos_errors, baseline_errors,
                            n_boot: int = 10_000, alpha: float = 0.05,
                            seed: int = 0) -> Dict[str, Any]:
    """Paired comparison of absolute close errors (identical timestamps).

    ``diffs = kronos - baseline``; a negative mean diff means Kronos produced
    smaller errors (better).
    """
    k = np.asarray(kronos_errors, dtype=float)
    b = np.asarray(baseline_errors, dtype=float)
    mask = np.isfinite(k) & np.isfinite(b)
    k, b = k[mask], b[mask]
    n = len(k)
    diffs = k - b
    if n == 0:
        return {'sample_size': 0, 'mean_diff': None, 'median_diff': None,
                'std_diff': None, 'cohens_dz': None,
                'bootstrap_ci_95': None, 'wilcoxon_p_value': None,
                'wilcoxon_statistic': None, 'wilcoxon_n_nonzero': 0,
                'winner_by_mean': None,
                'note': 'no paired observations (negative diff = Kronos lower error)'}
    boot = bootstrap_mean_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed)
    w = wilcoxon_signed_rank(diffs, seed=seed)
    mean_d = float(diffs.mean())
    median_d = float(np.median(diffs))
    std_d = float(diffs.std(ddof=1)) if n > 1 else None
    cohens_dz = (mean_d / std_d) if (std_d is not None and std_d > EPS) else None
    winner = 'kronos' if mean_d < 0 else ('baseline' if mean_d > 0 else 'tie')
    return {
        'sample_size': n,
        'mean_diff': mean_d,
        'median_diff': median_d,
        'std_diff': std_d,
        'cohens_dz': cohens_dz,
        'bootstrap_ci_95': [boot['ci_low'], boot['ci_high']],
        'bootstrap_alpha': alpha,
        'bootstrap_n_boot': n_boot,
        'wilcoxon_p_value': w['p_value'],
        'wilcoxon_statistic': w['statistic'],
        'wilcoxon_n_nonzero': w['n_nonzero'],
        'wilcoxon_method': w['method'],
        'wilcoxon_note': w['note'],
        'winner_by_mean': winner,
        'note': 'paired absolute close error; negative diff = Kronos lower error',
    }


def paired_direction_comparison(kronos_correct, baseline_correct,
                                n_boot: int = 10_000, alpha: float = 0.05,
                                seed: int = 0) -> Dict[str, Any]:
    """Paired comparison of directional correctness (identical timestamps).

    Both input lists must be aligned on the same non-flat actual candles.
    """
    k = np.asarray(kronos_correct, dtype=bool)
    b = np.asarray(baseline_correct, dtype=bool)
    n = len(k)
    if n == 0:
        return {'sample_size': 0, 'kronos_accuracy': None, 'baseline_accuracy': None,
                'accuracy_delta': None, 'bootstrap_ci_95': None,
                'mcnemar_b': 0, 'mcnemar_c': 0, 'mcnemar_p_value': None,
                'mcnemar_method': None, 'winner_by_accuracy': None,
                'note': 'no paired non-flat observations'}

    acc_k = float(k.mean())
    acc_b = float(b.mean())
    delta = acc_k - acc_b

    # Bootstrap CI on the paired accuracy difference (resample observations).
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = k[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo = float(np.percentile(deltas, 100 * alpha / 2))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))

    nb = int((k & ~b).sum())  # Kronos correct, baseline incorrect
    nc = int((~k & b).sum())  # Kronos incorrect, baseline correct
    mcn = mcnemar(nb, nc)

    winner = 'kronos' if delta > 0 else ('baseline' if delta < 0 else 'tie')
    return {
        'sample_size': n,
        'kronos_accuracy': acc_k,
        'baseline_accuracy': acc_b,
        'accuracy_delta': delta,
        'bootstrap_ci_95': [lo, hi],
        'bootstrap_alpha': alpha,
        'bootstrap_n_boot': n_boot,
        'mcnemar_b': nb,
        'mcnemar_c': nc,
        'mcnemar_p_value': mcn['p_value'],
        'mcnemar_method': mcn['method'],
        'mcnemar_note': mcn['note'],
        'winner_by_accuracy': winner,
        'note': 'paired on non-flat actual candles only; positive delta = Kronos better',
    }


def build_statistical_comparison(kronos_rows, persistence_rows,
                                 previous_direction_rows,
                                 direction_threshold: float,
                                 n_boot: int = 10_000, alpha: float = 0.05,
                                 seed: int = 0) -> Dict[str, Any]:
    """Build the ``statistical_comparison`` report block.

    Aligns all systems on identical ``prediction_timestamp`` values, restricts
    the directional comparison to non-flat actual candles, and runs the paired
    tests.
    """
    from .evaluation import direction  # lazy import avoids a module cycle

    kp, pp = _align(kronos_rows, persistence_rows)
    kd, dd = _align(kronos_rows, previous_direction_rows)

    # Non-flat actual candles only, aligned identically for each pair.
    kp_nonflat = [r for r in kp if direction(r.actual_return, direction_threshold) != 0]
    kd_nonflat = [r for r in kd if direction(r.actual_return, direction_threshold) != 0]
    pp_nonflat = [r for r in pp if direction(r.actual_return, direction_threshold) != 0]
    dd_nonflat = [r for r in dd if direction(r.actual_return, direction_threshold) != 0]

    return {
        'paired_close_error': {
            'kronos_vs_persistence': paired_error_comparison(
                [r.absolute_close_error for r in kp],
                [r.absolute_close_error for r in pp],
                n_boot=n_boot, alpha=alpha, seed=seed),
            'kronos_vs_previous_direction': paired_error_comparison(
                [r.absolute_close_error for r in kd],
                [r.absolute_close_error for r in dd],
                n_boot=n_boot, alpha=alpha, seed=seed),
        },
        'directional': {
            'kronos_vs_persistence': paired_direction_comparison(
                [r.directional_correct for r in kp_nonflat],
                [r.directional_correct for r in pp_nonflat],
                n_boot=n_boot, alpha=alpha, seed=seed),
            'kronos_vs_previous_direction': paired_direction_comparison(
                [r.directional_correct for r in kd_nonflat],
                [r.directional_correct for r in dd_nonflat],
                n_boot=n_boot, alpha=alpha, seed=seed),
        },
        'notes': [
            'paired on identical prediction timestamps only',
            'bootstrap percentile CI with seed=%d, n_boot=%d, alpha=%.3f'
            % (seed, n_boot, alpha),
            'persistence predicts no direction (flat), so its directional '
            'accuracy is 0 by policy and the paired test vs persistence is '
            'one-sided by construction',
            'statistical significance is NOT trading profitability',
        ],
    }
