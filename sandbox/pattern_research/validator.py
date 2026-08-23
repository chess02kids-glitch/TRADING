"""Validation layer for the pattern-research sandbox.

Three responsibilities, all identical in spirit and in arithmetic to Phase 9A
(``phase9a/dm_test.py`` + ``phase9a/continuation_tester.py``) but re-implemented
here so the sandbox stays standalone (no imports from the production tree, no
DB, no secrets):

1. :func:`run_dm_test` — one-sided Diebold-Mariano test of the signal against a
   50/50 coin flip, Newey-West (Bartlett) HAC variance with **3 lags**.
2. :func:`run_gate_checks` — the pre-registered **G1–G6** gates and the
   all-or-nothing verdict.
3. :func:`run_walk_forward` — chronological older / middle / recent hit rates.

A parity test (``tests/test_validator.py``) asserts this DM implementation
returns bit-identical numbers to ``phase9a.dm_test.compute_dm_statistic``.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# --- Pre-registered constants (do not tune) ---------------------------------
SIGNIFICANCE_LEVEL = 0.05
HAC_LAGS = 3
PRIMARY_HORIZON = 1
BOTH_ASSETS = ("BTC/USDT", "ETH/USDT")
MIN_EVENTS_PER_ASSET = 30      # G6 (same as Phase 9A)
MIN_OCCURRENCES = 50           # sandbox rule 6: below this a pattern is not worth testing
G1_HIT_RATE = 0.55
G3_HIT_RATE = 0.50
G4_HIT_RATE = 0.50
G5_MAX_DEGRADE = 0.10

REQUIRED_COLUMNS = ("signal", "forward_return", "correct")


# ---------------------------------------------------------------------------
# 1. Diebold-Mariano
# ---------------------------------------------------------------------------
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


def run_dm_test(actual_directions: Sequence, predicted_directions: Sequence) -> Dict[str, object]:
    """One-sided DM test: is the pattern better than a coin flip?

    Loss of the signal is ``1{actual != predicted}``; the random benchmark loss
    is ``0.5``; ``d_t = 0.5 - loss_signal`` and
    ``DM = mean(d) / (HAC_std(d) / sqrt(n))`` with 3 Bartlett lags. The p-value
    is the right tail ``1 - Φ(DM)`` — small p means a real directional edge.

    Returns ``{"dm_stat", "p_value", "hit_rate", "n_obs", "conclusion"}``.
    """
    actual = np.sign(np.asarray(actual_directions, dtype=float)).astype(int)
    predicted = np.sign(np.asarray(predicted_directions, dtype=float)).astype(int)
    if actual.size != predicted.size:
        raise ValueError("actual and predicted must have the same length")
    n = int(actual.size)

    if n == 0:
        return {"dm_stat": 0.0, "p_value": 1.0, "hit_rate": 0.0,
                "n_obs": 0, "conclusion": "NO DATA"}

    loss_signal = (predicted != actual).astype(float)
    d = 0.5 - loss_signal
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
        p_value = 1.0 - _normal_cdf(dm_stat)

    conclusion = (
        f"SIGNAL SIGNIFICANTLY BETTER THAN RANDOM (p={p_value:.3g} < {SIGNIFICANCE_LEVEL})"
        if p_value < SIGNIFICANCE_LEVEL
        else f"NO SIGNIFICANT EDGE OVER RANDOM (p={p_value:.3g})"
    )
    return {"dm_stat": dm_stat, "p_value": p_value, "hit_rate": hit_rate,
            "n_obs": n, "conclusion": conclusion}


# ---------------------------------------------------------------------------
# helpers on the results frame
# ---------------------------------------------------------------------------
def _validate(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df is None:
        raise ValueError("results_df is None")
    missing = [c for c in REQUIRED_COLUMNS if c not in results_df.columns]
    if missing:
        raise ValueError(f"results_df missing required columns: {missing}")
    df = results_df.copy()
    df = df.dropna(subset=["signal", "forward_return"])
    df = df[df["signal"] != 0]
    if "asset" not in df.columns:
        df["asset"] = "UNKNOWN"
    if "timestamp" not in df.columns:
        df = df.assign(timestamp=df.index)
    # A 'timestamp' column plus a 'timestamp' index name is ambiguous to
    # pandas sorting — drop the index, the column is the source of truth.
    return df.reset_index(drop=True).sort_values("timestamp")


def _hit_rate(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    return float(df["correct"].astype(float).mean())


def compute_hit_rate(results_df: pd.DataFrame) -> Dict[str, object]:
    """Overall / per-asset / per-direction hit rates for a results frame."""
    df = _validate(results_df)
    if df.empty:
        return {"overall_hit_rate": 0.0, "n_events": 0, "n_correct": 0,
                "by_asset": {}, "by_direction": {}}
    by_asset = {str(k): float(v) for k, v in
                df.groupby("asset")["correct"].mean().items()}
    by_direction = {int(k): float(v) for k, v in
                    df.groupby(df["signal"].astype(int))["correct"].mean().items()}
    return {
        "overall_hit_rate": _hit_rate(df),
        "n_events": int(len(df)),
        "n_correct": int(df["correct"].astype(int).sum()),
        "by_asset": by_asset,
        "by_direction": by_direction,
    }


def split_temporal_windows(results_df: pd.DataFrame, n_splits: int = 3) -> List[pd.DataFrame]:
    """Split events into ``n_splits`` near-equal chronological groups."""
    df = _validate(results_df).reset_index(drop=True)
    n = int(n_splits)
    if df.empty:
        return [df.copy() for _ in range(n)]
    # positional split (np.array_split on a DataFrame is not stable across
    # pandas versions, so split the positions and slice with .iloc)
    return [df.iloc[idx].copy() for idx in np.array_split(np.arange(len(df)), n)]


def compute_temporal_stability(results_df: pd.DataFrame) -> Dict[str, object]:
    """Older / middle / recent hit rates plus stability + degradation flags.

    ``is_stable`` = all three thirds > 0.50; ``degrading`` = recent is more than
    0.10 below older (both exactly as Phase 9A defines them).
    """
    older, middle, recent = split_temporal_windows(results_df, 3)
    o, m, r = _hit_rate(older), _hit_rate(middle), _hit_rate(recent)
    return {
        "older": o, "middle": m, "recent": r,
        "n_older": int(len(older)), "n_middle": int(len(middle)), "n_recent": int(len(recent)),
        "is_stable": all(x > G4_HIT_RATE for x in (o, m, r)),
        "degrading": bool(r < o - G5_MAX_DEGRADE),
    }


# ---------------------------------------------------------------------------
# 2. G1-G6 gates
# ---------------------------------------------------------------------------
def run_gate_checks(results_df: pd.DataFrame) -> Dict[str, object]:
    """Run the pre-registered G1–G6 gates on a results frame.

    Expected columns: ``signal``, ``forward_return``, ``correct`` (as produced
    by ``patterns.compute_forward_return``), optionally ``asset`` and
    ``timestamp``.

    Gate definitions (identical to Phase 9A — do not add criteria):

    * **G1** hit rate > 0.55 overall *and* on both assets
    * **G2** one-sided DM p < 0.05
    * **G3** hit rate > 0.50 on **both** BTC and ETH (cross-asset consistency)
    * **G4** every chronological third > 0.50
    * **G5** recent third not more than 0.10 below the older third
    * **G6** at least 30 events per asset

    Any failure ⇒ ``verdict = "CLOSED"``. A single-asset run cannot satisfy the
    cross-asset gates (G1/G3) by construction; that is reported in ``notes``
    rather than silently relaxed.
    """
    df = _validate(results_df)
    hit = compute_hit_rate(df)
    temporal = compute_temporal_stability(df)
    counts = ({str(k): int(v) for k, v in df.groupby("asset").size().items()}
              if not df.empty else {})

    if df.empty:
        dm = {"dm_stat": 0.0, "p_value": 1.0, "hit_rate": 0.0,
              "n_obs": 0, "conclusion": "NO DATA"}
    else:
        dm = run_dm_test(np.sign(df["forward_return"].to_numpy(dtype=float)),
                         np.sign(df["signal"].to_numpy(dtype=float)))

    by_asset = hit["by_asset"]
    overall = float(hit["overall_hit_rate"])

    g1 = overall > G1_HIT_RATE and all(
        by_asset.get(a, 0.0) > G1_HIT_RATE for a in BOTH_ASSETS)
    try:
        g2 = float(dm.get("p_value", 1.0)) < SIGNIFICANCE_LEVEL
    except (TypeError, ValueError):
        g2 = False
    g3 = all(a in by_asset and by_asset[a] > G3_HIT_RATE for a in BOTH_ASSETS)
    g4 = bool(temporal["is_stable"])
    g5 = not bool(temporal["degrading"])
    g6 = all(counts.get(a, 0) >= MIN_EVENTS_PER_ASSET for a in BOTH_ASSETS)

    notes: List[str] = []
    present = [a for a in BOTH_ASSETS if a in by_asset]
    if len(present) < 2:
        notes.append(
            "Single-asset run: cross-asset gates G1/G3/G6 require both BTC/USDT "
            "and ETH/USDT and therefore cannot pass. Run --asset both for a "
            "verdict.")
    if int(hit["n_events"]) < MIN_OCCURRENCES:
        notes.append(
            f"Only {hit['n_events']} occurrences (< {MIN_OCCURRENCES} minimum): "
            "sample too small to be worth testing — treat results as indicative "
            "only.")

    all_pass = all((g1, g2, g3, g4, g5, g6))
    return {
        "G1": bool(g1), "G2": bool(g2), "G3": bool(g3),
        "G4": bool(g4), "G5": bool(g5), "G6": bool(g6),
        "all_pass": bool(all_pass),
        "verdict": "SIGNAL FOUND" if all_pass else "CLOSED",
        "notes": notes,
        "details": {
            "hit_rate": hit,
            "dm": dm,
            "temporal": temporal,
            "per_asset_event_counts": counts,
            "enough_occurrences": int(hit["n_events"]) >= MIN_OCCURRENCES,
        },
    }


# ---------------------------------------------------------------------------
# 3. Walk-forward
# ---------------------------------------------------------------------------
def run_walk_forward(
    candles: pd.DataFrame,
    signal_func: Callable[[pd.DataFrame], pd.Series],
    n_splits: int = 3,
    horizon: int = PRIMARY_HORIZON,
) -> Dict[str, object]:
    """Hit rate of ``signal_func`` across chronological slices of ``candles``.

    The candles are cut into ``n_splits`` contiguous, non-overlapping blocks
    (oldest first). ``signal_func`` is re-run **inside each block**, so no
    information crosses a block boundary; each block is scored with
    ``compute_forward_return`` at ``horizon``.

    Returns ``{"older", "middle", "recent", "splits", "n_splits", "is_stable",
    "degrading"}`` (the three named keys are present for ``n_splits == 3``; for
    other values they map to first / middle-most / last block).
    """
    from .patterns.momentum import compute_forward_return

    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if candles is None or candles.empty:
        return {"older": 0.0, "middle": 0.0, "recent": 0.0, "splits": [],
                "n_splits": int(n_splits), "is_stable": False, "degrading": False}

    blocks = np.array_split(np.arange(len(candles)), int(n_splits))
    splits: List[Dict[str, object]] = []
    for i, idx in enumerate(blocks):
        block = candles.iloc[idx]
        if block.empty:
            splits.append({"split": i, "n_events": 0, "hit_rate": 0.0,
                           "start": None, "end": None})
            continue
        signal = signal_func(block)
        events = compute_forward_return(block, signal, horizon=horizon)
        splits.append({
            "split": i,
            "n_events": int(len(events)),
            "hit_rate": _hit_rate(events) if len(events) else 0.0,
            "mean_forward_return": float(events["forward_return"].mean()) if len(events) else 0.0,
            "start": str(block.index[0]),
            "end": str(block.index[-1]),
        })

    rates = [float(s["hit_rate"]) for s in splits]
    older, recent = rates[0], rates[-1]
    middle = rates[len(rates) // 2]
    return {
        "older": older, "middle": middle, "recent": recent,
        "splits": splits, "n_splits": int(n_splits),
        "is_stable": all(r > G4_HIT_RATE for r in rates),
        "degrading": bool(recent < older - G5_MAX_DEGRADE),
    }


def summarize(results_df: pd.DataFrame) -> Dict[str, object]:
    """Convenience bundle: hit rate + DM + temporal + gates in one dict."""
    gates = run_gate_checks(results_df)
    out: Dict[str, object] = dict(gates["details"])  # type: ignore[arg-type]
    out["gates"] = gates
    return out


__all__ = [
    "run_dm_test", "run_gate_checks", "run_walk_forward", "compute_hit_rate",
    "compute_temporal_stability", "split_temporal_windows", "summarize",
    "MIN_OCCURRENCES", "MIN_EVENTS_PER_ASSET", "BOTH_ASSETS",
]
