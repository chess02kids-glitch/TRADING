"""Temporal stability and the G1–G6 gate checks for Phase 9A.

Consumes the same DataFrame contract as
:mod:`phase9a.direction_calculator` and decides whether the breakout-direction
signal clears the pre-registered gates. All gates are evaluated at the primary
horizon ``t+1``.

Gate operationalization (fixed — do not add criteria):

* **G1** — hit rate > 0.55 overall **and** on both assets (BTC, ETH).
* **G2** — one-sided DM test p < 0.05 vs a coin flip.
* **G3** — hit rate > 0.50 on **both** BTC and ETH (consistency).
* **G4** — temporal stability: every third (older/middle/recent) > 0.50.
* **G5** — not degrading: recent not more than 0.10 below older.
* **G6** — at least 30 events per asset.

All-or-nothing verdict: every gate passes ⇒ ``"SIGNAL FOUND"``; any fail ⇒
``"CLOSED"``.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from phase9a.direction_calculator import (
    _filter_horizon,
    _validate,
    compute_hit_rate,
    split_temporal_windows,
)
from phase9a.dm_test import compute_dm_statistic

PRIMARY_HORIZON = 1
BOTH_ASSETS = ("BTC/USDT", "ETH/USDT")


def _window_hit_rate(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    s = df.dropna(subset=["breakout_direction", "forward_direction"])
    if s.empty:
        return 0.0
    return float((s["forward_direction"] == s["breakout_direction"]).mean())


def compute_temporal_stability(df: pd.DataFrame, horizon: int = PRIMARY_HORIZON) -> Dict[str, object]:
    """Hit rate across older / middle / recent chronological thirds.

    Returns ``{"older", "middle", "recent", "is_stable", "degrading"}``.
    ``is_stable`` requires all three thirds > 0.50; ``degrading`` is True when
    the recent third is more than 0.10 below the older third.
    """
    _validate(df)
    sub = _filter_horizon(df, horizon)
    if sub.empty:
        return {"older": 0.0, "middle": 0.0, "recent": 0.0,
                "is_stable": False, "degrading": False}
    older, middle, recent = split_temporal_windows(sub)
    o = _window_hit_rate(older)
    m = _window_hit_rate(middle)
    r = _window_hit_rate(recent)
    return {
        "older": o,
        "middle": m,
        "recent": r,
        "is_stable": all(x > 0.50 for x in (o, m, r)),
        "degrading": (r < o - 0.10),
    }


def _per_asset_event_counts(df: pd.DataFrame, horizon: int) -> Dict[str, int]:
    sub = _filter_horizon(df, horizon).dropna(
        subset=["breakout_direction", "forward_direction"])
    if sub.empty:
        return {}
    return {str(k): int(v) for k, v in sub.groupby("asset").size().items()}


def run_all_gate_checks(df: pd.DataFrame, horizon: int = PRIMARY_HORIZON) -> Dict[str, object]:
    """Run G1–G6 at ``horizon`` (default primary t+1) and return verdict + details."""
    _validate(df)
    hit = compute_hit_rate(df, horizon)
    temporal = compute_temporal_stability(df, horizon)
    counts = _per_asset_event_counts(df, horizon)

    sub = _filter_horizon(df, horizon).dropna(
        subset=["breakout_direction", "forward_direction"])
    if sub.empty:
        dm = {"dm_stat": 0.0, "p_value": 1.0, "n_obs": 0,
              "hit_rate": 0.0, "conclusion": "NO DATA"}
    else:
        dm = compute_dm_statistic(
            sub["forward_direction"].to_numpy(),
            sub["breakout_direction"].to_numpy(),
        )

    by_asset = hit["by_asset"]
    overall = hit["overall_hit_rate"]

    g1 = (overall > 0.55
          and all(by_asset.get(a, 0.0) > 0.55 for a in BOTH_ASSETS))
    try:
        g2 = float(dm.get("p_value", 1.0)) < 0.05
    except (TypeError, ValueError):
        g2 = False
    g3 = all(a in by_asset and by_asset[a] > 0.50 for a in BOTH_ASSETS)
    g4 = bool(temporal["is_stable"])
    g5 = not bool(temporal["degrading"])
    g6 = all(counts.get(a, 0) >= 30 for a in BOTH_ASSETS)

    all_pass = all((g1, g2, g3, g4, g5, g6))
    return {
        "G1": bool(g1),
        "G2": bool(g2),
        "G3": bool(g3),
        "G4": bool(g4),
        "G5": bool(g5),
        "G6": bool(g6),
        "all_pass": bool(all_pass),
        "verdict": "SIGNAL FOUND" if all_pass else "CLOSED",
        "details": {
            "hit_rate": hit,
            "temporal": temporal,
            "dm": dm,
            "per_asset_event_counts": counts,
        },
    }
