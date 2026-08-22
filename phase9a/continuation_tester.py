"""Hit-rate statistics, temporal stability and the G1–G6 gate checks (Step 2).

Consumes the two tables produced by
:mod:`phase9a.direction_calculator` and answers the Phase 9A question: *does
breakout-bar direction persist into the next bars?*

Pre-registered gate operationalization (fixed — do not add criteria):

* **G1** — Hit rate > 55% at ``t+1``. The *overall* hit rate exceeds 0.55 **and**
  every asset present exceeds 0.55.
* **G2** — DM test (one-sided) p < 0.05 vs random. Filled in by
  :mod:`phase9a.dm_test`; ``run_gate_checks`` receives it via ``dm_dict`` and
  returns ``False`` (placeholder) when no DM result is supplied.
* **G3** — Consistent in both BTC and ETH. Both ``BTC/USDT`` and ``ETH/USDT``
  are represented **and** each has a hit rate above 0.50. A single-asset run
  therefore fails G3 (you cannot claim consistency across both having tested one).
* **G4** — Stable windows. Hit rate of each chronological third (older /
  middle / recent) is above 50%.
* **G5** — No degradation. The recent third is not more than 10 percentage
  points worse than the older third.
* **G6** — Sample size. At least 30 breakout events per asset (falls back to
  the total event count when per-asset counts are unavailable).

All-or-nothing verdict: every gate must pass for ``"SIGNAL FOUND"``; any
failure ⇒ ``"CLOSED"``.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from phase9a.direction_calculator import _to_epoch_ms

logger = logging.getLogger(__name__)

HIT_RATE_GATE = 0.55          # G1 threshold (overall + per asset)
CONSISTENCY_GATE = 0.50       # G3 per-asset "signal present" threshold
TEMPORAL_GATE = 0.50          # G4 per-window threshold
DEGRADATION_PP = 0.10         # G5: recent worse than older by more than this
MIN_EVENTS_PER_ASSET = 30     # G6

BOTH_ASSETS = ("BTC/USDT", "ETH/USDT")


def merge_direction_returns(
    direction_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Inner-join breakout direction with forward direction at one horizon.

    Adds a boolean ``hit`` column (``forward_direction == breakout_direction``).
    A forward direction of ``0`` (zero return) matches neither ``+1`` nor
    ``-1`` and is therefore a miss — conservative.
    """
    if direction_df.empty or returns_df.empty:
        return pd.DataFrame(columns=[
            "timestamp", "asset", "breakout_direction", "forward_direction",
            "forward_return", "hit", "regime",
        ])
    sub = returns_df[returns_df["horizon"] == int(horizon)].copy()
    merged = direction_df.merge(
        sub[["timestamp", "asset", "horizon", "forward_return", "forward_direction"]],
        on=["timestamp", "asset"],
        how="inner",
    )
    if merged.empty:
        return merged.assign(hit=pd.Series(dtype=bool))
    merged["hit"] = (merged["forward_direction"] == merged["breakout_direction"]).astype(int)
    return merged


def compute_hit_rate(
    direction_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    horizon: int,
) -> Dict[str, object]:
    """Overall / per-asset / per-regime hit rate at ``horizon``.

    Returns a dict with ``hit_rate``, ``n_events``, ``n_correct``,
    ``by_asset`` (asset → rate), ``by_asset_n`` (asset → event count) and
    ``by_regime`` (regime → rate). With zero events every rate is ``0.0``.
    """
    merged = merge_direction_returns(direction_df, returns_df, horizon)
    if merged.empty:
        return {
            "hit_rate": 0.0,
            "n_events": 0,
            "n_correct": 0,
            "by_asset": {},
            "by_asset_n": {},
            "by_regime": {},
        }
    hits = merged["hit"]
    n = int(len(merged))
    n_correct = int(hits.sum())
    by_asset = {str(k): float(v) for k, v in merged.groupby("asset")["hit"].mean().items()}
    by_asset_n = {str(k): int(v) for k, v in merged.groupby("asset").size().items()}
    by_regime: Dict[str, float] = {}
    if "regime" in merged.columns:
        reg = merged.dropna(subset=["regime"])
        reg = reg[reg["regime"].notna()]
        if not reg.empty:
            by_regime = {str(k): float(v)
                         for k, v in reg.groupby("regime")["hit"].mean().items()}
    return {
        "hit_rate": float(n_correct / n) if n else 0.0,
        "n_events": n,
        "n_correct": n_correct,
        "by_asset": by_asset,
        "by_asset_n": by_asset_n,
        "by_regime": by_regime,
    }


def _third_hit_rates(
    direction_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    horizon: int,
) -> List[float]:
    """Hit rates of the older / middle / recent chronological thirds.

    Events are ordered by breakout timestamp (epoch-ms) and split into three
    nearly-equal groups with ``numpy.array_split`` (earlier groups absorb the
    remainder). An empty third yields ``0.0``.
    """
    merged = merge_direction_returns(direction_df, returns_df, horizon)
    if merged.empty:
        return [0.0, 0.0, 0.0]
    merged = merged.copy()
    merged["_e"] = merged["timestamp"].map(_to_epoch_ms)
    merged = merged.sort_values("_e").reset_index(drop=True)
    parts = np.array_split(np.arange(len(merged)), 3)
    rates: List[float] = []
    for part in parts:
        if part.size == 0:
            rates.append(0.0)
        else:
            rates.append(float(merged.loc[part, "hit"].mean()))
    return rates  # [older, middle, recent]


def compute_temporal_stability(
    direction_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    horizon: int = 1,
) -> Dict[str, object]:
    """Hit rate across the older / middle / recent thirds of the data.

    ``is_stable`` is ``True`` only when all three thirds exceed 50% (gate G4).
    """
    older, middle, recent = _third_hit_rates(direction_df, returns_df, horizon)
    return {
        "older_hit_rate": older,
        "middle_hit_rate": middle,
        "recent_hit_rate": recent,
        "is_stable": all(r > TEMPORAL_GATE for r in (older, middle, recent)),
    }


def compute_degradation_flag(
    direction_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    horizon: int = 1,
) -> bool:
    """``True`` when the recent third is more than 10pp worse than the older third."""
    older, _middle, recent = _third_hit_rates(direction_df, returns_df, horizon)
    if older == 0.0 and recent == 0.0:
        return False
    return (older - recent) > DEGRADATION_PP


def run_gate_checks(
    hit_rate_dict: Dict[str, object],
    temporal_dict: Dict[str, object],
    n_events: int,
    dm_dict: Optional[Dict[str, object]] = None,
    horizon: int = 1,
) -> Dict[str, object]:
    """Evaluate the six pre-registered gates and produce the verdict.

    Args:
        hit_rate_dict: output of :func:`compute_hit_rate`.
        temporal_dict: output of :func:`compute_temporal_stability`.
        n_events: total breakout events (used for G6 fallback / reporting).
        dm_dict: output of :func:`phase9a.dm_test.compute_dm_statistic`
            (``None`` ⇒ G2 fails as a placeholder).
        horizon: analysed horizon (gates are canonically defined at ``t+1``).

    Returns a dict ``{G1..G6: bool, all_pass: bool, verdict: str}``.
    """
    by_asset: Dict[str, float] = dict(hit_rate_dict.get("by_asset") or {})
    by_asset_n: Dict[str, int] = dict(hit_rate_dict.get("by_asset_n") or {})
    overall = float(hit_rate_dict.get("hit_rate") or 0.0)

    # G1: > 55% overall AND every asset > 55%.
    g1_asset_ok = all(v > HIT_RATE_GATE for v in by_asset.values()) if by_asset else True
    g1 = (overall > HIT_RATE_GATE) and g1_asset_ok

    # G2: DM one-sided p < 0.05 vs random (placeholder when no DM result).
    if dm_dict is None:
        g2 = False
    else:
        try:
            g2 = float(dm_dict.get("p_value", 1.0)) < 0.05
        except (TypeError, ValueError):
            g2 = False

    # G3: both BTC/USDT and ETH/USDT present and each > 50%.
    g3 = all(a in by_asset and by_asset[a] > CONSISTENCY_GATE for a in BOTH_ASSETS)

    # G4: every chronological third > 50%.
    older = float(temporal_dict.get("older_hit_rate") or 0.0)
    middle = float(temporal_dict.get("middle_hit_rate") or 0.0)
    recent = float(temporal_dict.get("recent_hit_rate") or 0.0)
    g4 = all(r > TEMPORAL_GATE for r in (older, middle, recent))

    # G5: recent not worse than older by more than 10pp.
    g5 = not ((older - recent) > DEGRADATION_PP)

    # G6: >= 30 events per asset (fallback to total count).
    if by_asset_n:
        g6 = all(v >= MIN_EVENTS_PER_ASSET for v in by_asset_n.values())
    else:
        g6 = int(n_events) >= MIN_EVENTS_PER_ASSET

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
    }
