"""Phase 9A hit-rate computation and temporal-window splitting.

This module is DB-free: it operates on a ``pd.DataFrame`` whose rows are
breakout events with their realised forward direction. The canonical source is
``forward_return_logger.get_phase9a_data()`` (exported to CSV), which yields:

    breakout_timestamp, asset, breakout_direction, horizon,
    target_timestamp, forward_return, forward_direction, breakout_close_price

A row is a **hit** when ``forward_direction == breakout_direction``. Functions
here never look ahead — they only summarise already-realised rows.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"breakout_direction", "forward_direction", "asset", "breakout_timestamp"}


def _validate(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")


def _filter_horizon(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return rows at ``horizon`` (no-op when the frame has no horizon column)."""
    if "horizon" in df.columns:
        return df[df["horizon"] == int(horizon)].copy()
    return df.copy()


def compute_hit_rate(df: pd.DataFrame, horizon: int = 1) -> Dict[str, object]:
    """Overall / per-asset / per-breakout-direction hit rate at ``horizon``.

    Returns ``{"overall_hit_rate", "n_events", "n_correct", "by_asset",
    "by_direction"}``. With zero usable events every rate is ``0.0``.
    """
    _validate(df)
    sub = _filter_horizon(df, horizon).dropna(
        subset=["breakout_direction", "forward_direction"])
    if sub.empty:
        return {"overall_hit_rate": 0.0, "n_events": 0, "n_correct": 0,
                "by_asset": {}, "by_direction": {}}
    sub = sub.copy()
    sub["_hit"] = (sub["forward_direction"] == sub["breakout_direction"]).astype(int)
    n = int(len(sub))
    n_correct = int(sub["_hit"].sum())
    by_asset = {str(k): float(v)
                for k, v in sub.groupby("asset")["_hit"].mean().items()}
    by_direction: Dict[int, float] = {}
    for k, v in sub.groupby("breakout_direction")["_hit"].mean().items():
        try:
            by_direction[int(k)] = float(v)
        except (TypeError, ValueError):
            by_direction[str(k)] = float(v)
    return {
        "overall_hit_rate": float(n_correct / n),
        "n_events": n,
        "n_correct": n_correct,
        "by_asset": by_asset,
        "by_direction": by_direction,
    }


def split_temporal_windows(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split events into ``(older, middle, recent)`` chronological thirds.

    Events are ordered by ``breakout_timestamp`` and split into three
    near-equal groups by position (``numpy.array_split``), so the split is by
    timestamp *order*, not by an arbitrary row grouping. Returns empty frames
    for any third that has no events (e.g. very small inputs).
    """
    _validate(df)
    sub = df.sort_values("breakout_timestamp").reset_index(drop=True)
    if sub.empty:
        empty = sub.iloc[0:0]
        return empty, empty.copy(), empty.copy()
    parts = np.array_split(np.arange(len(sub)), 3)
    return (
        sub.loc[parts[0]].reset_index(drop=True),
        sub.loc[parts[1]].reset_index(drop=True),
        sub.loc[parts[2]].reset_index(drop=True),
    )
