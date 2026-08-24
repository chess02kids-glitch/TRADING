"""
Gate 2 - Walk-Forward OOS consistency.

Pre-registered parameters (GATE_CONFIG["walk_forward"]):
  n_splits = 3, train_pct = 0.60, oos_sharpe_threshold = 0.0,
  consistency: OOS Sharpe positive in >= 2 of 3 splits.

Interpretation (documented): the sample is split chronologically into a
60% training region and a 40% out-of-sample region; the OOS region is
divided into n_splits contiguous blocks. All strategy indicators are
strictly backward-looking (computed once on the full series), so slicing
signals into OOS blocks introduces no lookahead. Per-block Sharpe ratios
are computed on each OOS block independently.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from research.gate_config import GATE_CONFIG
from research.screener import simulate, portfolio_metrics


def oos_blocks(n_bars: int, cfg: Dict) -> List[Tuple[int, int]]:
    train_end = int(n_bars * cfg["train_pct"])
    oos_len = n_bars - train_end
    k = cfg["n_splits"]
    block = oos_len // k
    return [(train_end + i * block, train_end + (i + 1) * block) for i in range(k)]


def gate2_walk_forward(spec) -> Tuple[bool, Dict[str, float]]:
    """spec: research.pipeline.SignalSpec"""
    cfg = GATE_CONFIG["walk_forward"]
    n = len(spec.price)
    sharpes: List[float] = []
    rets: List[float] = []
    for (s, e) in oos_blocks(n, cfg):
        sl = slice(s, e)
        pf = simulate(
            spec.price.iloc[sl],
            spec.entries.iloc[sl],
            spec.exits.iloc[sl],
            spec.short_entries.iloc[sl],
            spec.short_exits.iloc[sl],
            size=spec.size_for_slice(sl),
        )
        m = portfolio_metrics(pf)
        sharpes.append(m["sharpe"])
        rets.append(m["total_return_pct"])

    sharpes_arr = np.array(sharpes)
    mean_sharpe = float(np.mean(sharpes_arr))
    positive_splits = int(np.sum(sharpes_arr > 0))

    passed = (
        mean_sharpe > cfg["oos_sharpe_threshold"]
        and positive_splits >= cfg["min_positive_splits"]
    )
    metrics = {
        "oos_sharpe": mean_sharpe,
        "oos_sharpe_splits": [round(float(x), 3) for x in sharpes],
        "oos_return_splits": [round(float(x), 3) for x in rets],
        "positive_splits": positive_splits,
        "n_splits": cfg["n_splits"],
    }
    return passed, metrics
