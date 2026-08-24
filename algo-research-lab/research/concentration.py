"""
Gate 3 - Trade concentration.

Pre-registered parameters (GATE_CONFIG["concentration"]):
  max_single_trade_pct = 0.20  (no single trade > 20% of total PnL)
  max_top5_pct         = 0.60  (top 5 trades < 60% of total PnL)
  min_total_return     = 0.0   (total trade PnL must be positive)

Shares are computed on absolute per-trade PnL (portfolio currency),
so variable position sizing is correctly accounted for.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from research.gate_config import GATE_CONFIG
from research.screener import simulate


def gate3_concentration(spec) -> Tuple[bool, Dict[str, float]]:
    cfg = GATE_CONFIG["concentration"]
    pf = simulate(
        spec.price,
        spec.entries,
        spec.exits,
        spec.short_entries,
        spec.short_exits,
        size=spec.size_for_slice(None),
    )
    trades = pf.trades.records_readable
    n_trades = len(trades)
    if n_trades == 0:
        return False, {
            "gate": "CONCENTRATION", "reason_hint": "ZERO_TRADES_BUG",
            "single_trade_pct": None, "top5_pct": None, "total_pnl": 0.0, "n_trades": 0,
        }

    pnl = trades["PnL"].astype(float)
    total_pnl = float(pnl.sum())
    if total_pnl <= cfg["min_total_return"]:
        return False, {
            "single_trade_pct": None, "top5_pct": None,
            "total_pnl": total_pnl, "n_trades": n_trades,
        }

    pnl_pos = pnl[pnl > 0]
    # Concentration is only meaningful against the profit pool; measure
    # shares of positive PnL relative to net total (conservative).
    sorted_pnl = pnl.sort_values(ascending=False)
    single = float(sorted_pnl.iloc[0]) / total_pnl
    top5 = float(sorted_pnl.head(5).sum()) / total_pnl

    metrics = {
        "single_trade_pct": float(single),
        "top5_pct": float(top5),
        "total_pnl": total_pnl,
        "n_trades": n_trades,
    }
    passed = single <= cfg["max_single_trade_pct"] and top5 <= cfg["max_top5_pct"]
    return passed, metrics
