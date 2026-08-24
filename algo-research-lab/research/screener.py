"""
Gate 1 - Screening (fast rejection) with the ZERO-TRADES BUG GUARD.

The zero-trades check is the FIRST check performed on every genome,
before any other gate metric is evaluated. It exists because of the
historical vectorbt bug: passing `size` arrays without
`size_type="percent"` silently produced 0 trades and empty portfolios
that "passed" every gate. It must never slip through again.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

from research.gate_config import GATE_CONFIG


def simulate(
    price: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    short_entries: pd.Series = None,
    short_exits: pd.Series = None,
    size=None,
    fees: float = None,
    slippage: float = None,
    init_cash: float = None,
):
    """Single entry point for every portfolio simulation in the lab.

    ALWAYS passes size_type="percent" (the certified fix).
    """
    cfg = GATE_CONFIG["sim"]
    sc = GATE_CONFIG["screening"]
    pf = vbt.Portfolio.from_signals(
        price,
        entries,
        exits,
        short_entries=short_entries if short_entries is not None else pd.Series(False, index=price.index),
        short_exits=short_exits if short_exits is not None else pd.Series(False, index=price.index),
        size=size if size is not None else 1.0,
        size_type="percent",
        fees=sc["base_fees"] if fees is None else fees,
        slippage=sc["base_slippage"] if slippage is None else slippage,
        freq=cfg["freq"],
        init_cash=cfg["init_cash"] if init_cash is None else init_cash,
        # opposite entries while positioned are converted to exits by the
        # generator's state machine; tell vectorbt to never attempt a
        # percent-size position reversal (which it rejects by design).
        upon_opposite_entry="Ignore",
    )
    return pf


def portfolio_metrics(pf) -> Dict[str, float]:
    stats = pf.stats()
    out = {
        "total_trades": int(stats.get("Total Trades", 0) or 0),
        "total_return_pct": float(stats.get("Total Return [%]", 0.0) or 0.0),
        "sharpe": float(stats.get("Sharpe Ratio", 0.0) or 0.0),
        "sortino": float(stats.get("Sortino Ratio", 0.0) or 0.0),
        "max_drawdown_pct": float(stats.get("Max Drawdown [%]", 0.0) or 0.0),
        "profit_factor": float(stats.get("Profit Factor", 0.0) or 0.0),
        "win_rate_pct": float(stats.get("Win Rate [%]", 0.0) or 0.0),
    }
    for k, v in out.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out[k] = 0.0
    return out


def gate1_screening(pf) -> Tuple[bool, Dict[str, str]]:
    """Gate 1. Returns (passed, {reason, gate, failure_code})."""
    cfg = GATE_CONFIG["screening"]
    metrics = portfolio_metrics(pf)

    # ---- ZERO-TRADES BUG GUARD: FIRST CHECK, ALWAYS ----------------------
    if metrics["total_trades"] == 0:
        return False, {"gate": "SCREENING", "reason": "ZERO_TRADES_BUG", "passed": False}
    # -----------------------------------------------------------------------

    if metrics["total_trades"] < cfg["min_total_trades"]:
        return False, {"gate": "SCREENING", "reason": "LOW_TRADE_COUNT", "passed": False}
    if metrics["profit_factor"] < cfg["min_profit_factor"]:
        return False, {"gate": "SCREENING", "reason": "LOW_PROFIT_FACTOR", "passed": False}
    if metrics["total_return_pct"] < 0:
        return False, {"gate": "SCREENING", "reason": "NEGATIVE_RETURN", "passed": False}
    if metrics["max_drawdown_pct"] < cfg["max_drawdown_pct"]:
        return False, {"gate": "SCREENING", "reason": "CATASTROPHIC_DRAWDOWN", "passed": False}
    return True, {"gate": "SCREENING", "reason": None, "passed": True}
