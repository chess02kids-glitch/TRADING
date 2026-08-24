"""
Five-gate validation pipeline.

Gates run IN ORDER (Rule 2). The first failing gate stops the pipeline
and later gates are skipped; every genome's result is recorded with its
failure code and all metrics computed up to the failure point.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from research.screener import simulate, portfolio_metrics, gate1_screening
from research.walk_forward import gate2_walk_forward
from research.concentration import gate3_concentration
from research.robustness import gate4_cost_stress
from research.stability import gate5_stability


@dataclass
class SignalSpec:
    """Compiled strategy: what to trade and when.

    size: scalar float or pd.Series of per-bar fractions of equity
          (certified path: always passed with size_type="percent").
    leg_multiplier: cost multiplier (2.0 for two-leg spread trades).
    """
    price: pd.Series
    entries: pd.Series
    exits: pd.Series
    short_entries: pd.Series
    short_exits: pd.Series
    size: Any = 1.0
    leg_multiplier: float = 1.0
    asset: str = "BTC/USDT"
    meta: Dict = field(default_factory=dict)

    def size_for_slice(self, sl: Optional[slice]):
        """Return the size argument appropriate for a (possibly) sliced sim."""
        if isinstance(self.size, pd.Series):
            return self.size.iloc[sl] if sl is not None else self.size
        return self.size


def run_all_gates(genome: dict, ctx, compile_fn, seed: int) -> Dict:
    """Run the 5 gates in order for one genome. Never raises."""
    result = {
        "genome_id": genome.get("genome_id"),
        "genome": genome,
        "passed_all_gates": False,
        "gate_failed": None,
        "failure_reason": None,
        "metrics": {},
        "gate_detail": {},
    }
    try:
        spec = compile_fn(genome, ctx)

        # ---------------- Gate 1: Screening (zero-trades guard first) ------
        pf = simulate(
            spec.price, spec.entries, spec.exits,
            spec.short_entries, spec.short_exits,
            size=spec.size_for_slice(None),
        )
        m1 = portfolio_metrics(pf)
        result["metrics"].update(m1)
        result["asset"] = spec.asset
        ok1, d1 = gate1_screening(pf)
        result["gate_detail"]["gate1"] = d1
        if not ok1:
            result["gate_failed"] = "SCREENING"
            result["failure_reason"] = d1["reason"]
            return result

        # ---------------- Gate 2: Walk-forward OOS consistency ------------
        ok2, m2 = gate2_walk_forward(spec)
        result["metrics"]["oos_sharpe"] = m2["oos_sharpe"]
        result["metrics"]["oos_positive_splits"] = m2["positive_splits"]
        result["gate_detail"]["gate2"] = m2
        if not ok2:
            result["gate_failed"] = "WALK_FORWARD"
            result["failure_reason"] = "FAILED_OOS_CONSISTENCY"
            return result

        # ---------------- Gate 3: Concentration ----------------------------
        ok3, m3 = gate3_concentration(spec)
        result["metrics"]["single_trade_pct"] = m3.get("single_trade_pct")
        result["metrics"]["top5_pct"] = m3.get("top5_pct")
        result["gate_detail"]["gate3"] = m3
        if not ok3:
            result["gate_failed"] = "CONCENTRATION"
            if m3.get("total_pnl", 0.0) <= 0:
                result["failure_reason"] = "NEGATIVE_RETURN"
            else:
                result["failure_reason"] = "HIGH_CONCENTRATION"
            return result

        # ---------------- Gate 4: Robustness (cost stress) -----------------
        ok4, m4 = gate4_cost_stress(spec)
        result["metrics"]["cost_survival_limit"] = m4["cost_survival_limit"]
        result["metrics"]["robustness_pass_scenarios"] = m4["n_pass"]
        result["gate_detail"]["gate4"] = m4
        if not ok4:
            result["gate_failed"] = "ROBUSTNESS"
            result["failure_reason"] = "FAILED_COST_STRESS"
            return result

        # ---------------- Gate 5: Parameter stability ----------------------
        ok5, m5 = gate5_stability(genome, compile_fn, ctx, m1["sharpe"], seed)
        result["metrics"]["stability_score"] = m5["stability_score"]
        result["gate_detail"]["gate5"] = m5
        if not ok5:
            result["gate_failed"] = "PARAMETER_STABILITY"
            result["failure_reason"] = "FRAGILE"
            return result

        result["passed_all_gates"] = True
        return result

    except Exception:  # Rule 3: isolation - one crash never stops the loop
        result["gate_failed"] = "CRASH"
        result["failure_reason"] = "EXCEPTION: " + traceback.format_exc(limit=3).replace("\n", " | ")[-400:]
        return result
