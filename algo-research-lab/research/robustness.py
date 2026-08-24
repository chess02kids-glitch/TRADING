"""
Gate 4 - Robustness (cost stress).

Pre-registered parameters (GATE_CONFIG["robustness"]):
  fee_scenarios     = [0.001, 0.002, 0.003]      (per side)
  slippage_scenarios= [0.0005, 0.001, 0.002]     (per side)
  scenarios are paired by index -> 3 scenarios
  must_pass_majority= True (>= 2 of 3)
A scenario passes when total return remains positive under its costs.
"""
from __future__ import annotations

from typing import Dict, Tuple

from research.gate_config import GATE_CONFIG
from research.screener import simulate, portfolio_metrics


def gate4_cost_stress(spec) -> Tuple[bool, Dict]:
    cfg = GATE_CONFIG["robustness"]
    scenarios = list(zip(cfg["fee_scenarios"], cfg["slippage_scenarios"]))
    rows = []
    n_pass = 0
    survival_limit = 0.0
    for fees, slip in scenarios:
        pf = simulate(
            spec.price,
            spec.entries,
            spec.exits,
            spec.short_entries,
            spec.short_exits,
            size=spec.size_for_slice(None),
            fees=fees * spec.leg_multiplier,
            slippage=slip * spec.leg_multiplier,
        )
        m = portfolio_metrics(pf)
        ok = m["total_return_pct"] > 0
        n_pass += int(ok)
        if ok:
            survival_limit = max(survival_limit, fees)
        rows.append({
            "fees": fees, "slippage": slip,
            "return_pct": round(m["total_return_pct"], 3),
            "sharpe": round(m["sharpe"], 3),
            "profit_factor": round(m["profit_factor"], 3),
            "trade_count": m["total_trades"],
            "passed": ok,
        })
    need = len(scenarios) if cfg["must_pass_all_scenarios"] else (len(scenarios) // 2 + 1)
    passed = n_pass >= need
    return passed, {
        "scenarios": rows,
        "n_pass": n_pass,
        "need": need,
        "cost_survival_limit": survival_limit,
    }
