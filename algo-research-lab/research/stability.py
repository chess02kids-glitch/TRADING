"""
Gate 5 - Parameter stability.

Pre-registered parameters (GATE_CONFIG["stability"]):
  perturbation_levels           = [0.10, 0.20]
  n_perturbations_per_level     = 5
  sharpe_degradation_threshold  = 0.30  (Sharpe must not drop > 30%)
  must_pass_majority            = True  (>= 6 of 10 perturbed runs)

Each perturbation run perturbs ALL numeric genome parameters
simultaneously by (1 +/- level) with random sign; integer parameters
are rounded (minimum 2). The perturbed strategy is recompiled and
re-simulated at base costs on the full sample.
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, Tuple

import numpy as np

from research.gate_config import GATE_CONFIG
from research.screener import simulate, portfolio_metrics


def perturb_genome(genome: dict, level: float, rng: np.random.RandomState) -> dict:
    g = copy.deepcopy(genome)
    for key, val in list(g.items()):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        sign = 1.0 if rng.random() < 0.5 else -1.0
        new = val * (1.0 + sign * level)
        if isinstance(val, int):
            new = max(2, int(round(new)))
        else:
            new = float(new)
        g[key] = new
    # categorical / string fields pass through unchanged
    return g


def gate5_stability(genome: dict, compile_fn: Callable, ctx, base_sharpe: float,
                    seed: int) -> Tuple[bool, Dict]:
    cfg = GATE_CONFIG["stability"]
    rng = np.random.RandomState(seed)
    runs = []
    n_pass = 0
    total_runs = 0
    for level in cfg["perturbation_levels"]:
        for i in range(cfg["n_perturbations_per_level"]):
            total_runs += 1
            g2 = perturb_genome(genome, level, rng)
            try:
                spec = compile_fn(g2, ctx)
                pf = simulate(
                    spec.price, spec.entries, spec.exits,
                    spec.short_entries, spec.short_exits,
                    size=spec.size_for_slice(None),
                )
                m = portfolio_metrics(pf)
                if m["total_trades"] == 0:
                    ok = False
                    sharpe_p = 0.0
                else:
                    sharpe_p = m["sharpe"]
                    if base_sharpe <= 0:
                        ok = sharpe_p > base_sharpe
                    else:
                        degradation = (base_sharpe - sharpe_p) / abs(base_sharpe)
                        ok = degradation <= cfg["sharpe_degradation_threshold"]
            except Exception as exc:  # noqa: BLE001 - a crashing perturbation is a fragility signal
                ok = False
                sharpe_p = None
            n_pass += int(ok)
            runs.append({"level": level, "run": i, "sharpe": sharpe_p, "passed": bool(ok)})
    need = total_runs // 2 + 1
    passed = n_pass >= need
    return passed, {
        "runs": runs,
        "n_pass": n_pass,
        "need": need,
        "stability_score": round(n_pass / total_runs, 3) if total_runs else 0.0,
    }
