"""
Generation driver for the algo-research-lab Gen 1 v2 reset.

Usage:
  python agent/run_generation.py --gen 1 --n 80 --seed 20260824
  python agent/run_generation.py --gen 2 --n 50 --seed 424242            # evolve if survivors, else fresh focused batch
  python agent/run_generation.py --gen 3 --n 50 --seed 777777 [--loosen screening.min_total_trades=30]

Every genome result is inserted into research_generations (Supabase or
SQLite fallback) immediately, one row at a time, with its seed.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB_ROOT = os.path.dirname(HERE)
sys.path.insert(0, LAB_ROOT)

from agent.lab_context import build_context
from agent.results_logger import ResultsLogger, notify
from agent import evolve
from research.gate_config import GATE_CONFIG
from research.pipeline import run_all_gates
from strategy_genome.generator import GenomeGeneratorV2, compile_genome_v2, PARAM_SPACE_V2

RESULTS_DIR = os.path.join(LAB_ROOT, "research", "results")


def apply_loosen(spec_str: str):
    """--loosen section.key=value (documented gate loosening for Gen 3)."""
    if not spec_str:
        return None
    section, rest = spec_str.split(".", 1)
    key, value = rest.split("=")
    old = GATE_CONFIG[section][key]
    cast = type(old)
    GATE_CONFIG[section][key] = cast(value)
    return f"{section}.{key}: {old} -> {GATE_CONFIG[section][key]}"


def run_batch(genomes, ctx, gen_number, seed, logger, log):
    results = []
    n = len(genomes)
    for i, g in enumerate(genomes, 1):
        t0 = time.time()
        res = run_all_gates(g, ctx, compile_genome_v2, seed=seed + i)
        res["seed"] = seed + i
        res["generation"] = gen_number
        m = res.get("metrics", {})
        row = {
            "generation": gen_number,
            "genome_id": g["genome_id"],
            "genome": json.dumps(g, default=str),
            "signal_type": g["signal_type"],
            "asset": res.get("asset", ""),
            "total_trades": m.get("total_trades", 0),
            "profit_factor": m.get("profit_factor"),
            "sharpe_ratio": m.get("sharpe"),
            "max_drawdown": m.get("max_drawdown_pct"),
            "total_return_pct": m.get("total_return_pct"),
            "passed_all_gates": res["passed_all_gates"],
            "gate_failed": res["gate_failed"],
            "failure_reason": (res["failure_reason"] or "")[:500] if res["failure_reason"] else None,
            "oos_sharpe": m.get("oos_sharpe"),
            "oos_positive_splits": m.get("oos_positive_splits"),
            "concentration_score": m.get("top5_pct"),
            "robustness_score": m.get("robustness_pass_scenarios"),
            "stability_score": m.get("stability_score"),
            "seed": seed + i,
            "parent_genome_ids": g.get("parents"),
        }
        logger.insert_genome_result(row)
        results.append(res)
        status = "SURVIVOR" if res["passed_all_gates"] else f"FAIL {res['gate_failed']}/{res['failure_reason'][:40] if res['failure_reason'] else ''}"
        print(f"[{i}/{n}] {g['genome_id']} {g['signal_type'][:24]:24s} "
              f"sharpe={m.get('sharpe', 0):.2f} pf={m.get('profit_factor', 0):.2f} "
              f"trades={m.get('total_trades', 0):4d} {status} ({time.time()-t0:.1f}s)", flush=True)
        if res["passed_all_gates"]:
            print(f"SURVIVOR FOUND: {g['genome_id']}")
            notify(f"SURVIVOR FOUND gen{gen_number}: {g['genome_id']} {json.dumps(g, default=str)}")
    return results


def load_prev_results(gen_number):
    path = os.path.join(RESULTS_DIR, f"gen{gen_number}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--loosen", type=str, default=None,
                    help="loosen ONE gate, e.g. screening.min_total_trades=30")
    args = ap.parse_args()

    np.random.seed(args.seed)
    loosen_note = apply_loosen(args.loosen)
    if loosen_note:
        print(f"=== GATE LOOSENED (documented): {loosen_note} ===")

    print("=== DATA VERIFICATION ===")
    ctx = build_context()
    print(f"Window A  BTC/USDT 1h OHLCV : {len(ctx['btc_A'])} rows  {ctx['windows']['A'][0]} -> {ctx['windows']['A'][1]}")
    print(f"Window A  ETH/USDT 1h OHLCV : {len(ctx['eth_A'])} rows  (aligned grid)")
    print(f"Window B  BTC price 1h      : {len(ctx['btc_B'])} rows  {ctx['windows']['B'][0]} -> {ctx['windows']['B'][1]}")
    print(f"Funding   BTCUSDT 8h events : {len(ctx['funding_ev_B'])} rows -> hourly merged/shifted")
    print(f"HAR predicted range variants: {sorted(ctx['har_range_A'])} "
          f"(regime counts: " +
          ", ".join(f"{v}: {dict(ctx['har_regime_A'][v].value_counts())}" for v in ("5", "22")) + ")")

    # ---------------- genome construction ----------------
    prev = load_prev_results(args.gen - 1)
    survivors = [r for r in (prev or []) if r.get("passed_all_gates")]
    mode = "fresh"
    genomes = []
    if args.gen >= 2 and survivors:
        mode = "evolve"
        genomes = evolve.next_generation(survivors, prev, args.n, args.seed, gen_number=args.gen)
    elif args.gen >= 2:
        mode = "focused-fresh"
        genomes = evolve.focused_fresh_batch(prev, args.n, args.seed, gen_number=args.gen)
    else:
        gen = GenomeGeneratorV2(args.seed)
        genomes = gen.generate(args.n)

    print(f"=== GENERATION {args.gen} START ===")
    print(f"Seed: {args.seed} | Mode: {mode} | Target genomes: {len(genomes)}")

    logger = ResultsLogger()
    print(f"[ResultsLogger] backend: {logger.mode}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log = open(os.path.join(RESULTS_DIR, f"gen{args.gen}_console.txt"), "w")

    results = run_batch(genomes, ctx, args.gen, args.seed, logger, log)

    out_path = os.path.join(RESULTS_DIR, f"gen{args.gen}.json")
    slim = []
    for r in results:
        slim.append({
            "genome_id": r["genome_id"], "genome": r["genome"],
            "passed_all_gates": r["passed_all_gates"],
            "gate_failed": r["gate_failed"], "failure_reason": r["failure_reason"],
            "metrics": {k: (v if not isinstance(v, (list, dict)) else v) for k, v in r["metrics"].items()},
        })
    with open(out_path, "w") as f:
        json.dump(slim, f, indent=1, default=str)

    # ---------------- summary ----------------
    n_pass = sum(1 for r in results if r["passed_all_gates"])
    print("\n" + "=" * 60)
    print(f"GENERATION {args.gen} COMPLETE: tested={len(results)} survivors={n_pass}")
    by_type = {}
    for r in results:
        t = r["genome"]["signal_type"]
        d = by_type.setdefault(t, {"tested": 0, "survived": 0, "best_sharpe": -9e9, "fail": {}})
        d["tested"] += 1
        d["survived"] += int(r["passed_all_gates"])
        d["best_sharpe"] = max(d["best_sharpe"], r["metrics"].get("sharpe") or -9e9)
        if r["gate_failed"]:
            reason = r["failure_reason"].split(":")[0] if r["failure_reason"] else "?"
            d["fail"][reason] = d["fail"].get(reason, 0) + 1
    for t, d in sorted(by_type.items()):
        print(f"  {t:26s} {d['survived']}/{d['tested']} best_sharpe={d['best_sharpe']:.2f} fails={d['fail']}")
    return results


if __name__ == "__main__":
    main()
