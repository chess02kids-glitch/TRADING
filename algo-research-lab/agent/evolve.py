"""
Evolution operators for Generations 2 and 3.

Mutation rules (per parameter): 0.30 perturb +/-20%, 0.10 resample from
the full range, 0.60 keep unchanged. 5-10 children per survivor.

Crossover rules (same signal type ONLY): signal type from parent A,
holding bars from parent B, size from the parent with better Sharpe,
other params 50/50 split.

Gen 3 twist: each mutated survivor child gets exactly one structural
twist - longer holding (+50%), alternative exit, or an added HAR regime
filter (window A types only).
"""
from __future__ import annotations

import copy
import json
import os
from typing import List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB_ROOT = os.path.dirname(HERE)

import sys
sys.path.insert(0, LAB_ROOT)

from strategy_genome.generator import GenomeGeneratorV2, PARAM_SPACE_V2, genome_id

P_PERTURB, P_RESAMPLE, P_KEEP = 0.30, 0.10, 0.60


def _mutate_params(genome: dict, rng) -> dict:
    g = copy.deepcopy(genome)
    st = g["signal_type"]
    for key, spec in PARAM_SPACE_V2[st].items():
        if key not in g:
            continue
        u = rng.random()
        if u < P_KEEP:
            continue
        elif u < P_KEEP + P_PERTURB and isinstance(g[key], (int, float)) and not isinstance(g[key], bool):
            sign = 1 if rng.random() < 0.5 else -1
            new = g[key] * (1 + sign * 0.20)
            if isinstance(g[key], int):
                new = max(2, int(round(new)))
            g[key] = new
        else:
            kind = spec[0]
            if kind == "uniform":
                g[key] = float(rng.uniform(spec[1], spec[2]))
            elif kind == "randint":
                g[key] = int(rng.randint(spec[1], spec[2] + 1))
            elif kind == "choice":
                g[key] = spec[1][int(rng.randint(len(spec[1])))]
            elif kind == "boolean":
                g[key] = bool(rng.random() < 0.5)
    return g


def _re_id(g: dict, parents=None, gen=None) -> dict:
    g.pop("genome_id", None)
    g.pop("name", None)
    if parents:
        g["parents"] = parents
    if gen:
        g["generation"] = gen
    g["genome_id"] = genome_id(g)
    g["name"] = f"{g['signal_type'][:10]}_{g['genome_id']}"
    return g


def mutate_survivor(survivor_genome: dict, rng, n_children: int, gen_number: int) -> List[dict]:
    children = []
    for _ in range(n_children):
        child = _mutate_params(survivor_genome, rng)
        children.append(_re_id(child, parents=[survivor_genome["genome_id"]], gen=gen_number))
    return children


def crossover(a: dict, b: dict, rng, gen_number: int):
    """Same signal type only."""
    if a["signal_type"] != b["signal_type"]:
        raise ValueError("crossover between different signal types is forbidden")
    child = copy.deepcopy(a)
    st = a["signal_type"]
    sharpe_a = a.get("_sharpe", 0) or 0
    sharpe_b = b.get("_sharpe", 0) or 0
    # holding bars from parent B
    for k in ("holding_bars",):
        if k in b:
            child[k] = b[k]
    # size from the better-Sharpe parent
    better = a if sharpe_a >= sharpe_b else b
    for k in ("size_pct", "size_formula"):
        if k in better:
            child[k] = better[k]
    # everything else 50/50
    for k, spec in PARAM_SPACE_V2[st].items():
        if k in ("size_pct", "size_formula") or k not in a or k not in b or k == "holding_bars":
            continue
        if isinstance(a[k], (bool, str)):
            child[k] = a[k] if rng.random() < 0.5 else b[k]
    return _re_id(child, parents=[a["genome_id"], b["genome_id"]], gen=gen_number)


def next_generation(survivors: List[dict], prev_results: List[dict], n: int, seed: int,
                    gen_number: int) -> List[dict]:
    rng = np.random.RandomState(seed)
    by_id = {r["genome_id"]: r for r in prev_results}
    sur = []
    for s in survivors:
        g = copy.deepcopy(s["genome"])
        g["_sharpe"] = (s.get("metrics") or {}).get("sharpe") or 0
        sur.append(g)
    genomes = []
    per_survivor = max(3, min(10, n // max(1, len(sur)) // 2))
    for g in sur:
        genomes += mutate_survivor(g, rng, per_survivor, gen_number)
    # crossovers between survivors of the same type
    while len(genomes) < n and len(sur) >= 2:
        i, j = rng.choice(len(sur), size=2, replace=False)
        if sur[int(i)]["signal_type"] == sur[int(j)]["signal_type"]:
            try:
                genomes.append(crossover(sur[int(i)], sur[int(j)], rng, gen_number))
            except Exception:
                genomes.append(mutate_survivor(sur[int(i)], rng, 1, gen_number)[0])
        else:
            genomes.append(mutate_survivor(sur[int(i)], rng, 1, gen_number)[0])
    # gen 3 twist: mutate survivors with one structural twist each
    if gen_number >= 3 and sur:
        for g in sur[: max(1, len(sur))]:
            tw = twist(g, rng, gen_number)
            if tw:
                genomes.append(tw)
    return genomes[:n]


def twist(genome: dict, rng, gen_number: int):
    """One structural twist per survivor (Gen 3)."""
    g = copy.deepcopy(genome)
    opts = ["longer_hold"]
    st = g["signal_type"]
    if st in ("funding_rate_contrarian",):
        opts.append("flip_exit")
    if st in ("vol_regime_breakout", "multi_asset_momentum", "spread_zscore"):
        opts.append("har_filter")
    pick = opts[int(rng.randint(len(opts)))]
    if pick == "longer_hold" and "holding_bars" in g:
        g["holding_bars"] = int(max(2, round(g["holding_bars"] * 1.5)))
    elif pick == "flip_exit":
        g["exit_type"] = "funding_flip" if g.get("exit_type") != "funding_flip" else "fixed_hold"
    elif pick == "har_filter":
        g["har_regime_filter"] = str(rng.choice(["low", "medium"]))
    else:
        return None
    g["twist"] = pick
    return _re_id(g, parents=[genome["genome_id"]], gen=gen_number)


# Gen 2/3 focused exploration ranges: WIDER than the pre-registered Gen 1
# space (documented deviation, allowed by the lab brief for the
# "0 survivors -> new random batch with wider parameter exploration"
# branch). Longer holding periods and simpler entry conditions.
FOCUSED_SPACE = {
    "spread_zscore": {
        "zscore_window": ("randint", 12, 336),
        "entry_zscore": ("uniform", 1.0, 3.0),
        "exit_zscore": ("uniform", 0.0, 1.0),
    },
    "funding_rate_contrarian": {
        "long_threshold": ("uniform", -0.05, -0.003),
        "short_threshold": ("uniform", 0.003, 0.10),
        "holding_bars": ("randint", 1, 48),
        "exit_type": ("choice", ["fixed_hold", "funding_flip"]),
    },
    "multi_asset_momentum": {
        "lookback_bars": ("randint", 12, 240),
        "momentum_threshold": ("uniform", 0.002, 0.05),
        "holding_bars": ("randint", 2, 48),
    },
    "funding_trend": {
        "funding_ma_window": ("randint", 3, 48),
        "trend_threshold": ("uniform", 0.0005, 0.02),
        "holding_bars": ("randint", 2, 48),
    },
    "har_regime_sized": {
        "holding_bars": ("randint", 1, 24),
        "breakout_multiplier": ("uniform", 1.5, 4.0),
    },
    "vol_regime_breakout": {
        "expansion_factor": ("uniform", 1.05, 2.0),
        "breakout_lookback": ("randint", 5, 96),
        "holding_bars": ("randint", 2, 48),
        "regime_filter": ("choice", ["any", "any", "low_only"]),
    },
}


def focused_fresh_batch(prev_results: List[dict], n: int, seed: int, gen_number: int) -> List[dict]:
    """Gen 2/3 when the previous generation had 0 survivors: a fresh random
    batch focused on the signal types that came closest (best pre-failure
    Sharpe), with wider parameter exploration and longer holding periods
    and simpler entry conditions."""
    if not prev_results:
        gen = GenomeGeneratorV2(seed)
        return gen.generate(n)
    stats = {}
    for r in prev_results:
        t = r["genome"]["signal_type"]
        s = (r.get("metrics") or {}).get("sharpe") or -9e9
        d = stats.setdefault(t, {"count": 0, "best": -9e9})
        d["count"] += 1
        d["best"] = max(d["best"], s)
    ranked = sorted(stats.items(), key=lambda kv: kv[1]["best"], reverse=True)
    top_types = [t for t, _ in ranked[:3]]
    print(f"[evolve] focused fresh batch on top types by best Sharpe: "
          f"{[(t, round(d['best'],2)) for t, d in ranked]}")
    gen = GenomeGeneratorV2(seed)
    genomes = []
    weights = [0.45, 0.33, 0.22]
    for t, w in zip(top_types, weights):
        genomes += gen.generate_many_focused(t, int(round(n * w)), FOCUSED_SPACE.get(t, {}))
    while len(genomes) < n:
        genomes += gen.generate(1)
    # longer holding periods on a random 30% (breathe for 12-48h)
    rng = np.random.RandomState(seed + 1)
    for g in genomes:
        if "holding_bars" in g and rng.random() < 0.30:
            g["holding_bars"] = int(min(48, round(g["holding_bars"] * 1.5)))
        # simpler entry conditions
        if g["signal_type"] == "multi_asset_momentum" and rng.random() < 0.7:
            g["require_confirmation"] = False
        if g["signal_type"] == "funding_trend" and rng.random() < 0.5:
            g["price_confirm"] = True
        g.pop("genome_id", None)
        g["genome_id"] = genome_id(g)
        g["name"] = f"{g['signal_type'][:10]}_{g['genome_id']}"
    return genomes[:n]
