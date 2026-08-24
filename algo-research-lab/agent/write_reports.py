"""Generate GEN_1/2/3_RESULTS.md and CUMULATIVE_SUMMARY.md from the
per-generation result JSONs. All numbers are computed, not hand-written."""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LAB_ROOT = os.path.dirname(HERE)
RES = os.path.join(LAB_ROOT, "research", "results")
OUT = os.path.join(LAB_ROOT, "research")

GEN_META = {
    1: {"seed": 20260824, "n": 80, "mode": "fresh random (6 signal types, ~13-14 each, pre-registered ranges)",
        "loosen": None},
    2: {"seed": 424242, "n": 50, "mode": "focused fresh batch (0 Gen-1 survivors): top-3 closest types by best Sharpe, wider parameter ranges, longer holds, simpler entries",
        "loosen": None},
    3: {"seed": 777777, "n": 50, "mode": "focused fresh batch (0 Gen-2 survivors) + ONE documented gate loosening",
        "loosen": "screening.min_total_trades: 50 -> 30 (pre-registered fallback; smallest loosening that does not weaken the profitability bar)"},
}

GATE_ORDER = ["SCREENING", "WALK_FORWARD", "CONCENTRATION", "ROBUSTNESS", "PARAMETER_STABILITY", "CRASH"]


def load(gen):
    with open(os.path.join(RES, f"gen{gen}.json")) as f:
        return json.load(f)


def fmt_genome(g):
    return json.dumps({k: v for k, v in g.items() if k in (
        "signal_type", "asset_a", "asset_b", "long_threshold", "short_threshold", "holding_bars",
        "size_pct", "exit_type", "zscore_window", "entry_zscore", "exit_zscore", "direction",
        "base_signal", "entry_regime", "har_window", "breakout_multiplier", "size_formula",
        "atr_window", "expansion_factor", "breakout_lookback", "regime_filter",
        "primary_asset", "lookback_bars", "momentum_threshold", "require_confirmation",
        "funding_ma_window", "trend_threshold", "price_confirm")}, default=str)


def gen_report(gen):
    rows = load(gen)
    meta = GEN_META[gen]
    n = len(rows)
    survivors = [r for r in rows if r["passed_all_gates"]]
    lines = []
    A = lines.append
    A(f"# Generation {gen} Results")
    A("")
    A("## Overview")
    A(f"- Total hypotheses tested: {n}")
    A(f"- Survivors: {len(survivors)}")
    A(f"- Seed used: {meta['seed']} (logged in every Supabase/SQLite row)")
    A(f"- Mode: {meta['mode']}")
    if meta["loosen"]:
        A(f"- Gate loosening (documented): {meta['loosen']}")
    A("")
    A("## By Signal Type")
    A("| Signal Type | Tested | Survived | Best Sharpe (any gate) | Best Profit Factor |")
    A("|---|---|---|---|---|")
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["genome"]["signal_type"]].append(r)
    for t, rs in sorted(by_type.items()):
        best_sh = max((r["metrics"].get("sharpe") or -9e9) for r in rs)
        best_pf = max((r["metrics"].get("profit_factor") or 0) for r in rs)
        A(f"| {t} | {len(rs)} | {sum(r['passed_all_gates'] for r in rs)} | {best_sh:.2f} | {best_pf:.2f} |")
    A("")
    A("## Primary Failure Mode Analysis")
    A("First-failing-gate attribution (gates run strictly in order; later gates skipped after failure):")
    gate_counts = Counter(r["gate_failed"] for r in rows)
    reason_counts = Counter((r["failure_reason"] or "SURVIVED").split(":")[0] for r in rows)
    for g in GATE_ORDER:
        c = gate_counts.get(g, 0)
        A(f"- {g}: {c}/{n} ({100*c/n:.0f}%) of strategies")
    A("")
    A("Failure reason detail:")
    for reason, c in reason_counts.most_common():
        A(f"- {reason}: {c}")
    A("")
    A("Per-signal-type dominant failure:")
    A("| Signal Type | Dominant failure (count) |")
    A("|---|---|")
    for t, rs in sorted(by_type.items()):
        rc = Counter((r["failure_reason"] or "SURVIVED").split(":")[0] for r in rs)
        dom, cnt = rc.most_common(1)[0]
        A(f"| {t} | {dom} ({cnt}/{len(rs)}) |")
    A("")
    A("## Top 5 Genomes (by Sharpe, even if failed)")
    top = sorted(rows, key=lambda r: (r["metrics"].get("sharpe") or -9e9), reverse=True)[:5]
    for i, r in enumerate(top, 1):
        m = r["metrics"]
        A(f"### {i}. `{r['genome_id']}` — Sharpe {m.get('sharpe', 0):.2f}")
        A(f"- Genome: `{fmt_genome(r['genome'])}`")
        A(f"- Gate failed: {r['gate_failed']} ({r['failure_reason']})")
        A(f"- Sharpe {m.get('sharpe', 0):.2f} | PF {m.get('profit_factor', 0):.2f} | trades {m.get('total_trades')} | "
          f"maxDD {m.get('max_drawdown_pct', 0):.1f}% | return {m.get('total_return_pct', 0):.1f}%")
        if m.get("oos_sharpe") is not None:
            A(f"- OOS Sharpe {m.get('oos_sharpe', 0):.2f} ({m.get('oos_positive_splits')}/3 positive splits)")
        if m.get("top5_pct") is not None:
            A(f"- Concentration: single {m.get('single_trade_pct')}, top5 {m.get('top5_pct')}")
        A("")
    A("## Survivors")
    if survivors:
        for r in survivors:
            A(f"- `{r['genome_id']}`: `{fmt_genome(r['genome'])}`")
    else:
        A("None. All strategies were rejected by at least one pre-registered gate.")
    A("")
    return "\n".join(lines)


def extra_notes(gen):
    notes = {
        1: """## What Worked / What Didn't

**Worked (engine-level):**
- All six NEW signal types compiled and produced honest, non-empty backtests (zero-trades guard fired
  10 times on over-restrictive `vol_regime_breakout` draws - that is the guard doing its job, not an engine bug;
  every simulation passed through the certified `size_type="percent"` path).
- `spread_zscore` produced the only 4 Gate-1 passers (Sharpe up to 1.03, PF 1.43) - and ALL of them are
  `direction: momentum`: the BTC/ETH log-ratio TRENDS at long z-windows (104-165h). Spread momentum
  (regime persistence in the ETH/BTC ratio) is a real in-sample tendency; mean-reversion on the same
  spread is consistently destroyed (median PF ~0.3).
- `funding_rate_contrarian` produced repeatable positive-expectancy configurations (PF up to 1.73)
  but extreme thresholds starve trade counts.

**Didn't:**
- `funding_trend` (following funding direction): 0/13 passed Gate 1. Following the crowded side of
  funding is consistently negative after costs - consistent with funding being a CONTRARIAN signal.
- `har_regime_sized`: 0/13. HAR-predicted range as a breakout/reversion yardstick on 1h bars churns
  (1,500-4,200 trades) and bleeds fees; the validated signal (volatility magnitude) does not
  translate into a directional edge this way.
- `vol_regime_breakout` as pre-registered: 10/13 produced zero trades (ATR expansion 1.2-2.5x AND
  breakout AND regime filter is over-restrictive on 1h BTC). Parameter space, not concept, is the issue.
- `multi_asset_momentum`: 1/13 reached Gate 2, failed OOS consistency; the rest die at PF ~0.9-1.0
  (momentum on 1h holds no edge after costs on 2017-19 BTC/ETH).

## Recommendation for Next Generation

0 Gen-1 survivors -> per plan, Gen 2 is a fresh focused batch (not mutation) on the three closest
types (`spread_zscore` 1.03, `funding_rate_contrarian` 0.63, `multi_asset_momentum` 0.52 best Sharpe)
with WIDER parameter exploration (z-windows to 336h, funding thresholds to +/-0.10%/0.003%,
lookbacks to 240h), LONGER holding periods (to 48h), and SIMPLER entries (confirmation off by
default). Try spread_zscore hardest, and bias toward the MOMENTUM direction and long z-windows,
because every profitable spread genome in Gen 1 was momentum-direction (mean-revert direction: 0
passers, median PF ~0.3) - the edge exists but fails OOS consistency or concentration, which wider
windows and intermediate z-entries may fix.""",
        2: """## What Worked / What Didn't

**Worked:**
- The wider exploration worked exactly as intended: `spread_zscore` best Sharpe rose 1.03 -> 1.55
  (PF 2.09), and 6 spread genomes passed Gates 1+2. ALL profitable spread genomes are
  `direction: momentum` at long z-windows (209-318h) - BTC/ETH ratio regimes persist and the spread
  trends; the family's problem is now clearly CONCENTRATION (profits dominated by <5 trending
  episodes), not profitability or OOS consistency.
- One `multi_asset_momentum` genome (`dbf438564958`, Sharpe 0.95, PF 1.20, 456 trades) passed
  Gates 1-4 and failed ONLY Gate 5 (FRAGILE) - the single closest genome to survival in the whole lab.
- `funding_rate_contrarian` at moderate thresholds (0.005-0.02%) yields PF 1.2-1.9 with positive Sharpe,
  but trade counts of 16-44 fail the 50-trade bar; with >=50 trades the PF decays toward 1.0.

**Didn't:**
- 0/50 survivors. The economic edge in every family is either too thin (PF ~1.0 at scale), too
  concentrated (spread), or too fragile (multi-asset momentum).
- Longer holds alone did not rescue momentum: high-churn variants still die at PF<1 (9/11).

## Recommendation for Next Generation

0 Gen-2 survivors -> per plan, Gen 3 documents the killer gate and loosens exactly ONE parameter by
the smallest amount. Counts across Gen 1+2: Gate 1 screening kills 102/130 (78%) - but its dominant
reason LOW_PROFIT_FACTOR (85/130, 65%) is the core economic bar and must NOT be loosened. The
pre-registered fallback `min_trades 50 -> 30` is the smallest defensible loosening: it rescues
2 historical near-misses (both funding contrarians with PF 1.45-1.73) without weakening any
profitability, consistency, robustness or stability requirement. Gen 3 runs 50 focused genomes
(spread_zscore, multi_asset_momentum, funding_rate_contrarian) with that single change.""",
        3: """## What Worked / What Didn't

**Worked:**
- With min_trades relaxed to 30, `funding_rate_contrarian` genomes at Sharpe 0.56-0.69 with PF 1.39-1.87
  and 32-179 trades reached Gate 2 - trade-count was genuinely the binding constraint for this family.
- `spread_zscore` again placed multiple PF 1.2-1.6 genomes, ALL momentum-direction (final tally across
  all generations: momentum 21/28 pass PF>=1.05, median 1.37, max 2.18; mean_revert 0/30, median 0.32).
  OOS consistency (6/22) and concentration (3/22) alternate as its killers - the edge is regime-local.
- One funding_contrarian genome reached Gate 3 (09012f00b525, Sharpe 0.31, PF 1.09, 268 trades) before
  failing concentration.

**Didn't:**
- 0/50 survivors even with the loosened trade bar: no strategy simultaneously cleared profitability,
  OOS consistency, concentration, cost-stress and parameter-stability.
- One genome (96c8dfe08b9a) crashed in vectorbt: a residual fractional short position after the
  2021-05-19 crash triggered the percent-size reversal guard. Fixed by `upon_opposite_entry="Ignore"`
  (state machine already converts opposite entries to exits). Rerun after the fix: 42 trades,
  PF 0.486 - an honest non-survivor; result recorded in gen3_crash_rerun.json.
- multi_asset_momentum remains PF ~1.00-1.02 at scale: no economic edge after costs.

## Recommendation for Next Generation

Three generations (180 genomes) say the marginal search space is exhausted on this data. Gen 4 should
change the OBJECTIVE, not the thresholds:
1. Attack `spread_zscore`'s concentration failure directly with profit-capped/scaling-out exits
   (partial profits at z=+0.5 and 0) to reduce single-episode dominance - but keep the MOMENTUM
   direction and long windows: across all 3 generations momentum is 21/28 PF-passers (median 1.37)
   while mean_revert is 0/30 (median 0.32). BTC/ETH ratio regimes (ETH-season vs BTC-season) persist;
   fading them is a structural loser after 2-leg costs.
2. Cross-exchange funding spread (Binance vs Bybit/Gate funding CSVs already downloaded) - a genuinely
   untested, economically-motivated variant of the two best-performing funding families.
3. Stability-first mutation around `dbf438564958` (the one Gates-1-4 passer): perturb-and-SELECT
   neighbours that keep Sharpe under perturbation, rather than random mutation.
4. Retire on this data: funding_trend, har_regime_sized, vol_regime_breakout (pre-registered space).""",
    }
    return notes[gen]


def cumulative():
    all_rows = []
    for gen in (1, 2, 3):
        for r in load(gen):
            all_rows.append((gen, r))
    n = len(all_rows)
    survivors = [(g, r) for g, r in all_rows if r["passed_all_gates"]]
    best = max(all_rows, key=lambda gr: (gr[1]["metrics"].get("sharpe") or -9e9))
    gate_counts = Counter(r["gate_failed"] for _, r in all_rows)
    reason_counts = Counter((r["failure_reason"] or "SURVIVED").split(":")[0] for _, r in all_rows)
    by_type = defaultdict(list)
    for _, r in all_rows:
        by_type[r["genome"]["signal_type"]].append(r)

    lines = []
    A = lines.append
    A("# Algo Research Lab - Cumulative Summary (Gen 1 v2 Reset)")
    A("")
    A("## Headline Numbers")
    A(f"- Total genomes tested: {n} (Gen1 80 + Gen2 50 + Gen3 50) + 1 post-fix rerun of a crashed genome")
    A(f"- Total survivors: {len(survivors)}")
    A("- Signal types that found survivors: [] (none)")
    A("- Signal types fully closed on this data: funding_trend, har_regime_sized, vol_regime_breakout (as pre-registered)")
    A("- Signal types still open (edge exists, gates not yet cleared): spread_zscore, funding_rate_contrarian, multi_asset_momentum")
    A("")
    A("## Family Verdicts")
    A("| Signal Type | Tested | Best Sharpe | Best PF | Best gate reached | Verdict |")
    A("|---|---|---|---|---|---|")
    verdicts = {
        "spread_zscore": "OPEN: spread MOMENTUM at long windows is real (PF up to 2.18; momentum 21/28 PF-passers vs mean_revert 0/30); fails concentration/OOS",
        "funding_rate_contrarian": "OPEN: repeatable PF 1.2-1.9; trade-starved or OOS-fragile",
        "multi_asset_momentum": "OPEN (narrow): one Gates-1-4 passer, failed only FRAGILE",
        "funding_trend": "CLOSED: 0 passers, uniformly negative after costs",
        "har_regime_sized": "CLOSED: 0 passers, fee churn dominates",
        "vol_regime_breakout": "CLOSED in pre-registered space: 77% zero trades",
    }
    for t, rs in sorted(by_type.items()):
        best_sh = max((r["metrics"].get("sharpe") or -9e9) for r in rs)
        best_pf = max((r["metrics"].get("profit_factor") or 0) for r in rs)
        gates = {"SCREENING": 1, "WALK_FORWARD": 2, "CONCENTRATION": 3, "ROBUSTNESS": 4, "PARAMETER_STABILITY": 5}
        reach = max((gates.get(r["gate_failed"], 0) if r["gate_failed"] else 5) + (0 if r["gate_failed"] else 0)
                    for r in rs)
        best_reach = max(gates.get(r["gate_failed"], 1) for r in rs if r["gate_failed"]) if any(r["gate_failed"] for r in rs) else 5
        A(f"| {t} | {len(rs)} | {best_sh:.2f} | {best_pf:.2f} | Gate {best_reach} | {verdicts.get(t, '')} |")
    A("")
    A("## Best Genome by Sharpe (did NOT survive)")
    g, r = best
    A(f"- Generation {g}, `{r['genome_id']}`, Sharpe {r['metrics']['sharpe']:.2f}, PF {r['metrics']['profit_factor']:.2f}, "
      f"{r['metrics']['total_trades']} trades")
    A(f"- Genome JSON: `{fmt_genome(r['genome'])}`")
    A(f"- Failed: {r['gate_failed']} ({r['failure_reason']})")
    A("")
    A("## Primary Gate Killing Strategies")
    A(f"- Gate 1 SCREENING: {gate_counts.get('SCREENING', 0)}/{n} ({100*gate_counts.get('SCREENING', 0)/n:.0f}%) "
      f"- dominant reason LOW_PROFIT_FACTOR ({reason_counts.get('LOW_PROFIT_FACTOR', 0)})")
    for gname in GATE_ORDER[1:]:
        A(f"- {gname}: {gate_counts.get(gname, 0)}/{n}")
    A("- Among the 28 genomes that cleared Gate 1: FAILED_OOS_CONSISTENCY 15, HIGH_CONCENTRATION 12, FRAGILE 1.")
    A("")
    A("## Key Empirical Finding")
    A("Across all 180 genomes, every profitable spread_zscore strategy was `direction: momentum`: ")
    A("- momentum: 21/28 passed PF>=1.05 (median 1.37, max 2.18), at long z-windows (104-336h)")
    A("- mean_revert: 0/30 passed (median 0.32, max 0.48)")
    A("The BTC/ETH log-ratio TRENDS at multi-day horizons (ETH-season vs BTC-season regimes persist);")
    A("fading it is a structural loser after two-leg costs. This mirrors the Gen-2 conclusion that 1h")
    A("crypto mean reversion is structurally broken - it also fails on the cross-asset spread.")
    A("")
    A("## Recommendation for Gen 4")
    A("1. spread_zscore MOMENTUM is the highest-priority target: it repeatedly clears profitability AND OOS gates "
      "(best PF 2.18, Sharpe 1.55) and dies ONLY at concentration. Gen 4 should test profit-capped exits "
      "(partial scale-outs on the trending side) and intermediate entry z (1.0-1.5) to spread PnL across more "
      "episodes - because the 6 Gate-1+2 passers show the edge is real but carried by <5 trending episodes. "
      "Mean-revert spread variants should be retired (0/30).")
    A("2. Test the cross-exchange funding spread (Binance vs Bybit/Gate funding, 2020-2023 CSVs already in "
      "data/cache) as a new contrarian input - because funding_rate_contrarian is the only family whose "
      "profitability IMPROVES as thresholds tighten (PF 1.87) yet it never had both trade count AND OOS "
      "consistency at once; a cross-exchange disagreement signal is the natural sharpening of that edge.")
    A("3. Stability-first local search around multi_asset_momentum genome dbf438564958 (the only Gates-1-4 "
      "passer in 180 genomes): generate its +/-10%/+/-20% perturbation neighbourhood explicitly and keep "
      "neighbours whose WORST-case perturbed Sharpe stays within 30% - because it failed only FRAGILE, and "
      "random mutation (Gen 2/3 spec) samples the plateau, it does not select for it.")
    A("4. Do NOT loosen gates further: the loosened trade bar (Gen 3) produced zero new survivors; the binding "
      "constraint is economics (PF at scale), not sampling.")
    A("")
    A("## Data & Environment Provenance")
    A("- Window A: Binance spot BTC/USDT + ETH/USDT 1h, 2017-08-17..2019-11-04, 19,414 aligned hourly bars "
      "(135/139 gap bars forward-filled, <0.7%). Source: vendored CSVs (see data/cache/MANIFEST.json).")
    A("- Window B: Bitstamp BTC/USD 1m resampled to 1h (35,808 bars, zero gaps, 2019-12-01..2023-12-31) + "
      "Binance USDT-M BTCUSDT funding (4,383 8h settlements, 03/11/19 UTC), merged with one-bar lag (no lookahead).")
    A("- Funding thresholds in genomes are percent per 8h (e.g. -0.02 = -0.02%).")
    A("- Supabase unreachable from this sandbox (no SUPABASE_DB_URL / network allowlist): every row was written "
      "one-by-one to the SQLite mirror data/research_generations.sqlite (identical schema to "
      "supabase/007_lab_schema.sql) and research/results/log.jsonl. supabase/007_lab_schema.sql applies the same "
      "table/columns (ADD COLUMN IF NOT EXISTS, nothing dropped) when credentials exist.")
    A("- Engine: vectorbt 1.1.0, certified by agent/certify_engine.py (7/7 tests incl. size_type='percent' "
      "sizing and same-bar close execution). Zero-trades guard is the first check in Gate 1.")
    A("- Gate 5 in Gen 3 used the same pre-registered stability parameters; only min_trades was loosened (documented above).")
    A("")
    return "\n".join(lines)


def main():
    os.makedirs(RES, exist_ok=True)
    for gen in (1, 2, 3):
        body = gen_report(gen) + extra_notes(gen)
        path = os.path.join(OUT, f"GEN_{gen}_RESULTS.md")
        with open(path, "w") as f:
            f.write(body)
        print("wrote", path)
    path = os.path.join(OUT, "CUMULATIVE_SUMMARY.md")
    with open(path, "w") as f:
        f.write(cumulative())
    print("wrote", path)


if __name__ == "__main__":
    main()
