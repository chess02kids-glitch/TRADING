# Phase 9A — Breakout-Direction Continuation Analysis

A **pure statistical** test of one pre-registered hypothesis, with no
machine-learning models and no live trading. It answers the one question the
existing HAR system leaves open: *HAR predicts **how much** price will move —
does the candle direction of a breakout bar also tell us **which way**?*

## The hypothesis (pre-registered — do not change)

> When `actual_range > 2 × har_predicted_range` (a **breakout event**), the
> breakout bar's candle direction (`close ≥ open` → **UP / +1**, else
> **DOWN / −1`) persists into the next 1, 2, and 3 bars.

This module reads completed breakout rows from the existing
`public.har_predictions` table (read-only — it never writes there) and the
matching candle history from KuCoin, then tests whether the breakout
direction predicts the sign of the forward return.

## Modules

| File | Responsibility |
|------|----------------|
| `direction_calculator.py` | Past-only breakout-bar direction (`+1`/`-1`) and 1/2/3-bar forward returns. No look-ahead by construction. |
| `continuation_tester.py` | Hit rates (overall / per-asset / per-regime), temporal stability across older/middle/recent thirds, degradation flag, and the **G1–G6** gate checks. |
| `dm_test.py` | One-sided Diebold-Mariano test of the directional signal vs a 50/50 random baseline, with HAC (Newey-West) standard errors. |
| `phase9a_runner.py` | CLI orchestrator: fetch rows + candles → analyze → print report → optionally save `PHASE9A_RESULTS.md`. |

## The gates (all must pass → `SIGNAL FOUND`; any fail → `CLOSED`)

| Gate | Rule (operationalization) |
|------|---------------------------|
| **G1** | Overall hit rate **> 55%** *and* every asset **> 55%** at t+1. |
| **G2** | Diebold-Mariano one-sided **p < 0.05** vs the random baseline. |
| **G3** | Both **BTC/USDT** and **ETH/USDT** are present **and** each > 50% (consistency across both assets). |
| **G4** | All three chronological thirds (older / middle / recent) **> 50%** (stable windows). |
| **G5** | The recent third is **not > 10 percentage points** worse than the older third (no degradation). |
| **G6** | At least **30** breakout events per asset (≥ 30 total as fallback). |

## How to run

```bash
# Read-only analysis against Supabase + KuCoin, default horizon t+1:
python -m phase9a.phase9a_runner --db-url "$SUPABASE_DB_URL" --asset both --horizon 1

# Dry run (no file written):
python -m phase9a.phase9a_runner --db-url "$SUPABASE_DB_URL" --asset BTC/USDT --dry-run
```

`--asset` is `BTC/USDT`, `ETH/USDT`, or `both` (default). `--horizon` is `1`,
`2`, or `3` (default `1`, which is the canonical gate horizon). The DB URL can
also be supplied via the `SUPABASE_DB_URL` environment variable. Results are
saved to `phase9a/PHASE9A_RESULTS.md` unless `--dry-run` is passed.

The pure analysis core (`run_analysis`) is fully decoupled from I/O, so the
whole experiment can be exercised with synthetic data — see
`phase9a/tests/test_phase9a_runner.py`.

## What the output means

```
===========================
PHASE 9A RESULTS
===========================
Asset: both (BTC/USDT, ETH/USDT)
Breakout events: 184
Horizon: t+1

Hit rate (overall): 61.2%      ← % of breakouts whose direction matched the next bar
Hit rate (BTC): 60.1%
Hit rate (ETH): 62.3%
Hit rate (high regime): 63.0%  ← only meaningful when regimes are populated
Hit rate (low regime): 58.0%

DM statistic: -2.41            ← negative = signal beats the coin-flip baseline
p-value: 8.0e-03               ← one-sided; < 0.05 ⇒ statistically significant

Temporal stability:
  Older: 59.0%
  Middle: 61.0%
  Recent: 63.0%

Gate results:
  G1 (hit rate > 55%): PASS
  G2 (DM p < 0.05):    PASS
  G3 (both assets):    PASS
  G4 (stable windows): PASS
  G5 (no degradation): PASS
  G6 (n >= 30):        PASS

VERDICT: SIGNAL FOUND
```

A **negative DM statistic** with a **small p-value** means the breakout
direction is a *statistically significant* improvement over a coin flip. The
verdict is `SIGNAL FOUND` only when **all six gates** pass; otherwise it is
`CLOSED`, and the closed experiment should be documented, not retried.

## Leakage discipline

* Direction uses **only** the breakout bar's own `open`/`close`.
* Forward returns use bars **strictly after** the breakout bar.
* When the `t+N` bar does not exist, that horizon is **skipped** (no row, no
  forward-fill).
* Breakout timestamps (ISO8601 bar open time) are matched exactly against
  candle open times; unmatched events are dropped.

## Tests

```bash
python -m pytest phase9a/ -q
```

All tests use synthetic data and a fake exchange — no DB, no network.
