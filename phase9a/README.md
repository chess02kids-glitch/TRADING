# Phase 9A — Breakout-Direction Continuation Analysis

A **pure statistical** test of one pre-registered hypothesis, with no
machine-learning models and no live trading. It answers the one question the
existing HAR system leaves open: *HAR predicts **how much** price will move —
does the breakout bar's candle direction also tell us **which way**?*

## The hypothesis (pre-registered — do not change)

> When `actual_range > 2 × har_predicted_range` (a **breakout event**), the
> breakout bar's candle direction (`close ≥ open` → **UP / +1**, else
> **DOWN / −1**) persists into the next 1, 2, and 3 bars.

This module is **DB-free**: it receives a `pd.DataFrame` of forward-return
data produced by `kronos_trading.alerts.forward_return_logger.get_phase9a_data`
(typically exported to CSV), and tests whether the breakout direction predicts
the sign of the forward return.

## Modules

| File | Responsibility |
|------|----------------|
| `direction_calculator.py` | `compute_hit_rate(df, horizon)` (overall / per-asset / per-direction) and `split_temporal_windows(df)` (older/middle/recent thirds). |
| `continuation_tester.py` | `compute_temporal_stability(df, horizon)` and `run_all_gate_checks(df, horizon)` → G1–G6 + verdict. |
| `dm_test.py` | One-sided Diebold-Mariano test vs a 50/50 coin flip (Newey-West HAC, 3 lags). |
| `phase9a_runner.py` | CLI: load a CSV → analyze → print the results box → optionally save markdown. |

## Input DataFrame columns

`breakout_timestamp, asset, breakout_direction, horizon, target_timestamp,
forward_return, forward_direction, breakout_close_price` — exactly what
`forward_return_logger.get_phase9a_data()` returns.

## The gates (all must pass → `SIGNAL FOUND`; any fail → `CLOSED`)

| Gate | Rule |
|------|------|
| **G1** | Hit rate > 55% overall **and** on both BTC + ETH. |
| **G2** | One-sided DM test **p < 0.05** vs a coin flip. |
| **G3** | Hit rate > 50% on **both** BTC and ETH. |
| **G4** | All three chronological thirds > 50% (temporal stability). |
| **G5** | Not degrading (recent not > 0.10 below older). |
| **G6** | At least **30** events per asset. |

## How to run

```bash
# Agent 2 exports the DB table to CSV, then:
python -m phase9a.phase9a_runner --data-file phase9a_data.csv \
    --asset both --horizon 1 --output phase9a/PHASE9A_RESULTS.md
```

`--asset` is `BTC/USDT`, `ETH/USDT`, or `both` (default). `--horizon` is `1`,
`2`, or `3` (default `1`, the primary gate horizon). If fewer than 30 events
per asset are available it prints `"Not enough data yet"` and exits cleanly.

## DM test convention

`d_t = loss_random − loss_signal` with `loss_random = 0.5` (expected coin-flip
loss). A **positive DM statistic** with a **small one-sided p-value** means the
signal is a statistically significant improvement over a coin flip.

## Tests

```bash
python -m pytest phase9a/ -q
```

All tests use synthetic DataFrames — no DB, no network.
