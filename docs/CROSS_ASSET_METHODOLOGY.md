# Phase 7 — Cross-Asset Information Experiment (precise specification)

**Date:** 2026-08-14 · **Branch:** `arena/01a00001-trading`

## Question

Does information from the **other** crypto asset (BTC ↔ ETH) add predictive
information beyond the target asset's frozen single-asset HAR model, for the
normalized next-candle range target?

A negative result is a valid scientific result; this experiment is **not** an
attempt to make cross-asset features win.

## Target

- Primary: normalized next-candle range `(high_{t+1} − low_{t+1}) / close_t`.
- Secondary: raw range `high_{t+1} − low_{t+1}`.

## Baseline (frozen, never re-tuned)

Single-asset HAR: `β0 + β1·range_{t-1} + β2·mean(range_{t-5:t}) +
β3·mean(range_{t-22:t})`, expanding past-only OLS, min 24 observations, refit
per prediction step. Implementation: `volatility_baselines.har_forecast`
(constants `HAR_MIN_TRAIN=24`, `ROLLING_WINDOWS=(5,22)`, `EWMA_SPAN=22`).

## Augmented model (linear extension only)

Fitted by expanding past-only OLS on **raw range** (HAR's native units):

```
target_range_t = beta0
    + beta1 * target_range_{t-1}
    + beta2 * target_mean5
    + beta3 * target_mean22
    + gamma1 * cross_nr_1
    + gamma2 * cross_rv22
    + gamma3 * cross_ret1
    + gamma4 * cross_ret22
```

## Exact cross-asset features (other asset only — exactly four)

At information timestamp `t` (the other asset's last closed candle):

1. `cross_nr_1  = (high_other[t] − low_other[t]) / close_other[t]`
2. `cross_rv22  = std of the other asset's 22 log returns ending at t` (ddof=1)
3. `cross_ret1  = log(close_other[t] / close_other[t−1])`
4. `cross_ret22 = log(close_other[t] / close_other[t−22])`

All use information `<= t`; none use `t+1`.

## Temporal alignment (critical, enforced + tested)

For predicting the target asset's candle with open time `T`:

- Target-asset features use only target candles with open time `< T`.
- Cross-asset features use the other asset's candle with open time **exactly**
  `T − step` (the last other candle closed at or before `T`), verified by exact
  timestamp equality on the aligned Binance candle grid.
- If the cross candle is missing, the prediction is **skipped** (never
  forward-filled, never inferred from future candles).
- Enforced invariant: `max_information_timestamp <= prediction_timestamp`.

## Walk-forward procedure

Expanding window; at prediction step `j` the OLS coefficients are fitted on
rows with index strictly `< j` (minimum 24 training rows), refit every step.
A `leaks` counter (must be 0) records any violation of the training cutoff.

## Evaluation windows

Exactly the frozen 12 primary windows: BTC/ETH × 1h/4h × older/middle/recent.
Daily windows are supplementary and excluded from the gate.

## Metrics

Primary: normalized-range MAE / RMSE. Secondary: raw-range MAE / RMSE, Pearson,
Spearman, bias, dispersion ratio `std(pred)/std(actual)`, bias ratio, and
improvement % `100·(HAR_error − cross_error)/HAR_error` (positive = improvement).

## Statistics (paired on identical timestamps)

Primary comparison: cross-asset vs frozen HAR.

- Diebold-Mariano, two-sided, Newey-West/HAC variance, system A = `cross`,
  system B = `har` (negative mean loss diff ⇒ cross better).
- Circular block bootstrap 95% CI on paired error differences.
- Wilcoxon signed-rank (robustness).
- Single primary comparison → α = 0.05. Secondary comparisons (cross vs
  previous/EWMA/rolling5/rolling22) reported without gate weight.
- Pooled statistics are reported alongside per-window and per-asset summaries.

## Regime analysis

Reuse the frozen past-only tercile regimes (low/medium/high). Report whether
cross beats HAR per regime; regimes are never redefined from the results.

## Success gate (pre-registered)

| # | Criterion |
|---|---|
| C1 | cross beats HAR on normalized MAE in ≥2 of 3 windows for BTC AND/OR ETH |
| C2 | improvement not confined to one asset — both BTC **and** ETH show ≥2/3 normalized-MAE wins |
| C3 | cross beats HAR on normalized RMSE in broadly the same pattern (both assets ≥2/3) |
| C4 | improvement survives raw-range evaluation (cross beats HAR raw MAE in > half of primary windows) |
| C5 | pooled DM (cross vs HAR, normalized) p < 0.05 and the loss difference favors cross |
| C6 | benefit appears in ≥2 of 3 volatility regimes |
| C7 | no look-ahead / leakage (all alignment + leak tests pass) |

Verdict: **PASS** = all C1–C7 hold · **B** = C1 holds but not all · **C** = C1
fails (cross-asset does not broadly beat HAR).

## STOP condition

If the gate fails: close the cross-asset family. Do not add more cross features,
more indicators, nonlinear ML, more assets, or tuned coefficients. Document:

> BTC↔ETH cross-asset information did not demonstrate robust incremental value
> beyond frozen single-asset HAR.

and move to the next information class (derivatives: funding / open interest /
basis) — not implemented now.

## Implementation

- `kronos_trading/cross_asset.py` — feature builder + expanding-OLS walk +
  evaluator + gate.
- CLI: `python -m kronos_trading.cli cross-asset --db
  data\db\kronos_trading_verified.db --assets BTC/USDT ETH/USDT --timeframes
  1h 4h 1d --window-size 1000`.
- Report: `data/eval/cross_asset_volatility_report.json`.
- Tests: `tests/test_phase7_cross_asset.py` (alignment, no future info, no
  forward-fill, missing history, training cutoff, deterministic OLS, identical
  timestamps, leakage counter, gate A/B/C, frozen-HAR-unchanged).
