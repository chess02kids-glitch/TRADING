# Classical Volatility Benchmark — Methodology (Phase 5c)

**Date:** 2026-08-14 · **Upstream:** Kronos pinned `67b630e67f6a`

## Question

> Is the crypto candle-range forecasting problem itself predictable, and does
> the classical HAR model already capture that structure? Does Kronos add
> anything beyond the best classical baseline?

Kronos is treated as a **challenger**, not the foundation. The frozen Phase 4 /
Phase 5 / Phase 5b reports are unchanged.

## Classical baselines (all past-only, fixed a priori)

| ID | Definition | Fixed constant |
|---|---|---|
| previous-range | `range_{t-1}` | — |
| rolling-5 | `mean(range_{t-5:t})` | window 5 |
| rolling-22 | `mean(range_{t-22:t})` | window 22 |
| EWMA | `e_t = α·range_t + (1−α)·e_{t-1}`, seeded on first range | `span=22`, `α=2/23` |
| **HAR** (primary) | `β0 + β1·range_{t-1} + β2·mean5 + β3·mean22` | expanding past-only OLS, refit per step, min 24 rows |

Every baseline uses only closed candles strictly before the prediction
timestamp. HAR coefficients are re-estimated at each step from the expanding
past-only window (no test-window fitting).

## Targets (never mixed)

- **Raw range**: `high_future − low_future`.
- **Normalized range**: `(high_future − low_future) / close_current`
  (scale-invariant; denominator is the past close only).

## Windows / series

Fixed chronological `older / middle / recent` on BTC/USDT & ETH/USDT at
1h / 4h / 1d. Primary decision uses 1h + 4h only (12 windows); daily
(~73 samples) is low-power supplementary and excluded from the gate. Windows
with < 30 samples are `low_power`.

## Metrics (per model/window)

MAE, RMSE, normalized MAE/RMSE, Pearson, Spearman, bias
`mean(pred − actual)`, std of predictions and actuals, dispersion ratio
`std(pred)/std(actual)`, bias ratio `mean(pred)/mean(actual)`, and
improvement % vs previous-range.

## Statistics (paired on identical timestamps)

Primary comparisons: **HAR vs previous-range, EWMA, rolling-5, rolling-22**.
- Diebold-Mariano (two-sided) with Newey-West HAC variance — appropriate for
  serially correlated forecast errors.
- Circular block bootstrap 95% CI on paired error differences.
- Wilcoxon signed-rank (nonparametric robustness).
- Multiple testing: 4 primary comparisons → **Bonferroni α = 0.0125**.
- Pooled statistics are reported as **supplementary only**; per-window results
  are primary.

## Shrinkage / adequacy

HAR adequacy is tested directly: group rows by the past-only volatility regime
(terciles of a rolling-22 mean range), and check that HAR's per-regime forecast
means are monotone in the per-regime actual means **and** that HAR's forecast
spread reaches ≥ 10% of the actual spread (a pure shrinker has ~zero spread and
fails). Dispersion and bias ratios are reported alongside.

## Pre-registered decision (classical predictability)

| # | Criterion |
|---|---|
| 1 | HAR beats previous-range (raw MAE) in > half of eligible windows |
| 2 | HAR beats EWMA (raw MAE) in > half of eligible windows |
| 3 | HAR beats previous-range on ≥ 2 of 4 series |
| 4 | HAR beats previous-range in ≥ 2 of 3 windows |
| 5 | HAR beats previous-range on normalized MAE in > half of windows |
| 6 | pooled DM (HAR vs previous-range) p < 0.0125 (Bonferroni) |
| 7 | HAR improvement not purely shrinkage (regime tracking in > half of windows) |
| 8 | HAR beats previous-range across > 1 regime |

- **A** — all 8 hold → *classical volatility predictability established*.
- **B** — c1 holds but ≥ 1 other fails → *weak / ambiguous*.
- **C** — c1 fails → *no robust classical volatility predictability*.

## Kronos incremental-value question (separate from A/B/C)

Kronos "adds incremental value" beyond HAR **only if** it wins a majority of
eligible windows on range MAE **and** the pooled DM (normalized errors)
significantly favors Kronos. A marginal MAE edge without statistical support
is not counted as incremental value.

## Out-of-sample purity

- Coefficients estimated only from data strictly before each forecast.
- No evaluation-window fitting; no future normalization; no future regime
  thresholds; no model selection on test windows.
- No tuning of EWMA span, HAR windows, or regime thresholds.
