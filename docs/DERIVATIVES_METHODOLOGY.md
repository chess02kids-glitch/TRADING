# Phase 8 (F-01) — Funding-Only Positioning vs Frozen HAR

**Date:** 2026-08-14 · **Branch:** `arena/01a00001-trading`

## Question / hypothesis

- H0: funding-rate information provides **no** incremental information for
  next-candle normalized range beyond frozen HAR.
- H1: funding-rate information provides incremental information.

This is a test of **new information**, not a new architecture. A negative
result is a valid scientific result. The experiment is **funding-only** —
no open interest, no basis/premium, no liquidations, no long/short ratios.

## Target

- Primary: normalized next-candle range `(high_{t+1} − low_{t+1}) / close_t`.
- Secondary: raw range `high_{t+1} − low_{t+1}`.

## Frozen baseline (never re-tuned)

Single-asset HAR: `β0 + β1·range_{t-1} + β2·mean5 + β3·mean22`, expanding
past-only OLS, min 24 observations, refit per step. Implementation
`volatility_baselines.har_forecast` (constants `HAR_MIN_TRAIN=24`,
`ROLLING_WINDOWS=(5,22)`, `EWMA_SPAN=22`).

## External features (exactly two, derived from settled funding)

At prediction timestamp `t` (the target candle's open time), using only
settled Binance USD-M perpetual funding rates with `funding_time ≤ t`:

1. `funding_mean_24h = mean of settled funding rates with funding_time in (t-24h, t]`
2. `abs_funding_mean_24h = mean of |funding_rate| over the same (t-24h, t] window`

## Model (linear extension only)

```
raw_range_t = β0 + β1·range_{t-1} + β2·mean5 + β3·mean22
            + γ1·funding_mean_24h + γ2·abs_funding_mean_24h
```

Expanding past-only OLS, refit every step, minimum 24 training observations.
No ML, no hyperparameter search, no feature selection, no tuning.

## Data rules (strict)

- Point-in-time only: `max(funding_time) <= prediction_timestamp`.
- Never use the next funding interval.
- Never forward-fill across a missing required observation: if the most recent
  settled rate is older than one 8h funding interval at `t`, skip that row.
- Missing funding in the 24h window => skip that row. No interpolation.
- No future revisions/restatements.
- Preserve exact target timestamps between the extension and HAR.
- Leakage counter must equal 0.

## Data acquisition

Public Binance USD-M market-data endpoint only (no API key, no trading/order
endpoints, no credentials): `GET /fapi/v1/fundingRate` (settled funding).
Stored under `data/derivatives/` (gitignored) via
`fetch_funding_only(symbol, start_ms)`.

**Pagination:** Binance returns at most 1000 rows per request. The fetcher
paginates chronologically — each page advances to `latest fundingTime + 1 ms`
— until the requested range is exhausted or the endpoint returns no rows, then
deduplicates by funding timestamp and sorts chronologically. No silent
truncation, no forward-fill, no synthesized observations. For ~730 days this
yields ~2190 settled 8h funding observations (not 1000). The open-interest and
premium-index fetchers remain as documented FUTURE utilities and are never
called by F-01.

## Evaluation windows

Exactly the frozen 12 primary windows: BTC/ETH × 1h/4h × older/middle/recent.
Daily is supplementary and excluded from the gate.

## Metrics

Primary: normalized MAE/RMSE. Secondary: raw MAE/RMSE, Pearson, Spearman,
bias, dispersion ratio, bias ratio, improvement-% vs HAR.

## Statistics (paired, identical timestamps)

Primary comparison A = funding extension, B = HAR.

- Diebold-Mariano, two-sided, Newey-West/HAC (negative mean loss diff ⇒ ext
  better).
- Circular block bootstrap 95% CI on paired error differences.
- Wilcoxon signed-rank (robustness).
- Single primary comparison → α = 0.05.
- Pooled statistics supplementary; per-window/per-asset primary.

## Success gate (pre-registered, unchanged)

| # | Criterion |
|---|---|
| C1 | ext beats HAR on normalized MAE in ≥2/3 windows for ≥1 asset |
| C2 | evidence exists in BOTH BTC and ETH (each ≥2/3 nMAE wins) |
| C3 | normalized RMSE has the same broad pattern (both assets ≥2/3) |
| C4 | improvement survives raw-range evaluation (> half windows raw MAE) |
| C5 | pooled DM p < 0.05 AND mean loss difference favors ext |
| C6 | improvement exists in ≥2/3 volatility regimes |
| C7 | no leakage (leaks == 0 everywhere) |

**PASS** = all 7 · **B** = C1 true but not all · **C** = C1 false.

## STOP rule

If B or C: close the derivatives/positioning family — no more derivatives
features, no liquidation heatmaps, no extra venues, no nonlinear ML, no
feature stacking, no tuning. If PASS: exactly one replication/holdout
experiment before any trading work.
