# Phase 8 — Derivatives Positioning vs Frozen HAR (precise specification)

**Date:** 2026-08-14 · **Branch:** `arena/01a00001-trading`

## Question / hypothesis

- H0: funding rate, open-interest change, and basis provide **no** incremental
  information for next-candle normalized range beyond frozen HAR.
- H1: they provide incremental information.

This is a test of **new information**, not a new architecture. A negative
result is a valid scientific result.

## Target

- Primary: normalized next-candle range `(high_{t+1} − low_{t+1}) / close_t`.
- Secondary: raw range `high_{t+1} − low_{t+1}`.

## Frozen baseline (never re-tuned)

Single-asset HAR: `β0 + β1·range_{t-1} + β2·mean5 + β3·mean22`, expanding
past-only OLS, min 24 observations, refit per step. Implementation
`volatility_baselines.har_forecast` (constants `HAR_MIN_TRAIN=24`,
`ROLLING_WINDOWS=(5,22)`, `EWMA_SPAN=22`).

## External features (exactly three, point-in-time)

1. `funding_t`: last **settled** Binance USD-M perpetual funding rate with
   `funding_time <= prediction_timestamp` (never the next funding interval).
2. `oi_chg22_t`: 22-bar log change of aggregate open interest,
   `log(OI_t / OI_{t-22})`, using the last OI snapshot with `timestamp <= t`.
3. `basis_t`: perpetual premium `mark_price / spot_index − 1` using the last
   premium-index snapshot with `timestamp <= t`.

## Model (linear extension only)

```
raw_range_t = β0 + β1·range_{t-1} + β2·mean5 + β3·mean22
            + γ1·funding + γ2·oi_chg22 + γ3·basis
```

Expanding past-only OLS, refit every step, minimum 24 training observations.
No ML, no hyperparameter search, no feature selection, no tuning.

## Data rules (strict)

- Point-in-time only: `max(external_info_timestamp) <= prediction_timestamp`.
- Never use the next funding interval; never forward-fill; never interpolate;
  no future revisions/restatements.
- Missing required derivative observation => skip that row.
- Preserve exact target timestamps between the extension and HAR.
- Leakage counter must equal 0.

## Data acquisition

Public Binance USD-M market-data endpoints only (no API key, no trading/order
endpoints, no credentials): `GET /fapi/v1/fundingRate` (settled funding),
`GET /futures/data/openInterestHist` (aggregate OI),
`GET /fapi/v1/premiumIndex` (mark/index → basis). Stored under
`data/derivatives/` (gitignored). If the required data is unavailable, the
real experiment cannot run (the hypothesis remains untested, not "failed").

## Evaluation windows

Exactly the frozen 12 primary windows: BTC/ETH × 1h/4h × older/middle/recent.
Daily is supplementary and excluded from the gate.

## Metrics

Primary: normalized MAE/RMSE. Secondary: raw MAE/RMSE, Pearson, Spearman,
bias, dispersion ratio, bias ratio, improvement-% vs HAR.

## Statistics (paired, identical timestamps)

Primary comparison A = derivatives extension, B = HAR.

- Diebold-Mariano, two-sided, Newey-West/HAC (negative mean loss diff ⇒ ext
  better).
- Circular block bootstrap 95% CI on paired error differences.
- Wilcoxon signed-rank (robustness).
- Single primary comparison → α = 0.05.
- Pooled statistics supplementary; per-window/per-asset primary.

## Success gate (pre-registered)

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
