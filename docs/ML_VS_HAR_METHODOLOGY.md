# ML vs HAR Volatility — Methodology (Phase 6)

**Date:** 2026-08-14

## Question

> Can a simple supervised ML model extract incremental volatility information
> from OHLCV beyond the classical HAR model already established as champion?

- H0: HAR captures essentially all useful OHLCV-only volatility structure.
- H1: nonlinear interactions in past OHLCV contain incremental information
  beyond HAR that a supervised ML model can exploit.

Kronos is a retired historical challenger and is **not** used in this
experiment. The goal is not to find a complex model that wins — it is to
determine whether nonlinear ML provides information HAR does not already
contain.

## Target (frozen, unchanged from the classical benchmark)

- **Primary:** normalized next-candle range `(high_{t+1} − low_{t+1}) / close_t`.
- **Secondary:** raw range `high_{t+1} − low_{t+1}`.

## Features (33, all past-only, fixed a priori)

Returns: `ret_1/2/3/5/10/22`, `abs_ret_1/5/22`, `sq_ret_1/22`.
Normalized range (`nr_k = (high_k − low_k)/close_{k−1}`): `nr_1`,
`nr_mean_5/22`, `nr_std_5/22`. Raw range: `range_1`, `range_mean_5/22`,
`range_std_5/22`. Realized vol: `rv_5/22`. Parkinson: `park_5/22`. Volume:
`vol_ret`, `vol_z22`, `logvol_mean22`, `vol_cv22`. Structure:
`dist_ma22`, `pos_range22`. Time (prediction candle open, known at prediction
time): `hour`, `dow`.

Every feature at row `j` uses only candles with index `< j`, enforced by
`.shift(1)` and left-aligned rolling windows — no look-ahead by construction.

## Model (one primary model, fixed configuration)

- **LightGBM** (XGBoost fallback), CPU, deterministic seeds.
- `n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
  random_state=42, n_jobs=1` (XGBoost fallback: `max_depth=5`, same learning
  rate/subsampling/regularization).
- No hyperparameter search; the configuration is fixed before OOS evaluation.

## Walk-forward (strictly causal, expanding window)

- Feature matrix built once per series (rows aligned to the target candle).
- At each prediction index `j`, the model is trained only on rows with target
  index `< j` (expanding window).
- Retrain cadence fixed at **every 100 predictions**; between retrains the
  model is reused (training data always ends strictly before the prediction,
  so the cadence cannot introduce leakage).
- The walk tracks the maximum training index per prediction and reports a
  `leaks` counter (must be 0).

## Windows / timestamps

Identical to the frozen classical benchmark: `older/middle/recent` over the
same target indices, reusing `PredictionEvaluator.evaluate_windows` with a
stub predictor (so baselines, regimes and timestamps are byte-identical to the
classical run). Primary decision = the 12 non-daily (1h/4h) windows; daily is
low-power supplementary and excluded from the gate.

## Baselines

previous-range, rolling-5, rolling-22, EWMA (span=22), **HAR** (the champion).
The key comparison is **ML vs HAR**.

## Metrics

Primary: normalized-range MAE / RMSE. Secondary: raw-range MAE / RMSE, Pearson
and Spearman correlation, mean bias, prediction std, actual std, dispersion
ratio `std(pred)/std(actual)`, bias ratio, and `improvement_vs_HAR_pct` for
MAE/RMSE (positive = ML better).

## Statistics (paired on identical timestamps)

- Diebold-Mariano (two-sided) with Newey-West HAC variance (`a_name='ml',
  b_name='har'` — no stale Kronos labels).
- Circular block bootstrap 95% CI on paired error differences.
- Wilcoxon signed-rank (robustness).
- Single primary comparison (ML vs HAR) → α = 0.05; the four secondary
  ML-vs-baseline DMs are reported without gate weight (Bonferroni α=0.0125
  noted).
- Pooled statistics are supplementary; per-window results are primary.

## Regime analysis

Same predetermined past-only tercile regimes (rolling-22 mean range). Report
ML vs HAR per regime; do not select regimes after seeing results.

## Overfitting checks (reported, not tuned)

- Train vs OOS performance (expanding-window structure forces genuine OOS).
- Performance by chronological window (gate c7).
- Feature-importance mean/std across retrains (stability).
- Prediction dispersion ratio (shrinkage diagnostic) + regime tracking.
- `leaks` counter (look-ahead guard).

## Pre-registered success gate (ML PASS only if ALL hold)

| # | Criterion |
|---|---|
| 1 | ML beats HAR on normalized MAE in ≥2/3 windows for ≥2/4 series |
| 2 | ML beats HAR on normalized RMSE in ≥2/3 windows for ≥2/4 series |
| 3 | pooled DM (ML vs HAR, normalized) p < 0.05 and ML wins |
| 4 | improvement survives raw-range MAE (majority of windows) |
| 5 | improvement survives regime analysis (≥2/3 regimes) |
| 6 | improvement not explained purely by shrinkage (regime tracking) |
| 7 | effect present in ≥2/3 windows (not one isolated period) |
| 8 | no look-ahead / data leakage |

## Decision tree

- **A — ML adds genuine incremental value**: all 8 hold → next experiment may
  add richer data (cross-asset, funding, on-chain, sentiment/macro).
- **B — weak/ambiguous**: ML occasionally beats HAR but lacks robust
  significance/stability → HAR remains the better research model; at most one
  tightly justified follow-up.
- **C — ML fails**: HAR already captures the OHLCV-only structure → stop adding
  model complexity to this branch (a legitimate final conclusion).

## Out-of-sample purity

- Coefficients/features use only data strictly before each forecast.
- No evaluation-window fitting, no future scaling, no future regime
  thresholds, no model selection on test windows, no K-fold CV.
