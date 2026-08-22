# HAR Stop-Loss Multiplier Sweep Results

Generated: 2026-08-23 01:19 UTC
Research context: Day 3 of 30-day calibration (HAR beating persistence on both assets)

## Research Question

Does HAR predicted range as a dynamic stop-loss improve VolatilityBreakout performance when predictions are valid?

## Why This Differs From Freqtrade Experiment

Previous Freqtrade experiment failed because:
HAR predictions had too many NaN values per backtest window, causing fallback to fixed -5% stop. B = C because HAR was never actually applied.

This experiment computes HAR walk-forward on the FULL dataset, ensuring valid predictions for 95%+ of bars.

## HAR Prediction Quality

BTC/USDT 1h:
  Total bars: 17544
  Valid predictions: 17497
  NaN rate: 0.27%
  Passes threshold (< 5%): YES
  Mean predicted range: $552.74

ETH/USDT 1h:
  Total bars: 17544
  Valid predictions: 17497
  NaN rate: 0.27%
  Passes threshold (< 5%): YES
  Mean predicted range: $29.88

## BTC/USDT Results

| Multiplier | Trades | Win% | Avg% | Total% | Sharpe | MaxDD% |
|---|---|---|---|---|---|---|
| 0.00 (fixed -5%) | 380 | 33.9% | -0.06% | -29.56% | -0.42 | -45.46% |
| 0.50x HAR | 423 | 28.4% | -0.06% | -28.86% | -0.48 | -46.50% |
| 0.75x HAR | 420 | 29.0% | -0.07% | -31.09% | -0.52 | -47.32% |
| 1.00x HAR | 419 | 29.6% | -0.07% | -31.80% | -0.52 | -48.65% |
| 1.25x HAR | 416 | 30.0% | -0.08% | -35.20% | -0.60 | -51.31% |
| 1.50x HAR | 411 | 30.9% | -0.08% | -35.44% | -0.60 | -51.36% |
| 1.75x HAR | 404 | 31.9% | -0.07% | -30.60% | -0.47 | -47.47% |
| 2.00x HAR | 399 | 32.3% | -0.07% | -31.21% | -0.48 | -48.35% |
| 2.25x HAR | 398 | 32.4% | -0.09% | -37.27% | -0.63 | -52.90% |
| 2.50x HAR | 396 | 32.6% | -0.08% | -33.61% | -0.53 | -50.08% |
| 2.75x HAR | 393 | 33.1% | -0.07% | -32.72% | -0.50 | -49.41% |
| 3.00x HAR | 389 | 33.4% | -0.07% | -31.24% | -0.46 | -47.92% |


Best multiplier: 0.0x
Baseline Sharpe: -0.42
Best Sharpe: N/A

## ETH/USDT Results

| Multiplier | Trades | Win% | Avg% | Total% | Sharpe | MaxDD% |
|---|---|---|---|---|---|---|
| 0.00 (fixed -5%) | 336 | 33.0% | 0.08% | 7.65% | 0.29 | -55.28% |
| 0.50x HAR | 386 | 26.4% | -0.04% | -25.84% | -0.27 | -62.43% |
| 0.75x HAR | 378 | 27.5% | -0.01% | -19.18% | -0.13 | -59.01% |
| 1.00x HAR | 364 | 29.7% | 0.04% | -1.47% | 0.16 | -53.98% |
| 1.25x HAR | 359 | 30.1% | 0.02% | -11.45% | 0.02 | -57.59% |
| 1.50x HAR | 354 | 30.5% | 0.02% | -11.45% | 0.02 | -58.69% |
| 1.75x HAR | 352 | 31.0% | 0.04% | -5.47% | 0.11 | -57.67% |
| 2.00x HAR | 351 | 31.1% | 0.04% | -5.94% | 0.11 | -59.98% |
| 2.25x HAR | 348 | 31.6% | 0.04% | -5.04% | 0.12 | -59.68% |
| 2.50x HAR | 346 | 31.8% | 0.03% | -6.41% | 0.10 | -59.23% |
| 2.75x HAR | 342 | 32.2% | 0.09% | 12.44% | 0.34 | -56.70% |
| 3.00x HAR | 341 | 32.6% | 0.09% | 12.12% | 0.34 | -56.49% |


Best multiplier: 0.0x
Baseline Sharpe: 0.29
Best Sharpe: N/A

## Time Stability — Best Multiplier

### BTC/USDT

No data

### ETH/USDT

No data

## Statistical Significance

BTC best multiplier:
  p-value: N/A
  Interpretation: not significant

ETH best multiplier:
  p-value: N/A
  Interpretation: not significant

## Gate Criteria

G1 (Best Sharpe > Baseline): FAIL
G2 (Best DD < Baseline DD):  FAIL
G3 (Trades >= 30):           FAIL
G4 (Time stability):         FAIL
G5 (Both assets):            FAIL
G6 (p-value < 0.10):         FAIL

Overall: OPTION_C

## Recommendation

No multiplier produced positive risk-adjusted returns matching all gate criteria.
HAR stop-loss approach abandoned.
Recommend moving to:
  - Portfolio-level volatility targeting
  - Different entry strategy
  - Longer timeframe (4h/1d)

## Statistical Integrity Statement

Multipliers were pre-registered (0.5 to 3.0).
No multipliers changed after seeing results.
Single pre-registered timerange used.
No parameters tuned on test data.
Paper trading only. No real orders.
