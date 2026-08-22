# NautilusTrader HAR Volatility Targeting Results

Generated: 2026-08-22 20:47 UTC
Research context: HAR validated (p<1e-26),
testing volatility targeting as portfolio
risk management technique.

## Research Question

Does using HAR predicted range for portfolio
position sizing (volatility targeting) improve
risk-adjusted returns vs equal-weight?

## Strategy Specifications

Strategy A: Equal Weight
  50% BTC, 50% ETH
  Rebalance daily (>5% drift)
  Baseline comparison

Strategy B: HAR Volatility Targeting
  allocation = target_vol / vol_estimate
  vol_estimate = HAR_range / price
  target_vol = 0.02
  Clip: [0.05, 1.0]

Strategy C: Inverse HAR (Control)
  allocation = vol_estimate / target_vol
  Opposite of Strategy B

Pre-registered parameters (not tuned):
  target_vol = 0.02
  min_allocation = 0.05
  max_allocation = 1.0
  rebalance_threshold = 0.05
  fees = 10 bps

## Results

| Metric | A (Equal Weight) | B (HAR Targeting) | C (Inverse HAR) |
|---|---|---|---|
| Total Return % | -0.13 | -0.11 | -0.16 |
| Annualized Return % | -0.07 | -0.05 | -0.08 |
| Sharpe Ratio | -0.001 | -0.001 | -0.001 |
| Sortino Ratio | -0.001 | -0.001 | -0.001 |
| Max Drawdown % | -12.89 | -10.27 | -17.87 |
| Annual Vol % | 78.58 | 71.67 | 88.74 |
| Calmar Ratio | -0.005 | -0.005 | -0.005 |
| p-value | 0.999 | 0.9991 | 0.999 |

## Time Stability (Strategy B)

| Period | Start | End | Total Return % | Sharpe | Max DD % |
|---|---|---|---|---|---|
| P1 | 2024-01-01 | 2024-09-01 | 0.19 | 0.004 | -8.01 |
| P2 | 2024-09-01 | 2025-05-01 | 0.07 | 0.001 | -11.94 |
| P3 | 2025-05-01 | 2026-01-01 | -0.15 | -0.003 | -10.04 |

## Gate Criteria

G1 (B Sharpe > A): FAIL
G2 (B DD < A):     PASS
G3 (B Vol < A):    PASS
G4 (C < A):        FAIL
G5 (Stability):    FAIL
G6 (p<0.10):       FAIL

Overall: FAIL

## Recommendation

Volatility targeting did not demonstrate significant improvement. Consider alternative approaches.

## Statistical Integrity

Parameters pre-registered.
Single timerange used.
No parameters changed after results.
Paper trading only. No real orders.
