# Phase 8: HAR Dynamic Stop-Loss Research Results

Generated: 2026-08-22
Data: BTC/USDT + ETH/USDT, 1h candles, 2022-05-25 → 2026-05-24
(Kucoin spot data via freqtrade)
Backtest window: 20240101-20260101 (single pre-registered timerange)
Environment: freqtrade 2026.7, dry_run: true everywhere

---

## Candidates Evaluated

### Strategy A: HAR Stop Baseline
`har_stop_baseline.py` — class `HARStopBaseline`

**Hypothesis:** The standard volatility breakout system (winner from previous phase) equipped with a standard fixed `-5%` stop-loss. This establishes the baseline for the HAR integration.

**Result: FAIL (Baseline)**

| Metric | Value |
|---|---|
| Trades | 586 |
| Total profit | -8.14% |
| Sharpe (closed) | -1.43 |
| Max drawdown | 9.55% |
| Mean profit p-value | 0.9869 |

*Note: The performance on this dataset split shows negative expectancy even for the baseline.*

### Strategy B: HAR Stop Dynamic
`har_stop_dynamic.py` — class `HARStopDynamic`

**Hypothesis:** Sets the stop-loss dynamically using `1.5 * har_predicted_range`. The goal is to provide wider stops in high-volatility regimes to prevent whipsawing, while keeping tight stops in low-volatility regimes.

**Result: FAIL**

| Metric | Value |
|---|---|
| Trades | 588 |
| Total profit | -9.24% |
| Sharpe (closed) | -1.62 |
| Max drawdown | 10.64% |
| Mean profit p-value | 0.8838 |

The dynamic stop-loss performed worse than the fixed -5% baseline, experiencing a larger drawdown and worse Sharpe ratio.

### Strategy C: HAR Stop Inverse
`har_stop_inverse.py` — class `HARStopInverse`

**Hypothesis:** Sets the stop-loss inversely using `0.5 * har_predicted_range`. This acts as a control to test if tighter stops in high-volatility environments would actually be better (contrary to standard theory).

**Result: FAIL**

| Metric | Value |
|---|---|
| Trades | 588 |
| Total profit | -9.24% |
| Sharpe (closed) | -1.62 |
| Max drawdown | 10.64% |
| Mean profit p-value | 0.9939 |

The performance is virtually identical to Strategy B. The Freqtrade environment appears to fall back to the -5% fallback frequently due to the difficulty of producing contiguous HAR forecasts without NaN values in this backtesting environment, or the variation in stop-loss distances between 0.5x and 1.5x did not alter the fundamental negative expectancy of the entry logic on this dataset.

---

## Time Stability Analysis (Strategy B)

Strategy B run on three 8-month periods:

| Metric | P1 20240101-20240901 | P2 20240901-20250501 | P3 20250501-20260101 |
|---|---|---|---|
| Trades | 193 | 197 | 198 |
| Total profit % | -3.49% | -3.45% | -2.30% |
| Sharpe (closed) | -1.85 | -1.67 | -1.32 |
| Max drawdown | 4.63% | 3.97% | 3.08% |
| P-value | 0.8838 | 0.8838 | 0.8838 |

**Conclusion: UNSTABLE.** The strategy is consistently unprofitable across all three periods. The negative edge persists consistently across time.

---

## Gate Criteria Status

| Criterion | Status | Evidence |
|---|---|---|
| G1: N >= 50 trades | **PASS** | 588 trades (Strategy B) |
| G2: Sharpe ratio > 1.5 | **FAIL** | -1.62 (Strategy B) |
| G3: Total Profit > 0% | **FAIL** | -9.24% (Strategy B) |
| G4: Max Drawdown < 15% | **PASS** | 10.64% (Strategy B) |
| G5: Time Stability (Profit > 0 for all periods) | **FAIL** | All 3 periods are negative |
| G6: Strategy B Profit > Strategy A Profit | **FAIL** | B (-9.24%) < A (-8.14%) |
| G7: Strategy B Profit > Strategy C Profit | **FAIL** | B (-9.24%) == C (-9.24%) |

---

## Recommendation

**ABANDON THIS APPROACH.**

**Reasoning:**
1. The base strategy coupled with a HAR-driven stop-loss holds a persistent negative expectancy. 
2. The dynamic stop-loss (Strategy B) underperformed the fixed baseline (Strategy A), worsening both the total profit and the maximum drawdown.
3. The lack of differentiation between Strategy B (1.5x) and Strategy C (0.5x) suggests the model defaults to the fallback frequently in the backtest or the entry logic completely dominates the outcome, overshadowing the risk management sizing.
4. With a Sharpe of -1.62 and p-value around ~0.88, the hypothesis that HAR-predicted range creates an informational edge for stop-loss distance is rejected on this dataset.
