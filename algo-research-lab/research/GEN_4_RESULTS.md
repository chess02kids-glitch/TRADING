# Generation 4 Results — 2026-08-24

## Pre-Run Verification
| Check | Result |
|---|---|
| certify_engine.py | PASS (7/7) |
| size_type="percent" | all executable VectorBT calls corrected |
| `upon_opposite_entry="Ignore"` | all executable VectorBT calls corrected |
| zero_trades guard | PASS; first Gate 1 check |
| Data recency | 2024+ |
| dbf438 found | YES (SQLite) |
| Funding type | SYNTHETIC — Binance TLS endpoint unavailable |

## Data Used
| Asset | Rows | Start | End |
|---|---:|---|---|
| BTC/USDT 1h | 23,194 | 2024-01-01 | 2026-08-24 |
| ETH/USDT 1h | 23,194 | 2024-01-01 | 2026-08-24 |

## Target B — dbf438 Stability Fix
Fragile parameter (largest ±10% Sharpe drop): **lookback_bars**.

### Perturbation Table
| Parameter | -20% | -10% | Base | +10% | +20% |
|---|---:|---:|---:|---:|---:|
| lookback_bars | 0.04 | 0.09 | 0.05 | -0.08 | -0.05 |
| momentum_threshold | 0.00 | 0.13 | 0.05 | 0.07 | 0.22 |
| holding_bars | -0.21 | 0.01 | 0.05 | 0.10 | 0.25 |
| size_pct | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |

### Variant Results
| Multiplier | Value | Sharpe | Gate Failed |
|---:|---:|---:|---|
| 0.3× | 49 | -0.15 | SCREENING |
| 0.5× | 82 | -0.85 | SCREENING |
| 0.7× | 115 | 0.11 | SCREENING |
| 0.9× | 148 | 0.09 | SCREENING |
| 1× | 164 | 0.05 | SCREENING |
| 1.1× | 180 | -0.08 | SCREENING |
| 1.3× | 213 | -0.03 | SCREENING |
| 1.5× | 246 | 0.07 | SCREENING |
| 2× | 328 | 0.33 | WALK_FORWARD |
| 2.5× | 410 | -0.48 | SCREENING |

Variants passing Gate 5: 0/10.

## Target A — Spread Momentum v2

### Gate Attrition by Fix Set
| Fix | G1 | G2 | G3 | G4 | G5 | Survivors |
|---|---:|---:|---:|---:|---:|---:|
| A1_scale_out | 0 | 0 | 0 | 0 | 0 |
| A2_adaptive | 0 | 0 | 0 | 0 | 0 |
| A3_combined | 0 | 0 | 0 | 0 | 0 |
| A4_hedge_ratio | 2 | 2 | 0 | 0 | 0 |

### Key Questions Answered
**Q: Did scale-out exits fix concentration?**  **A: NO.** The runner used state-machine exits but the VectorBT single-leg representation did not produce separate partial close records; no A1 candidate reached Gate 3.

**Q: Did adaptive z-score fix OOS?**  **A: NO.** No A2 candidate cleared Gate 1, so none reached OOS validation.

**Q: Did spread momentum hold on 2024–26 data?**  **A: NO in this 40-genome targeted sample.** Only 2/40 spread candidates cleared screening and both failed concentration; the 2024–26 result does not replicate the prior 2017–23 profit-factor breadth.

### Top 5 Genomes by Sharpe
| Genome | Fix | Sharpe | PF | Trades | Gate |
|---|---|---:|---:|---:|---|
| `023fb8ebc9b3` | A4_hedge_ratio | 0.41 | 1.10 | 223 | CONCENTRATION |
| `f99a6226f5d6` | B_stability | 0.33 | 1.05 | 251 | WALK_FORWARD |
| `22030b9b0f11` | A4_hedge_ratio | 0.32 | 1.13 | 99 | CONCENTRATION |
| `ca2f97a1c7c8` | A1_scale_out | 0.27 | 1.01 | 64 | SCREENING |
| `35571813ef8e` | A1_scale_out | 0.22 | 0.99 | 67 | SCREENING |

## Survivors
No survivors this generation.

## Overall Generation 4
| Total genomes tested | 50 |
| Survivors | 0 |
| Primary gate killer | SCREENING (47/50) |
| Highest Sharpe reached | 0.41 (`023fb8ebc9b3`) |

## Structural Analysis
The primary result is a data-regime mismatch: the prior spread momentum edge was not reproduced after realistic 2024–26 costs in this representation. Target B also did not stabilize; no lookback variant reached Gate 5. A correct Generation 5 should first implement true two-leg ETH/BTC PnL and true partial-close accounting, then re-test only after validating those mechanics against hand-calculated fixtures.

## Recommendation for Generation 5
Do not sample another random batch. Validate a two-leg portfolio engine with explicit ETH and BTC legs, funding/cost attribution, and partial exits. Only then rerun a small pre-registered spread test.
