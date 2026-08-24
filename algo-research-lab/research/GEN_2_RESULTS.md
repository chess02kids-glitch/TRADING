# Generation 2 Results

## Overview
- Total hypotheses tested: 50
- Survivors: 0
- Seed used: 424242 (logged in every Supabase/SQLite row)
- Mode: focused fresh batch (0 Gen-1 survivors): top-3 closest types by best Sharpe, wider parameter ranges, longer holds, simpler entries

## By Signal Type
| Signal Type | Tested | Survived | Best Sharpe (any gate) | Best Profit Factor |
|---|---|---|---|---|
| funding_rate_contrarian | 17 | 0 | 0.70 | 1.45 |
| multi_asset_momentum | 11 | 0 | 0.95 | 1.20 |
| spread_zscore | 22 | 0 | 1.55 | 2.18 |

## Primary Failure Mode Analysis
First-failing-gate attribution (gates run strictly in order; later gates skipped after failure):
- SCREENING: 37/50 (74%) of strategies
- WALK_FORWARD: 5/50 (10%) of strategies
- CONCENTRATION: 7/50 (14%) of strategies
- ROBUSTNESS: 0/50 (0%) of strategies
- PARAMETER_STABILITY: 1/50 (2%) of strategies
- CRASH: 0/50 (0%) of strategies

Failure reason detail:
- LOW_PROFIT_FACTOR: 27
- LOW_TRADE_COUNT: 10
- HIGH_CONCENTRATION: 7
- FAILED_OOS_CONSISTENCY: 5
- FRAGILE: 1

Per-signal-type dominant failure:
| Signal Type | Dominant failure (count) |
|---|---|
| funding_rate_contrarian | LOW_TRADE_COUNT (9/17) |
| multi_asset_momentum | LOW_PROFIT_FACTOR (9/11) |
| spread_zscore | LOW_PROFIT_FACTOR (13/22) |

## Top 5 Genomes (by Sharpe, even if failed)
### 1. `0a2d8d986dd7` — Sharpe 1.55
- Genome: `{"signal_type": "spread_zscore", "asset_a": "BTC/USDT", "asset_b": "ETH/USDT", "zscore_window": 238, "entry_zscore": 1.1023769843736404, "exit_zscore": 0.11497095978170002, "size_pct": 0.5543630439614519, "direction": "momentum"}`
- Gate failed: CONCENTRATION (HIGH_CONCENTRATION)
- Sharpe 1.55 | PF 2.09 | trades 95 | maxDD 11.3% | return 52.1%
- OOS Sharpe 0.79 (2/3 positive splits)
- Concentration: single 0.21445246727693767, top5 0.7252916202729616

### 2. `e71cf1fb2fb0` — Sharpe 1.27
- Genome: `{"signal_type": "spread_zscore", "asset_a": "BTC/USDT", "asset_b": "ETH/USDT", "zscore_window": 261, "entry_zscore": 1.710161986657705, "exit_zscore": 0.09156010889011357, "size_pct": 0.3297000063999865, "direction": "momentum"}`
- Gate failed: CONCENTRATION (HIGH_CONCENTRATION)
- Sharpe 1.27 | PF 2.18 | trades 70 | maxDD 6.7% | return 21.6%
- OOS Sharpe 0.78 (2/3 positive splits)
- Concentration: single 0.1608923940536332, top5 0.7572074123584426

### 3. `8660b51a0a56` — Sharpe 1.19
- Genome: `{"signal_type": "spread_zscore", "asset_a": "BTC/USDT", "asset_b": "ETH/USDT", "zscore_window": 311, "entry_zscore": 1.2460302435937505, "exit_zscore": 0.23305514979596054, "size_pct": 0.5805558212919131, "direction": "momentum"}`
- Gate failed: CONCENTRATION (HIGH_CONCENTRATION)
- Sharpe 1.19 | PF 1.87 | trades 82 | maxDD 11.2% | return 36.7%
- OOS Sharpe 1.03 (2/3 positive splits)
- Concentration: single 0.18648501206228804, top5 0.7994362633720302

### 4. `679633d5978f` — Sharpe 1.12
- Genome: `{"signal_type": "spread_zscore", "asset_a": "BTC/USDT", "asset_b": "ETH/USDT", "zscore_window": 209, "entry_zscore": 1.4082723076010064, "exit_zscore": 0.10497974567154189, "size_pct": 0.5153005433664152, "direction": "momentum"}`
- Gate failed: CONCENTRATION (HIGH_CONCENTRATION)
- Sharpe 1.12 | PF 1.69 | trades 101 | maxDD 10.5% | return 31.7%
- OOS Sharpe 0.47 (2/3 positive splits)
- Concentration: single 0.2067252814446346, top5 0.8441916319418513

### 5. `0df6f315f890` — Sharpe 1.01
- Genome: `{"signal_type": "spread_zscore", "asset_a": "BTC/USDT", "asset_b": "ETH/USDT", "zscore_window": 253, "entry_zscore": 1.4746426097367473, "exit_zscore": 0.3812362368763349, "size_pct": 0.5246274722597228, "direction": "momentum"}`
- Gate failed: CONCENTRATION (HIGH_CONCENTRATION)
- Sharpe 1.01 | PF 1.71 | trades 94 | maxDD 9.8% | return 26.3%
- OOS Sharpe 0.68 (2/3 positive splits)
- Concentration: single 0.2108533344967241, top5 0.8918691278046507

## Survivors
None. All strategies were rejected by at least one pre-registered gate.
## What Worked / What Didn't

**Worked:**
- The wider exploration worked exactly as intended: `spread_zscore` best Sharpe rose 1.03 -> 1.55
  (PF 2.09), and 6 spread genomes passed Gates 1+2. ALL profitable spread genomes are
  `direction: momentum` at long z-windows (209-318h) - BTC/ETH ratio regimes persist and the spread
  trends; the family's problem is now clearly CONCENTRATION (profits dominated by <5 trending
  episodes), not profitability or OOS consistency.
- One `multi_asset_momentum` genome (`dbf438564958`, Sharpe 0.95, PF 1.20, 456 trades) passed
  Gates 1-4 and failed ONLY Gate 5 (FRAGILE) - the single closest genome to survival in the whole lab.
- `funding_rate_contrarian` at moderate thresholds (0.005-0.02%) yields PF 1.2-1.9 with positive Sharpe,
  but trade counts of 16-44 fail the 50-trade bar; with >=50 trades the PF decays toward 1.0.

**Didn't:**
- 0/50 survivors. The economic edge in every family is either too thin (PF ~1.0 at scale), too
  concentrated (spread), or too fragile (multi-asset momentum).
- Longer holds alone did not rescue momentum: high-churn variants still die at PF<1 (9/11).

## Recommendation for Next Generation

0 Gen-2 survivors -> per plan, Gen 3 documents the killer gate and loosens exactly ONE parameter by
the smallest amount. Counts across Gen 1+2: Gate 1 screening kills 102/130 (78%) - but its dominant
reason LOW_PROFIT_FACTOR (85/130, 65%) is the core economic bar and must NOT be loosened. The
pre-registered fallback `min_trades 50 -> 30` is the smallest defensible loosening: it rescues
2 historical near-misses (both funding contrarians with PF 1.45-1.73) without weakening any
profitability, consistency, robustness or stability requirement. Gen 3 runs 50 focused genomes
(spread_zscore, multi_asset_momentum, funding_rate_contrarian) with that single change.