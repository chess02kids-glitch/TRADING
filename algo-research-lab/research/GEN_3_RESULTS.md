# Generation 3 Results

## Overview
- Total hypotheses tested: 50
- Survivors: 0
- Seed used: 777777 (logged in every Supabase/SQLite row)
- Mode: focused fresh batch (0 Gen-2 survivors) + ONE documented gate loosening
- Gate loosening (documented): screening.min_total_trades: 50 -> 30 (pre-registered fallback; smallest loosening that does not weaken the profitability bar)

## By Signal Type
| Signal Type | Tested | Survived | Best Sharpe (any gate) | Best Profit Factor |
|---|---|---|---|---|
| funding_rate_contrarian | 12 | 0 | 0.69 | 1.87 |
| multi_asset_momentum | 16 | 0 | 0.73 | 1.11 |
| spread_zscore | 22 | 0 | 1.10 | 1.59 |

## Primary Failure Mode Analysis
First-failing-gate attribution (gates run strictly in order; later gates skipped after failure):
- SCREENING: 32/50 (64%) of strategies
- WALK_FORWARD: 12/50 (24%) of strategies
- CONCENTRATION: 5/50 (10%) of strategies
- ROBUSTNESS: 0/50 (0%) of strategies
- PARAMETER_STABILITY: 0/50 (0%) of strategies
- CRASH: 1/50 (2%) of strategies

Failure reason detail:
- LOW_PROFIT_FACTOR: 30
- FAILED_OOS_CONSISTENCY: 12
- HIGH_CONCENTRATION: 5
- LOW_TRADE_COUNT: 2
- EXCEPTION: 1

Per-signal-type dominant failure:
| Signal Type | Dominant failure (count) |
|---|---|
| funding_rate_contrarian | FAILED_OOS_CONSISTENCY (5/12) |
| multi_asset_momentum | LOW_PROFIT_FACTOR (14/16) |
| spread_zscore | LOW_PROFIT_FACTOR (13/22) |

## Top 5 Genomes (by Sharpe, even if failed)
### 1. `82fd85eeb1b3` — Sharpe 1.10
- Genome: `{"signal_type": "spread_zscore", "asset_a": "BTC/USDT", "asset_b": "ETH/USDT", "zscore_window": 272, "entry_zscore": 1.134194652590513, "exit_zscore": 0.40086605547717025, "size_pct": 0.8804205490381921, "direction": "momentum"}`
- Gate failed: CONCENTRATION (HIGH_CONCENTRATION)
- Sharpe 1.10 | PF 1.59 | trades 117 | maxDD 14.7% | return 51.3%
- OOS Sharpe 0.89 (2/3 positive splits)
- Concentration: single 0.22636963624269152, top5 0.9947018211178896

### 2. `430bb349e945` — Sharpe 0.93
- Genome: `{"signal_type": "spread_zscore", "asset_a": "BTC/USDT", "asset_b": "ETH/USDT", "zscore_window": 65, "entry_zscore": 1.902395093865095, "exit_zscore": 0.004552859102896378, "size_pct": 0.568281720780358, "direction": "momentum"}`
- Gate failed: WALK_FORWARD (FAILED_OOS_CONSISTENCY)
- Sharpe 0.93 | PF 1.42 | trades 159 | maxDD 13.9% | return 29.8%
- OOS Sharpe -0.76 (1/3 positive splits)

### 3. `2e2f9aacf1ea` — Sharpe 0.73
- Genome: `{"signal_type": "multi_asset_momentum", "primary_asset": "ETH/USDT", "lookback_bars": 158, "momentum_threshold": 0.016305710225435546, "require_confirmation": true, "holding_bars": 36, "size_pct": 0.8177720812772143}`
- Gate failed: WALK_FORWARD (FAILED_OOS_CONSISTENCY)
- Sharpe 0.73 | PF 1.07 | trades 525 | maxDD 51.8% | return 75.2%
- OOS Sharpe 0.46 (1/3 positive splits)

### 4. `c3b9d692e426` — Sharpe 0.72
- Genome: `{"signal_type": "spread_zscore", "asset_a": "BTC/USDT", "asset_b": "ETH/USDT", "zscore_window": 180, "entry_zscore": 2.315287509201891, "exit_zscore": 0.23773544701340465, "size_pct": 0.5375246205930362, "direction": "momentum"}`
- Gate failed: WALK_FORWARD (FAILED_OOS_CONSISTENCY)
- Sharpe 0.72 | PF 1.47 | trades 82 | maxDD 10.0% | return 17.4%
- OOS Sharpe -0.15 (2/3 positive splits)

### 5. `12011867ef9b` — Sharpe 0.71
- Genome: `{"signal_type": "spread_zscore", "asset_a": "BTC/USDT", "asset_b": "ETH/USDT", "zscore_window": 335, "entry_zscore": 1.4083860167147026, "exit_zscore": 0.8384358253138966, "size_pct": 0.49582930910409717, "direction": "momentum"}`
- Gate failed: CONCENTRATION (HIGH_CONCENTRATION)
- Sharpe 0.71 | PF 1.37 | trades 116 | maxDD 9.1% | return 15.4%
- OOS Sharpe 0.01 (2/3 positive splits)
- Concentration: single 0.39781585307106515, top5 1.4679969924347176

## Survivors
None. All strategies were rejected by at least one pre-registered gate.
## What Worked / What Didn't

**Worked:**
- With min_trades relaxed to 30, `funding_rate_contrarian` genomes at Sharpe 0.56-0.69 with PF 1.39-1.87
  and 32-179 trades reached Gate 2 - trade-count was genuinely the binding constraint for this family.
- `spread_zscore` again placed multiple PF 1.2-1.6 genomes, ALL momentum-direction (final tally across
  all generations: momentum 21/28 pass PF>=1.05, median 1.37, max 2.18; mean_revert 0/30, median 0.32).
  OOS consistency (6/22) and concentration (3/22) alternate as its killers - the edge is regime-local.
- One funding_contrarian genome reached Gate 3 (09012f00b525, Sharpe 0.31, PF 1.09, 268 trades) before
  failing concentration.

**Didn't:**
- 0/50 survivors even with the loosened trade bar: no strategy simultaneously cleared profitability,
  OOS consistency, concentration, cost-stress and parameter-stability.
- One genome (96c8dfe08b9a) crashed in vectorbt: a residual fractional short position after the
  2021-05-19 crash triggered the percent-size reversal guard. Fixed by `upon_opposite_entry="Ignore"`
  (state machine already converts opposite entries to exits). Rerun after the fix: 42 trades,
  PF 0.486 - an honest non-survivor; result recorded in gen3_crash_rerun.json.
- multi_asset_momentum remains PF ~1.00-1.02 at scale: no economic edge after costs.

## Recommendation for Next Generation

Three generations (180 genomes) say the marginal search space is exhausted on this data. Gen 4 should
change the OBJECTIVE, not the thresholds:
1. Attack `spread_zscore`'s concentration failure directly with profit-capped/scaling-out exits
   (partial profits at z=+0.5 and 0) to reduce single-episode dominance - but keep the MOMENTUM
   direction and long windows: across all 3 generations momentum is 21/28 PF-passers (median 1.37)
   while mean_revert is 0/30 (median 0.32). BTC/ETH ratio regimes (ETH-season vs BTC-season) persist;
   fading them is a structural loser after 2-leg costs.
2. Cross-exchange funding spread (Binance vs Bybit/Gate funding CSVs already downloaded) - a genuinely
   untested, economically-motivated variant of the two best-performing funding families.
3. Stability-first mutation around `dbf438564958` (the one Gates-1-4 passer): perturb-and-SELECT
   neighbours that keep Sharpe under perturbation, rather than random mutation.
4. Retire on this data: funding_trend, har_regime_sized, vol_regime_breakout (pre-registered space).