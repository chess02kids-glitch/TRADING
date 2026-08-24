# Generation 1 Results

## Overview
- Total hypotheses tested: 80
- Survivors: 0
- Seed used: 20260824 (logged in every Supabase/SQLite row)
- Mode: fresh random (6 signal types, ~13-14 each, pre-registered ranges)

## By Signal Type
| Signal Type | Tested | Survived | Best Sharpe (any gate) | Best Profit Factor |
|---|---|---|---|---|
| funding_rate_contrarian | 14 | 0 | 0.63 | 1.73 |
| funding_trend | 13 | 0 | -0.72 | 0.80 |
| har_regime_sized | 13 | 0 | -2.99 | 0.79 |
| multi_asset_momentum | 13 | 0 | 0.52 | 1.07 |
| spread_zscore | 14 | 0 | 1.03 | 1.45 |
| vol_regime_breakout | 13 | 0 | -0.76 | 0.53 |

## Primary Failure Mode Analysis
First-failing-gate attribution (gates run strictly in order; later gates skipped after failure):
- SCREENING: 75/80 (94%) of strategies
- WALK_FORWARD: 4/80 (5%) of strategies
- CONCENTRATION: 1/80 (1%) of strategies
- ROBUSTNESS: 0/80 (0%) of strategies
- PARAMETER_STABILITY: 0/80 (0%) of strategies
- CRASH: 0/80 (0%) of strategies

Failure reason detail:
- LOW_PROFIT_FACTOR: 58
- ZERO_TRADES_BUG: 10
- LOW_TRADE_COUNT: 7
- FAILED_OOS_CONSISTENCY: 4
- HIGH_CONCENTRATION: 1

Per-signal-type dominant failure:
| Signal Type | Dominant failure (count) |
|---|---|
| funding_rate_contrarian | LOW_PROFIT_FACTOR (10/14) |
| funding_trend | LOW_PROFIT_FACTOR (13/13) |
| har_regime_sized | LOW_PROFIT_FACTOR (13/13) |
| multi_asset_momentum | LOW_PROFIT_FACTOR (12/13) |
| spread_zscore | LOW_PROFIT_FACTOR (9/14) |
| vol_regime_breakout | ZERO_TRADES_BUG (10/13) |

## Top 5 Genomes (by Sharpe, even if failed)
### 1. `29dcf8477fe3` — Sharpe 1.03
- Genome: `{"signal_type": "spread_zscore", "asset_a": "BTC/USDT", "asset_b": "ETH/USDT", "zscore_window": 108, "entry_zscore": 1.8897633646808962, "exit_zscore": 0.0897321386352074, "size_pct": 0.7791131607885973, "direction": "momentum"}`
- Gate failed: WALK_FORWARD (FAILED_OOS_CONSISTENCY)
- Sharpe 1.03 | PF 1.43 | trades 148 | maxDD 14.5% | return 43.2%
- OOS Sharpe -0.70 (1/3 positive splits)

### 2. `cef29269e7e2` — Sharpe 0.85
- Genome: `{"signal_type": "spread_zscore", "asset_a": "BTC/USDT", "asset_b": "ETH/USDT", "zscore_window": 165, "entry_zscore": 1.9531154401480648, "exit_zscore": 0.0671114002883394, "size_pct": 0.9083396992125413, "direction": "momentum"}`
- Gate failed: CONCENTRATION (HIGH_CONCENTRATION)
- Sharpe 0.85 | PF 1.45 | trades 101 | maxDD 18.8% | return 40.3%
- OOS Sharpe 0.06 (2/3 positive splits)
- Concentration: single 0.2503120658964364, top5 1.122659850207048

### 3. `63b8d96677e2` — Sharpe 0.79
- Genome: `{"signal_type": "spread_zscore", "asset_a": "BTC/USDT", "asset_b": "ETH/USDT", "zscore_window": 104, "entry_zscore": 1.9590918979715615, "exit_zscore": 0.15136459091123833, "size_pct": 0.27429333687986657, "direction": "momentum"}`
- Gate failed: WALK_FORWARD (FAILED_OOS_CONSISTENCY)
- Sharpe 0.79 | PF 1.34 | trades 157 | maxDD 6.7% | return 10.7%
- OOS Sharpe -1.07 (1/3 positive splits)

### 4. `1e275223cb40` — Sharpe 0.63
- Genome: `{"signal_type": "funding_rate_contrarian", "long_threshold": -0.0060943130554276526, "short_threshold": 0.04531258453590022, "holding_bars": 23, "size_pct": 0.5733198258310187, "exit_type": "funding_flip"}`
- Gate failed: SCREENING (LOW_PROFIT_FACTOR)
- Sharpe 0.63 | PF 0.95 | trades 115 | maxDD 119.9% | return -7.6%

### 5. `4f298c92ecb7` — Sharpe 0.52
- Genome: `{"signal_type": "multi_asset_momentum", "primary_asset": "BTC/USDT", "lookback_bars": 18, "momentum_threshold": 0.047222071588379946, "require_confirmation": true, "holding_bars": 14, "size_pct": 0.7716405837850278}`
- Gate failed: WALK_FORWARD (FAILED_OOS_CONSISTENCY)
- Sharpe 0.52 | PF 1.07 | trades 395 | maxDD 41.4% | return 34.0%
- OOS Sharpe -0.26 (2/3 positive splits)

## Survivors
None. All strategies were rejected by at least one pre-registered gate.
## What Worked / What Didn't

**Worked (engine-level):**
- All six NEW signal types compiled and produced honest, non-empty backtests (zero-trades guard fired
  10 times on over-restrictive `vol_regime_breakout` draws - that is the guard doing its job, not an engine bug;
  every simulation passed through the certified `size_type="percent"` path).
- `spread_zscore` produced the only 4 Gate-1 passers (Sharpe up to 1.03, PF 1.43) - and ALL of them are
  `direction: momentum`: the BTC/ETH log-ratio TRENDS at long z-windows (104-165h). Spread momentum
  (regime persistence in the ETH/BTC ratio) is a real in-sample tendency; mean-reversion on the same
  spread is consistently destroyed (median PF ~0.3).
- `funding_rate_contrarian` produced repeatable positive-expectancy configurations (PF up to 1.73)
  but extreme thresholds starve trade counts.

**Didn't:**
- `funding_trend` (following funding direction): 0/13 passed Gate 1. Following the crowded side of
  funding is consistently negative after costs - consistent with funding being a CONTRARIAN signal.
- `har_regime_sized`: 0/13. HAR-predicted range as a breakout/reversion yardstick on 1h bars churns
  (1,500-4,200 trades) and bleeds fees; the validated signal (volatility magnitude) does not
  translate into a directional edge this way.
- `vol_regime_breakout` as pre-registered: 10/13 produced zero trades (ATR expansion 1.2-2.5x AND
  breakout AND regime filter is over-restrictive on 1h BTC). Parameter space, not concept, is the issue.
- `multi_asset_momentum`: 1/13 reached Gate 2, failed OOS consistency; the rest die at PF ~0.9-1.0
  (momentum on 1h holds no edge after costs on 2017-19 BTC/ETH).

## Recommendation for Next Generation

0 Gen-1 survivors -> per plan, Gen 2 is a fresh focused batch (not mutation) on the three closest
types (`spread_zscore` 1.03, `funding_rate_contrarian` 0.63, `multi_asset_momentum` 0.52 best Sharpe)
with WIDER parameter exploration (z-windows to 336h, funding thresholds to +/-0.10%/0.003%,
lookbacks to 240h), LONGER holding periods (to 48h), and SIMPLER entries (confirmation off by
default). Try spread_zscore hardest, and bias toward the MOMENTUM direction and long z-windows,
because every profitable spread genome in Gen 1 was momentum-direction (mean-revert direction: 0
passers, median PF ~0.3) - the edge exists but fails OOS consistency or concentration, which wider
windows and intermediate z-entries may fix.