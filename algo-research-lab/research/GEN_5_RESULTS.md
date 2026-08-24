# Generation 5 Results — 2026-08-24

## Two-Leg Engine Validation
**TWO-LEG FIXTURES: PASS.** The manual engine executes signals one bar later and charges fee plus slippage on both legs on entry and exit.

| Fixture | Expected net PnL | Result |
|---|---:|---|
| ETH +10%, BTC +5% | +4.40% | PASS |
| ETH -3%, BTC +4% | -7.60% | PASS |
| ETH +5%, BTC +5% | -0.60% | PASS |

## Data and Inputs
- BTC/ETH 1h cache: 2024-01-01 through 2026-08-24, 23,194 bars each.
- OI: **PROXY** (24-hour rolling BTC volume); public OI data was not available in this run. Results for this family are degraded and not evidence about real OI.

## Results
| Signal | Tested | G1 pass | Furthest gate | Best PF | Best Sharpe |
|---|---:|---:|---|---:|---:|
| spread_two_leg | 20 | 0 | Gate 1 | 1.46 | 0.11 |
| realized_vol_regime | 10 | 2 | Gate 2 | 1.28 | 0.19 |
| open_interest_delta | 10 | 0 | Gate 1 | 0.00 | 0.00 |
| liquidation_bounce | 10 | 0 | Gate 1 | 0.00 | 0.00 |

## Required Answers
1. **Did two-leg simulation validate properly?** Yes: all three hand-calculated fixtures passed.
2. **Did spread perform differently from the single-leg proxy?** Yes. All 20 honest two-leg candidates failed Gate 1; 19/20 had PF below 0.80, and the remaining candidate had PF 1.46 but only 41 trades. Both directions were non-viable after two-leg costs.
3. **Which new family got furthest?** Realized-volatility regime: two rising-vol momentum candidates reached Gate 2, but failed OOS consistency.
4. **Primary gate killer:** SCREENING (48/50 genomes).
5. **Efficient-market assessment:** Within these 2024–26 data, costs, signal definitions, and proxy OI limitation, no tested structure survived. This is evidence against these implementations, not proof that every market signal is efficiently priced.

## Stop Rules
- Two-leg spread: **DEFINITIVELY CLOSED**. All 20 failed Gate 1; 19 were PF < 0.80, and the single exception was below the 50-trade requirement.
- Entire-research stop: **TRIGGERED.** All 50 failed Gate 1 or later and median PF was below 0.90. Do not run Generation 6 absent new data / HAR calibration evidence.

## Top Candidates
| Genome | Signal | Trades | PF | Sharpe | Gate |
|---|---|---:|---:|---:|---|
| `57276524f1ed` | realized_vol_regime | 207 | 1.28 | 0.19 | WALK_FORWARD |
| `47b13061f733` | spread_two_leg | 41 | 1.46 | 0.11 | SCREENING |
| `13df0d281524` | open_interest_delta | 0 | 0.00 | 0.00 | SCREENING |
| `27c1d562c103` | open_interest_delta | 0 | 0.00 | 0.00 | SCREENING |
| `d0ecbb4a68ae` | open_interest_delta | 0 | 0.00 | 0.00 | SCREENING |
