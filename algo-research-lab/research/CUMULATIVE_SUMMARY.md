# Algo Research Lab - Cumulative Summary (Gen 1 v2 Reset)

## Headline Numbers
- Total genomes tested: 180 (Gen1 80 + Gen2 50 + Gen3 50) + 1 post-fix rerun of a crashed genome
- Total survivors: 0
- Signal types that found survivors: [] (none)
- Signal types fully closed on this data: funding_trend, har_regime_sized, vol_regime_breakout (as pre-registered)
- Signal types still open (edge exists, gates not yet cleared): spread_zscore, funding_rate_contrarian, multi_asset_momentum

## Family Verdicts
| Signal Type | Tested | Best Sharpe | Best PF | Best gate reached | Verdict |
|---|---|---|---|---|---|
| funding_rate_contrarian | 43 | 0.70 | 1.87 | Gate 3 | OPEN: repeatable PF 1.2-1.9; trade-starved or OOS-fragile |
| funding_trend | 13 | -0.72 | 0.80 | Gate 1 | CLOSED: 0 passers, uniformly negative after costs |
| har_regime_sized | 13 | -2.99 | 0.79 | Gate 1 | CLOSED: 0 passers, fee churn dominates |
| multi_asset_momentum | 40 | 0.95 | 1.20 | Gate 5 | OPEN (narrow): one Gates-1-4 passer, failed only FRAGILE |
| spread_zscore | 58 | 1.55 | 2.18 | Gate 3 | OPEN: spread MOMENTUM at long windows is real (PF up to 2.18; momentum 21/28 PF-passers vs mean_revert 0/30); fails concentration/OOS |
| vol_regime_breakout | 13 | -0.76 | 0.53 | Gate 1 | CLOSED in pre-registered space: 77% zero trades |

## Best Genome by Sharpe (did NOT survive)
- Generation 2, `0a2d8d986dd7`, Sharpe 1.55, PF 2.09, 95 trades
- Genome JSON: `{"signal_type": "spread_zscore", "asset_a": "BTC/USDT", "asset_b": "ETH/USDT", "zscore_window": 238, "entry_zscore": 1.1023769843736404, "exit_zscore": 0.11497095978170002, "size_pct": 0.5543630439614519, "direction": "momentum"}`
- Failed: CONCENTRATION (HIGH_CONCENTRATION)

## Primary Gate Killing Strategies
- Gate 1 SCREENING: 144/180 (80%) - dominant reason LOW_PROFIT_FACTOR (115)
- WALK_FORWARD: 21/180
- CONCENTRATION: 13/180
- ROBUSTNESS: 0/180
- PARAMETER_STABILITY: 1/180
- CRASH: 1/180
- Among the 28 genomes that cleared Gate 1: FAILED_OOS_CONSISTENCY 15, HIGH_CONCENTRATION 12, FRAGILE 1.

## Key Empirical Finding
Across all 180 genomes, every profitable spread_zscore strategy was `direction: momentum`: 
- momentum: 21/28 passed PF>=1.05 (median 1.37, max 2.18), at long z-windows (104-336h)
- mean_revert: 0/30 passed (median 0.32, max 0.48)
The BTC/ETH log-ratio TRENDS at multi-day horizons (ETH-season vs BTC-season regimes persist);
fading it is a structural loser after two-leg costs. This mirrors the Gen-2 conclusion that 1h
crypto mean reversion is structurally broken - it also fails on the cross-asset spread.

## Recommendation for Gen 4
1. spread_zscore MOMENTUM is the highest-priority target: it repeatedly clears profitability AND OOS gates (best PF 2.18, Sharpe 1.55) and dies ONLY at concentration. Gen 4 should test profit-capped exits (partial scale-outs on the trending side) and intermediate entry z (1.0-1.5) to spread PnL across more episodes - because the 6 Gate-1+2 passers show the edge is real but carried by <5 trending episodes. Mean-revert spread variants should be retired (0/30).
2. Test the cross-exchange funding spread (Binance vs Bybit/Gate funding, 2020-2023 CSVs already in data/cache) as a new contrarian input - because funding_rate_contrarian is the only family whose profitability IMPROVES as thresholds tighten (PF 1.87) yet it never had both trade count AND OOS consistency at once; a cross-exchange disagreement signal is the natural sharpening of that edge.
3. Stability-first local search around multi_asset_momentum genome dbf438564958 (the only Gates-1-4 passer in 180 genomes): generate its +/-10%/+/-20% perturbation neighbourhood explicitly and keep neighbours whose WORST-case perturbed Sharpe stays within 30% - because it failed only FRAGILE, and random mutation (Gen 2/3 spec) samples the plateau, it does not select for it.
4. Do NOT loosen gates further: the loosened trade bar (Gen 3) produced zero new survivors; the binding constraint is economics (PF at scale), not sampling.

## Data & Environment Provenance
- Window A: Binance spot BTC/USDT + ETH/USDT 1h, 2017-08-17..2019-11-04, 19,414 aligned hourly bars (135/139 gap bars forward-filled, <0.7%). Source: vendored CSVs (see data/cache/MANIFEST.json).
- Window B: Bitstamp BTC/USD 1m resampled to 1h (35,808 bars, zero gaps, 2019-12-01..2023-12-31) + Binance USDT-M BTCUSDT funding (4,383 8h settlements, 03/11/19 UTC), merged with one-bar lag (no lookahead).
- Funding thresholds in genomes are percent per 8h (e.g. -0.02 = -0.02%).
- Supabase unreachable from this sandbox (no SUPABASE_DB_URL / network allowlist): every row was written one-by-one to the SQLite mirror data/research_generations.sqlite (identical schema to supabase/007_lab_schema.sql) and research/results/log.jsonl. supabase/007_lab_schema.sql applies the same table/columns (ADD COLUMN IF NOT EXISTS, nothing dropped) when credentials exist.
- Engine: vectorbt 1.1.0, certified by agent/certify_engine.py (7/7 tests incl. size_type='percent' sizing and same-bar close execution). Zero-trades guard is the first check in Gate 1.
- Gate 5 in Gen 3 used the same pre-registered stability parameters; only min_trades was loosened (documented above).

## Generation 4 Update
| Gen | Genomes | Survivors | Best Sharpe | Primary Gate |
|---|---:|---:|---:|---|
| 4 | 50 | 0 | 0.41 | SCREENING |
| TOTAL | 230 | 0 | 1.55 | |

| Signal Type | Status | Evidence |
|---|---|---|
| spread momentum | OPEN — mechanics validation required | 2024–26: 2/40 passed screening; both failed concentration. Current single-leg proxy did not implement actual partial closes. |
| multi_asset_momentum | CLOSED for this local stability neighborhood | 0/10 dbf438 lookback variants passed Gate 5. |
