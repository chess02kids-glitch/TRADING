# Generation 4 Results — Blocked Before Execution

## Status

**Generation 4 did not start. Zero genomes were generated or evaluated.** This is an intentional validation stop, not a failed 50-genome run. The stated Gen 4 requirement is recent 2024–2026 BTC/USDT and ETH/USDT hourly data (plus aligned funding). The explicitly permitted primary cache files are absent from this checkout, the repository data available to this session ends before 2024, and the exchange/API endpoints cannot be reached from this sandbox.

No gate was changed, loosened, or bypassed. No forbidden directory was read, modified, or used as a data source.

## Data Used

No data was used for a Gen 4 backtest.

| Requirement | Result |
|---|---|
| Assets | BTC/USDT + ETH/USDT |
| Required date range | Recent 2024–2026, last 730 days |
| Required primary cache | `sandbox/pattern_research/cache/BTCUSDT_1h_730d.csv` and `ETHUSDT_1h_730d.csv`: **not present** (the cache contains only `.gitignore`) |
| Available local aligned spot range | 2017-08-17 to 2019-11-04 |
| Available later price range | 2019-12-01 to 2023-12-31 (BTC/USD Bitstamp; not BTC/USDT + ETH/USDT) |
| Available funding range | 2020-01-01 to 2023-12-31 |
| Exchange/API retrieval | Blocked: TLS connections to Binance, Kraken, Coinbase, and Yahoo endpoints failed in this sandbox |
| Source accepted for Gen 4 | None |
| Bars accepted for Gen 4 | 0 |

The local `data/cache/MANIFEST.json` records the relevant latest endpoints as 2023-12-31 23:00 UTC for price and funding. The now-permitted primary cache directory was read-only inspected; its required CSVs are absent. Therefore the mandatory data-recency check failed. No file under `sandbox/pattern_research/` was modified.

## Pre-run Validation Checks

1. **Data recency — FAIL / STOP.** The explicitly permitted recent CSVs are absent; available repository data is pre-2024. The Gen 4 backtest must not run on this historical data.
2. **Zero-trades protection — PASS.** `research/screener.py::gate1_screening` checks `total_trades == 0` first and returns `ZERO_TRADES_BUG` before any other screening metric.
3. **`size_type="percent"` — PASS after remediation.** Every `vbt.Portfolio.from_signals` call under `algo-research-lab/` now explicitly includes `size_type="percent"`, including the certification harness. The Gen 4 pipeline already used this mode through `research/screener.py::simulate`.
4. **No lookahead in spread calculation — NOT EXECUTED ON RECENT DATA.** The existing spread implementation uses rolling mean/std only; no future window is used. A Gen-4-specific timestamp assertion could not be executed because its required recent BTC/ETH data is unavailable.
5. **Funding alignment — NOT EXECUTED ON RECENT DATA.** Existing context code performs an as-of hourly alignment followed by `.shift(1)`. It cannot validate 2024–2026 funding without the required recent dataset.

## Target A — Spread Momentum v2 (40 genomes)

Not run. `spread_momentum_v2` genomes were not generated because doing so would require evaluating them on prohibited pre-2024 data or fabricating results.

| Fix group | Tested | Screening | Walk-forward | Concentration | Robustness | Stability | Survivors |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 scale-out exits | 0/10 | 0 | 0 | 0 | 0 | 0 | 0 |
| A2 adaptive z-score | 0/10 | 0 | 0 | 0 | 0 | 0 | 0 |
| A1 + A2 | 0/10 | 0 | 0 | 0 | 0 | 0 | 0 |
| A1 + A3 + A4 | 0/10 | 0 | 0 | 0 | 0 | 0 | 0 |

**Overall:** Concentration gate: **NOT TESTED**. OOS gate: **NOT TESTED**.

## Target B — dbf438 Stability Fix (10 genomes)

### Exact Genome Retrieval

The exact original genome was retrieved from the SQLite mirror, `data/research_generations.sqlite`, rather than reconstructed from memory:

```json
{
  "signal_type": "multi_asset_momentum",
  "primary_asset": "ETH/USDT",
  "lookback_bars": 164,
  "momentum_threshold": 0.023263046816635283,
  "require_confirmation": false,
  "holding_bars": 48,
  "size_pct": 0.5222990858300037,
  "confirmation_asset": "BTC/USDT",
  "name": "multi_asse_dbf438564958",
  "genome_id": "dbf438564958"
}
```

### Fragile Parameter Identification

Not run. The required +10%, -10%, +20%, and -20% single-parameter perturbation table cannot be calculated honestly on the mandatory 2024–2026 data. No fragile parameter is identified yet.

### Variant Results

0/10 variants tested; 0/10 passed Gate 5; 0 survivors.

## Overall Generation 4

- Total tested: **0**
- Survivors: **0**
- Primary failure mode: **DATA_RECENCY_VALIDATION** (pre-run block; not a five-gate outcome)
- Did the finding hold on recent 2024–2026 data? **NOT DETERMINED.** No valid recent-data evaluation was possible.

## Updated Signal Type Status

| Signal type | Gen 4 survivors | Status |
|---|---:|---|
| spread_momentum_v2 | 0 (not tested) | OPEN / unvalidated on recent data |
| multi_asset_momentum | 0 (not tested) | OPEN / unvalidated on recent data |

## Recommendation for Generation 5

Do not generate further genomes. First provide an allowed, auditable 730-day BTC/USDT and ETH/USDT 1-hour dataset through 2024–2026 and aligned 8-hour funding data, outside the forbidden `sandbox/pattern_research/` directory, or restore outbound exchange access. Then rerun all five validation checks, implement and timestamp-test the Gen-4 spread compiler, manually identify dbf438's single fragile parameter, and only then evaluate the specified 40 + 10 genomes under the unchanged gates.
