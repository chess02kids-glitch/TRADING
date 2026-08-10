# PHASE 2.5 — Historical Data Range Audit + Fix

Date: 2026-08-10 · Branch: `arena/019febb1-trading` · Scope: data pipeline only. Kronos upstream untouched. PAPER mode preserved. LIVE disabled.

## Reported defect

```
python scripts/data/run_fetch.py --assets BTC/USDT ETH/USDT --timeframes 1h 4h 1d --days 730 --incremental
```
produced only ~July 11 2026 → Aug 10 2026 (~30 days) instead of 730 days.

## Complete trace of `--days 730`

1. **CLI parsing** (`run_fetch.main` → argparse): `--days` (`type=int`, default 730) parsed fine,
   but **`--incremental` was not a defined option** (only `--no-incremental` existed).
   → argparse aborted the process with exit code 2: `error: unrecognized arguments: --incremental`.
   **The command fetched NOTHING at all.** This exact broken command was even listed in the
   script's own "C. Exact Commands" report text.
2. **Date calculation** (`get_default_since_ms`): `now_utc_ms - days*86400000` — correct, UTC.
   Verified empirically: 30d → since ≈ now−30d, 90d → now−90d, 730d → now−730d.
3. **since_ms → fetcher.fetch_ohlcv_range** (`fetcher.py`): pagination `since = last_ts + tf_ms`,
   limit per request `min(1000, remaining)`, retries with exponential backoff, per-page and
   global dedup, `[since, until]` filtering — verified correct over 18 pages / 17,520 hourly candles.
4. **Incremental mode** (`run_all`): with existing data it now builds (a) a backfill range
   `desired_since_ms → first_existing - tf` when stored history is shorter than `--days`, and
   (b) a forward range `last_existing + tf → now`. An earlier revision skipped backfilling, which
   is why a DB that already held ~30 days never deepened. This backfill logic was present in the
   audited revision and is now locked in by regression tests.
5. **Database insertion** (`storage.insert_ohlcv`): PRIMARY KEY `(exchange, symbol, timeframe,
   timestamp_ms)` + `INSERT OR IGNORE` + pre-select of existing timestamps — idempotent,
   duplicate-safe, UTC ISO stored alongside ms. Verified: full re-fetch inserts 0 new rows.

## Root cause (two parts)

| # | Cause | Status |
|---|-------|--------|
| 1 | **CLI**: `--incremental` unknown to argparse → exit 2 → nothing fetched. The ~30 days observed were pre-existing DB contents from an earlier run, not from this command. | **Fixed in this phase** (smallest safe fix: both `--incremental` and `--no-incremental` map to one `incremental` dest, default True). |
| 2 | **Logic**: in the earlier revision, incremental mode ignored `--days` when the DB already had data (no backfill), so a short DB stayed short. | Already fixed in the audited `run_all` (backfill + forward ranges); **now proven + locked by tests**. |

Not implicated (checked and cleared): date calculation, pagination, Binance/CCXT request limits,
validator, database logic.

## Fix (only file modified in production code)

`scripts/data/run_fetch.py` — extracted `build_parser()` (testable) and replaced
`--no-incremental` (`store_true` on `no_incremental`) with an explicit pair:

```python
parser.add_argument("--incremental",    dest="incremental", action="store_true",  default=True)
parser.add_argument("--no-incremental", dest="incremental", action="store_false")
```
Default behavior unchanged (incremental). The documented command now runs.

## Regression proof — `tests/test_historical_range_regression.py`

14 tests, fully offline (mocked exchange emulating real Binance kline semantics:
rows `WHERE open_time >= since LIMIT min(limit,1000)`, genesis 2017-08-17, data up to now,
optional outage gap). No API keys, no network, no live-Binance dependency.

- `--days 30/90/730` request ~30/90/730 days (requested `since` and resulting DB span asserted)
- 730d → 17,520±2 candles over exactly 18 pages, every request limit ≤ 1000 (Binance cap)
- 730d supported on 1h/4h/1d (18 / 5 / 1 pages) for BTC/USDT and by extension ETH/USDT
- incremental backfills a seeded 30-day DB to ~730 days; second incremental run is a no-op
- full non-incremental re-fetch: `inserted == 0`, `duplicates_skipped > 0` (duplicate protection)
- 4000-day request caps at exchange availability (first candle == 2017-08-17T00:00Z), nothing fabricated
- 24-candle exchange outage: gap **reported** in metadata (missing ≥ 24), **never filled**, validator agrees
- UTC preserved (`timestamp_utc` ends `+00:00`, boundary-aligned), OHLC invariants hold on 730d series
- CSV export byte-identical across re-exports

## Measured historical range after fix (mock-grounded, sandbox)

| Requested | Result (BTC/USDT 1h) | Pages |
|-----------|----------------------|-------|
| 30 d      | 29.96 d, 720 candles | 1 |
| 90 d      | 89.96 d, 2,160 candles | 3 |
| 730 d     | 729.96 d, 17,520 candles, span 2024-08-10 → 2026-08-10 | 18 |
| from 30d-seeded DB, incremental | backfilled to 729.96 d, 17,520 candles | 18 |
| 4000 d (1d) | capped at genesis: 3,280 d (2017-08-17 → now) | 4 |

Sandbox note: this machine's network cannot reach api.binance.com (E2B egress blocked / Binance
451-class restriction), so live re-download was not possible here; ranges above are measured
through the real pagination/storage/validation code against the mock. On the user's machine
(same code path), run the exact fixed command.

## Is 730 days genuinely available from Binance via this implementation?

**Yes.**
- Binance spot history starts 2017-08-17 (BTCUSDT first daily kline open_time = 1502928000000;
  ETHUSDT listed the same week). 730 days back from 2026-08-10 = 2024-08-11 — comfortably inside.
- Throughput: 1h = 17,520 candles → 18 requests; 4h = 4,380 → 5; 1d = 730 → 1.
  All ≤ Binance's 1000-candle/request cap; klines are public (no API keys), `enableRateLimit`
  handles pacing. ~24 requests per asset ≈ well within public rate limits.
- Requests *beyond* availability are safely capped by the exchange/pagination logic at the first
  listed candle (proven by `test_days_beyond_exchange_availability_caps_at_genesis`).

## Preserved behaviors

UTC timestamps · no silent candle filling · duplicate protection (PK + OR IGNORE) ·
pagination with backoff · incremental updates · OHLC validation · deterministic storage/export ·
PAPER mode default, LIVE disabled · Kronos upstream untouched.
