# Phase 2 Report - Trustworthy Data Foundation

**Date:** 2026-08-10  
**Status:** Completed - Historical reliability prioritized over real-time  
**Assets:** BTC/USDT, ETH/USDT  
**Timeframes:** 1h primary, 4h confirmation, 1d regime (no 15m)  
**Exchange:** Binance public via CCXT - NO API keys required  
**LIVE Trading:** Completely disabled  
**Kronos Upstream:** Untouched (commit 67b630e)  
**Verification Language:** Per final correction #1, RTX 3060 results labeled as "target-machine verification pending" until actually run on RTX 3060

---

## A. Files Created/Modified

### Created in Phase 2:

1. **config/data_schema.yaml** - Machine-readable schema spec per requirement #6
   - Defines DB path, assets, timeframes, timeframe_ms mapping, table schemas, validation rules, fees metadata separation

2. **docs/DATA_SCHEMA.md** - Human-readable schema & validation spec per #6
   - Design principles, raw data source, DB schema (4 tables), validation spec (8 checks), fetcher spec, test plan, expected volume

3. **scripts/data/fetcher.py** - CCXT Binance public OHLCV fetcher per #8
   - No API keys (requirement #3)
   - Pagination: loop with `since = last_ts + timeframe_ms`, max 1000 per request
   - Rate-limit: `enableRateLimit=True` + exponential backoff retries (1s,2s,4s,8s,16s) with jitter on NetworkError, RateLimitExceeded, DDoSProtection, ExchangeNotAvailable
   - Duplicate detection within batch and across pages
   - Timestamp normalization to UTC: ms integer + ISO UTC via `pd.to_datetime(ms, unit='ms', utc=True)`
   - Deterministic incremental updates: `get_last_timestamp_ms + timeframe_ms`
   - CSV export derived from SQLite, sorted ASC

4. **scripts/data/validator.py** - Validation per #8, #9, #11
   - Duplicate detection
   - Missing candle detection (detects and reports, does NOT fill per #9)
   - Out-of-order detection
   - Invalid OHLC (high>=low etc.)
   - Timezone conversion check
   - All validations preserve original data

5. **scripts/data/storage.py** - SQLite storage per #8, #10
   - `ohlcv_raw` table with PRIMARY KEY (exchange, symbol, timeframe, timestamp_ms) - prevents duplicates, ensures repeated fetches produce no duplicates
   - No fees/slippage in raw table per #10 - those live in config/config.yaml
   - Deterministic incremental updates via `get_last_timestamp_ms`
   - CSV export `data/raw/binance_{SYMBOL}_{TF}.csv` derived from SQLite, sorted ASC
   - WAL mode, fetch_metadata, validation_reports tables

6. **scripts/data/run_fetch.py** - Orchestrator
   - Fetches all assets/timeframes, handles incremental vs full history, generates final report A-I
   - Measures: candles fetched, inserted, duplicates skipped, missing detected, date ranges, validation results

7. **tests/test_data_pipeline.py** - Tests per #11
   - 13 tests: duplicate, missing, out-of-order, invalid OHLC high_low, high_open, timezone, pagination boundaries, repeated fetches no duplicates, incremental updates, deterministic CSV export, volume negative, timestamp normalization UTC, public fetch no API keys
   - Uses mocked exchange and in-memory SQLite, no network needed, no API keys

### Modified in Phase 2:

8. **config/config.yaml** - Already audited in Phase 1, timeframes 1h/4h/1d confirmed, fees/slippage metadata present but not in raw table per #10

9. **docs/PHASE1_AUDIT_REPORT.md** - Updated language per final correction #1 to label RTX 3060 as "target-machine verification pending" not "expected 10/10 PASS"

### Generated Data (Demo for Report - Synthetic Valid Data):

10. **data/db/kronos_trading.db** - SQLite DB with synthetic valid OHLCV for 30 days demo (real Binance fetch pending on target machine due to sandbox 451 restriction)
11. **data/raw/binance_BTC_USDT_1h.csv** etc. - 6 CSVs derived from SQLite

### Not Created (Per Requirement #7):

- No websocket/streaming infrastructure - deferred
- No `ohlcv_processed` table - filling is explicit configurable step preserving original, not in Phase 2
- No prediction/backtesting

---

## B. Database Schema (Actual CREATE TABLE SQL)

**File:** `data/db/kronos_trading.db` - SQLite with WAL mode

**Actual SQL (from `storage.get_schema_sql()`):**

```sql
CREATE TABLE ohlcv_raw (
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    timestamp_utc TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'binance_ccxt_public',
    created_at TEXT NOT NULL,
    PRIMARY KEY (exchange, symbol, timeframe, timestamp_ms),
    CHECK (high >= low),
    CHECK (high >= open),
    CHECK (high >= close),
    CHECK (low <= open),
    CHECK (low <= close),
    CHECK (open > 0),
    CHECK (high > 0),
    CHECK (low > 0),
    CHECK (close > 0),
    CHECK (volume >= 0)
);

CREATE TABLE fetch_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    fetch_start_ms INTEGER,
    fetch_end_ms INTEGER,
    candles_fetched INTEGER,
    candles_inserted INTEGER,
    duplicates_skipped INTEGER,
    missing_candles_detected INTEGER,
    first_timestamp_ms INTEGER,
    last_timestamp_ms INTEGER,
    first_timestamp_utc TEXT,
    last_timestamp_utc TEXT,
    fetch_duration_s REAL,
    status TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE validation_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    check_type TEXT NOT NULL,
    is_valid BOOLEAN NOT NULL,
    issues_found INTEGER,
    details TEXT,
    checked_from_ms INTEGER,
    checked_to_ms INTEGER,
    created_at TEXT NOT NULL
);
```

**Indexes:**
```sql
CREATE INDEX idx_ohlcv_symbol_timeframe_ts ON ohlcv_raw (symbol, timeframe, timestamp_ms);
CREATE INDEX idx_fetch_meta_symbol_tf ON fetch_metadata (symbol, timeframe, created_at);
```

**Key Points:**
- PRIMARY KEY on (exchange, symbol, timeframe, timestamp_ms) ensures no duplicates, idempotent repeated fetches
- CHECK constraints enforce OHLC sanity at DB level
- No fee/slippage columns - per #10, fees live in config/config.yaml
- CSV derived from this table, sorted ASC

---

## C. Exact Commands (Windows-Compatible - No &&, Separate Commands)

Per final correction #2: Provide separate commands, Windows-compatible (cmd and PowerShell both support separate lines).

**Phase 1 Verification (Windows-Compatible):**

```bat
REM Windows cmd - separate commands, no &&
python scripts/setup/bug_audit.py
python scripts/setup/environment_report.py
python scripts/setup/verify_install.py
python scripts/broker/trading_mode_guard.py
```

**Phase 2 Data Pipeline:**

```bat
REM 1. Install dependencies (if not already)
pip install -r requirements_exact.txt

REM 2. Run tests - no API keys, no network needed for mocked tests
pytest tests/test_data_pipeline.py -v

REM 3. Fetch historical data - NO API keys required per #3, public OHLCV only
REM    For 30 days demo (fast):
python scripts/data/run_fetch.py --assets BTC/USDT ETH/USDT --timeframes 1h 4h 1d --days 30 --no-incremental

REM    For full 730 days per config (production):
python scripts/data/run_fetch.py --assets BTC/USDT ETH/USDT --timeframes 1h 4h 1d --days 730 --incremental

REM    Or defaults (same as above):
python scripts/data/run_fetch.py

REM 4. Check database and CSVs
dir data\db\
dir data\raw\

REM 5. Validate manually (optional)
python scripts/data/validator.py
```

**Linux/Mac (separate commands):**

```bash
python scripts/setup/bug_audit.py
python scripts/setup/environment_report.py
python scripts/setup/verify_install.py
pytest tests/test_data_pipeline.py -v
python scripts/data/run_fetch.py --assets BTC/USDT ETH/USDT --timeframes 1h 4h 1d --days 30 --no-incremental
```

**PowerShell (Windows):**

```powershell
python scripts/setup/bug_audit.py
python scripts/setup/environment_report.py
python scripts/setup/verify_install.py
pytest tests/test_data_pipeline.py -v
python scripts/data/run_fetch.py
```

---

## D. Number of Candles Downloaded Per Asset/Timeframe (Measured)

**Note:** In sandbox environment, Binance public API returns 451 "Service unavailable from restricted location" - expected in restricted sandbox IP. Library works without API keys, but network fetch returns 0. To demonstrate pipeline reliability, synthetic valid OHLCV generated for 30 days per data_schema spec for report. On target RTX 3060 machine with unrestricted internet, same commands will fetch real Binance data producing similar counts.

**Measured in Sandbox with Synthetic Demo Data (30 days):**

| Asset | Timeframe | Candles Fetched | Candles Inserted (New) | Duplicates Skipped | Total in DB | Note |
|-------|-----------|-----------------|------------------------|-------------------|-------------|------|
| BTC/USDT | 1h | 720 | 720 | 0 | 725 (720+5 incremental test) | 30 days *24 =720 |
| BTC/USDT | 4h | 179 (180-1 missing) | 179 | 0 | 179 | Intentionally 1 missing for test |
| BTC/USDT | 1d | 30 | 30 | 0 | 30 | 30 days |
| ETH/USDT | 1h | 720 | 720 | 0 | 720 | 30 days |
| ETH/USDT | 4h | 180 | 180 | 0 | 180 | 6 per day *30 |
| ETH/USDT | 1d | 30 | 30 | 0 | 30 | 1 per day *30 |
| **Total** | - | **1859** | **1859** | **0** | **1864** with incremental | - |

**Repeated Fetch Test (Measured):**
- First insert BTC/USDT 1h 720 candles: inserted 720, dups 0
- Second insert same 720 candles: inserted 0, dups skipped 720 - **No duplicate rows, idempotent per #11**

**Incremental Update Test (Measured):**
- Initial BTC/USDT 1h: 720
- After incremental 5 new candles: 725 total, contiguous
- Verified no gaps after incremental

**Target-Machine Verification Pending (Per Final Correction #1):**
On RTX 3060 Windows with unrestricted internet, running `python scripts/data/run_fetch.py --days 730` is expected to produce:
- 1h: ~17,520 per asset (730*24), 2 assets = 35,040
- 4h: ~4,380 per asset (730*6), 2 assets = 8,760
- 1d: 730 per asset, 2 assets = 1,460
- Total ~45k rows - lightweight, but actual numbers will be **measured** on target machine, not claimed here. Report will be updated after you run commands on RTX 3060.

---

## E. Date Range Actually Available (Measured)

**Synthetic Demo Data (30 days) - Measured:**

| Asset | TF | First Timestamp ms | First ISO UTC | Last Timestamp ms | Last ISO UTC |
|-------|----|-------------------|---------------|-------------------|--------------|
| BTC/USDT | 1h | 1783748263241 | 2026-07-11T05:37:43.241000+00:00 | 1786336663241 | 2026-08-10T04:37:43.241000+00:00 |
| BTC/USDT | 4h | 1783748263241 | 2026-07-11T05:37:43.241000+00:00 | 1786325863241 | 2026-08-10T01:37:43.241000+00:00 |
| BTC/USDT | 1d | 1783748263241 | 2026-07-11T05:37:43.241000+00:00 | 1786253863241 | 2026-08-09T05:37:43.241000+00:00 |
| ETH/USDT | 1h | 1783748263241 | 2026-07-11T05:37:43.241000+00:00 | 1786336663241 | 2026-08-10T04:37:43.241000+00:00 |
| ETH/USDT | 4h | 1783748263241 | 2026-07-11T05:37:43.241000+00:00 | 1786325863241 | 2026-08-10T01:37:43.241000+00:00 |
| ETH/USDT | 1d | 1783748263241 | 2026-07-11T05:37:43.241000+00:00 | 1786253863241 | 2026-08-09T05:37:43.241000+00:00 |

**Real Binance Data (Target-Machine Verification Pending):**
When you run on RTX 3060, date range will be actual Binance history. For 730 days, first timestamp will be ~2 years ago from now (e.g., 2024-08-10 to 2026-08-10). Actual range will be measured and reported after you run fetch on target machine. Sandbox cannot fetch due to 451 restriction - this is known limitation, not data issue.

---

## F. Missing-Data Statistics (Measured, No Silent Filling per #9)

**Measured in Demo DB:**

| Asset | TF | Missing Candles Detected | Gaps Found | Details | Action Taken |
|-------|----|--------------------------|------------|---------|--------------|
| BTC/USDT | 1h | 0 | 0 | Contiguous 720 | None - data preserved |
| BTC/USDT | 4h | 1 | 1 | Gap: 2026-07-26T01:37:43 to 2026-07-26T09:37:43, missing 1 (intentionally removed for test) | **Detected and reported, NOT filled** - per #9 |
| BTC/USDT | 1d | 0 | 0 | Contiguous 30 | None |
| ETH/USDT | 1h | 0 | 0 | Contiguous 720 | None |
| ETH/USDT | 4h | 0 | 0 | Contiguous 180 | None |
| ETH/USDT | 1d | 0 | 0 | Contiguous 30 | None |

**Policy per Requirement #9:**
- Missing candles are **detected and reported** via `validator.check_missing_candles()` - returns gaps list with from/to, missing_count
- **No silent filling** - `ohlcv_raw` preserves original data with missing remaining missing
- If interpolation ever needed, it must be explicit configurable preprocessing step that creates new table `ohlcv_processed` with columns `is_interpolated BOOLEAN`, `original_timestamp_ms`, `processing_config TEXT`, never overwriting `ohlcv_raw`
- In this demo, BTC 4h missing 1 candle intentionally to test detection - remains missing, not filled

**Real Binance Data:**
On target machine, missing data statistics will be measured. Binance spot typically has no missing for major pairs like BTC/USDT, but exchange downtime can cause gaps - our validator will detect and report.

---

## G. Validation Results (Measured per Check)

**Per Requirement #11 and DATA_SCHEMA.md Section 4:**

| Asset | TF | Duplicate Valid | Missing Count | Out-of-Order Sorted | Was Out-of-Order | Invalid OHLC Valid | Invalid Count | Timezone Valid | Overall Valid |
|-------|----|-----------------|---------------|---------------------|------------------|--------------------|---------------|----------------|---------------|
| BTC/USDT | 1h | True (0 dups) | 0 | True | False | True | 0 | True | True |
| BTC/USDT | 4h | True (0 dups) | 1 (intentional gap) | True | False | True | 0 | True | True* |
| BTC/USDT | 1d | True | 0 | True | False | True | 0 | True | True |
| ETH/USDT | 1h | True | 0 | True | False | True | 0 | True | True |
| ETH/USDT | 4h | True | 0 | True | False | True | 0 | True | True |
| ETH/USDT | 1d | True | 0 | True | False | True | 0 | True | True |

*Overall valid True for 4h despite 1 missing - missing is reported not failing per #9 policy, but duplicate and invalid OHLC must be valid.

**Detailed Checks (Measured):**

- **Duplicate:** 0 duplicates across all 1864 rows - PRIMARY KEY prevents duplicates, repeated inserts produce 0 new rows
- **Missing:** 1 missing detected in BTC 4h (intentional test), 0 in others
- **Out-of-Order:** All sorted ASC, no out-of-order - storage enforces deterministic ASC order
- **Invalid OHLC:** 0 invalid - all candles satisfy high>=low, high>=open/close, low<=open/close, open/high/low/close>0, volume>=0
- **Timezone:** All timestamps UTC - ms to ISO conversion produces `+00:00` UTC, no local timezone

**Validation Method:**
```python
validator = DataValidator(timeframe)
results = validator.validate_all(candles)
# results contains duplicate, missing, out_of_order, invalid_ohlc, timezone, overall_valid
```

---

## H. Test Results (Measured)

**Command (Windows-Compatible, Separate Lines):**

```bat
pytest tests/test_data_pipeline.py -v
```

**Measured in Sandbox (CPU, No API Keys, No Network Needed for Mocked Tests):**

```
============================= test session starts ==============================
platform linux -- Python 3.13.14, pytest-9.0.3
collected 13 items

tests/test_data_pipeline.py::test_duplicate_candles PASSED               [  7%]
tests/test_data_pipeline.py::test_missing_candles PASSED                 [ 15%]
tests/test_data_pipeline.py::test_out_of_order_candles PASSED            [ 23%]
tests/test_data_pipeline.py::test_invalid_ohlc_high_low PASSED           [ 30%]
tests/test_data_pipeline.py::test_invalid_ohlc_high_open PASSED          [ 38%]
tests/test_data_pipeline.py::test_timezone_conversion PASSED             [ 46%]
tests/test_data_pipeline.py::test_pagination_boundaries PASSED           [ 53%]
tests/test_data_pipeline.py::test_repeated_fetches_no_duplicates PASSED  [ 61%]
tests/test_data_pipeline.py::test_incremental_updates PASSED             [ 69%]
tests/test_data_pipeline.py::test_deterministic_csv_export PASSED        [ 76%]
tests/test_data_pipeline.py::test_volume_negative PASSED                 [ 84%]
tests/test_data_pipeline.py::test_timestamp_normalization_utc PASSED     [ 92%]
tests/test_data_pipeline.py::test_public_fetch_no_api_keys PASSED        [100%]

======================== 13 passed, 1 warning in 0.86s =========================
```

**Test Coverage per Requirement #11:**

- ✅ duplicate candles - `test_duplicate_candles` - verifies duplicate timestamp detection and idempotent insert (0 new on repeat)
- ✅ missing candles - `test_missing_candles` - detects gap, reports missing count, verifies no silent filling
- ✅ out-of-order candles - `test_out_of_order_candles` - detects unsorted input, ensures storage sorts ASC
- ✅ invalid OHLC relationships - `test_invalid_ohlc_high_low`, `test_invalid_ohlc_high_open`, `test_volume_negative` - high<low, high<open rejected
- ✅ timezone conversion - `test_timezone_conversion`, `test_timestamp_normalization_utc` - UTC ms to ISO, roundtrip preserved
- ✅ pagination boundaries - `test_pagination_boundaries` - mock exchange 2 per page, 5 total via 3 pages, no duplicates at boundaries, since param advances correctly
- ✅ repeated fetches producing no duplicate rows - `test_repeated_fetches_no_duplicates` - insert same batch twice, second 0 inserted
- ✅ incremental updates - `test_incremental_updates` - t1..t10 initial, plus t11..t15 incremental, contiguous, no gaps

**Additional Tests:**
- `test_deterministic_csv_export` - CSV export twice produces identical content
- `test_public_fetch_no_api_keys` - ensures fetcher config has no apiKey/secret, per #3

All 13 tests use mocked exchange and in-memory SQLite, no API keys, no network - reliable even when Binance 451 blocked.

---

## I. Known Limitations

1. **Binance 451 in Sandbox** - Sandbox IP restricted by Binance: "Service unavailable from restricted location". Public `fetch_ohlcv` returns 0 in sandbox after 5 retries with exponential backoff. Library works without API keys (requirement #3 met), but Network fetch blocked. **Target-machine verification pending:** On RTX 3060 Windows with unrestricted internet, same commands will succeed and populate real data. This is environment limitation, not code flaw.

2. **Synthetic Demo Data for Report** - Due to 451, this report uses synthetic valid OHLCV (720 1h, 180 4h, 30 1d per asset for 30 days) to demonstrate pipeline reliability, validation, and measurements. Real Binance data will be fetched on target machine and produce ~45k rows for 730 days (17,520 1h, 4,380 4h, 730 1d per asset). Actual numbers will be measured on target.

3. **No Streaming Infrastructure** - Per requirement #7, no websocket/real-time infra in Phase 2. Historical reliability prioritized. Streaming can be added in Phase 8 if needed.

4. **No Silent Filling** - Per #9, missing candles remain missing in `ohlcv_raw`. If you need filling for Kronos (which expects contiguous), it must be explicit configurable preprocessing step that creates new table preserving original. Not implemented in Phase 2.

5. **Fees/Slippage Not in Raw Table** - Per #10, `ohlcv_raw` has no fee columns. Fees live in `config/config.yaml` `strategy.fee_pct` (0.001) and `broker.binance.fees`. Applied later in backtesting/execution.

6. **LIVE Trading Completely Disabled** - Per #4, guard in `trading_mode_guard.py` blocks LIVE unless triple confirmation. Phase 2 is data only, no orders.

7. **Kronos Upstream Untouched** - Per #5, `Kronos/` folder git status clean, commit 67b630e, no file modifications. Patches external via compatibility layer.

8. **SQLite WAL Mode** - Suitable for $100 small account, single writer multiple readers, lightweight. Not for high-frequency. File-based reproducible.

9. **Target-Machine Verification Pending** - Per final correction #1, RTX 3060 results not claimed as "expected PASS" but labeled as pending until you actually run commands on RTX 3060. This report's measured numbers are from sandbox synthetic demo - real Binance counts will be measured on target.

10. **Rate Limits** - Binance 1200 requests/min weight. CCXT `enableRateLimit` + exponential backoff handles, but full 730-day fetch for 6 pairs (2 assets *3 TF) = ~45 requests (1000 per request) = ~1 min with rate limiting.

11. **Timezone** - All timestamps UTC ms + ISO UTC. No local timezone. Ensure your Windows system time is correct UTC, not local.

12. **CSV Export** - Derived from SQLite, deterministic sorted ASC. If you manually edit CSV, DB remains source of truth - re-export will overwrite.

---

## Final Verification Commands (Windows-Compatible)

```bat
REM Phase 1
python scripts/setup/bug_audit.py
python scripts/setup/environment_report.py
python scripts/setup/verify_install.py

REM Phase 2
pytest tests/test_data_pipeline.py -v
python scripts/data/run_fetch.py --assets BTC/USDT ETH/USDT --timeframes 1h 4h 1d --days 30 --no-incremental

REM Check results
dir data\db\
dir data\raw\
```

**On RTX 3060 Target Machine (Verification Pending):**

```bat
REM After conda env setup per Phase 1

REM Full 2-year history, incremental mode
python scripts/data/run_fetch.py --assets BTC/USDT ETH/USDT --timeframes 1h 4h 1d --days 730 --incremental

REM You should then see:
REM - data/db/kronos_trading.db ~5-10 MB for 730 days
REM - data/raw/ 6 CSVs with 17k, 4k, 730 rows per asset
REM - logs/phase2_report.json with actual measured counts and date ranges
REM - No API keys required, public data only
REM - LIVE still disabled
```

---

## Conclusion

Phase 2 establishes trustworthy data foundation with:
- ✅ Clear schema and validation spec (DATA_SCHEMA.md, data_schema.yaml) for BTC/USDT, ETH/USDT, 1h/4h/1d
- ✅ CCXT Binance public fetcher, no API keys, pagination, rate-limit, retries with exponential backoff
- ✅ Duplicate detection via PRIMARY KEY, idempotent repeated fetches
- ✅ Timestamp UTC normalization, missing detection (no silent filling), OHLC sanity checks
- ✅ SQLite storage, CSV export, deterministic incremental updates
- ✅ Fees/slippage metadata separate per #10
- ✅ 13 tests covering all required cases per #11, all passed
- ✅ LIVE disabled, Kronos untouched, historical reliability prioritized
- ✅ Target-machine verification pending language per final correction #1
- ✅ Windows-compatible commands per #2

**Phase 2 Status: PASS with synthetic demo data measured, real Binance fetch target-machine verification pending due to sandbox 451 restriction.**

**Do not start Phase 3 (prediction) until Phase 2 real data fetch verified on RTX 3060.**

Next: After you run fetch on RTX 3060 and confirm real Binance data (not synthetic), we proceed to Phase 3: Kronos inference wrapper with confidence scoring, no model file modifications, external compatibility layer only.
