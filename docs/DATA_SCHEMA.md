# Phase 2 - Data Schema & Validation Specification

**Status:** Audited, trustworthy foundation first - before prediction/backtesting
**Assets:** BTC/USDT, ETH/USDT (Binance spot)
**Timeframes:** 1h (primary), 4h (confirmation), 1d (regime) - No 15m per Phase 1 audit
**Exchange:** Binance via CCXT public endpoints - NO API keys required for historical OHLCV
**LIVE Trading:** Completely disabled - Phase 2 is data only
**Kronos Upstream:** Untouched per audit

---

## 1. Design Principles (Per Audit Requirements)

1. **Public data without credentials** - `fetch_ohlcv` via CCXT `binance` public API, no `apiKey`/`secret` required. Keys only for future TESTNET/LIVE execution (Phase 7), not for Phase 2.
2. **Reliability over real-time** - Prioritize historical correctness. No websocket/streaming infra in Phase 2 (deferred to Phase 8 if needed).
3. **No silent filling** - Missing candles are **detected and reported**, not interpolated. Original data preserved. Filling, if ever needed, is explicit configurable preprocessing step that creates new table, never overwrites raw.
4. **Fees/slippage separation** - Raw OHLCV table contains only market data. Trading-cost assumptions (fee_pct, slippage_pct) live in `config/config.yaml` under `strategy:` and `broker:` metadata, not in raw table.
5. **Deterministic incremental updates** - Re-running fetcher for same range produces **zero duplicate rows** - idempotency via UNIQUE constraint + upsert logic.
6. **UTC normalization** - All timestamps stored as UTC milliseconds integer + ISO8601 UTC string. No local timezone.

---

## 2. Raw Data Source

- **Exchange:** Binance spot (CCXT id `binance`)
- **Endpoints:** `fetch_ohlcv` public REST - `https://api.binance.com/api/v3/klines`
- **Endpoints require:** No API key - public market data
- **Symbols:** `BTC/USDT`, `ETH/USDT` - CCXT format, maps to Binance `BTCUSDT`, `ETHUSDT`
- **Timeframes:** `1h`, `4h`, `1d` - CCXT timeframe strings
- **Limit:** Binance max 1000 candles per request - pagination required for larger ranges
- **Binance returns:** `[timestamp_ms, open, high, low, close, volume]` - timestamp is open time in UTC ms

---

## 3. Database Schema

### SQLite Database
- **Path:** `data/db/kronos_trading.db` (from config)
- **Engine:** SQLite via SQLAlchemy or sqlite3 - lightweight, reproducible, file-based
- **Modes:** WAL mode enabled for concurrent reads

### Table 1: `ohlcv_raw` - Raw Market Data (Preserved, No Filling)

**Purpose:** Store original exchange data exactly as received, after basic sanity checks but before any filling. This is source of truth.

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS ohlcv_raw (
    exchange TEXT NOT NULL,              -- e.g., 'binance'
    symbol TEXT NOT NULL,                -- e.g., 'BTC/USDT' (CCXT format)
    timeframe TEXT NOT NULL,             -- '1h', '4h', '1d'
    timestamp_ms INTEGER NOT NULL,        -- Open time, UTC milliseconds, e.g., 1690000000000
    timestamp_utc TEXT NOT NULL,          -- ISO8601 UTC, e.g., '2023-07-22T00:00:00Z' derived from timestamp_ms
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,                -- Base asset volume (BTC volume for BTC/USDT)
    -- Metadata, not trading assumptions:
    source TEXT NOT NULL DEFAULT 'binance_ccxt_public', -- Where data came from
    created_at TEXT NOT NULL,            -- When row inserted, ISO UTC
    -- Constraints:
    PRIMARY KEY (exchange, symbol, timeframe, timestamp_ms), -- UNIQUE + dedup per audit #8
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
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe_ts ON ohlcv_raw (symbol, timeframe, timestamp_ms);
```

**Columns Explained:**
- `timestamp_ms`: Integer UTC ms - primary time key, deterministic, timezone-agnostic
- `timestamp_utc`: ISO8601 UTC for human readability, derived via `pd.to_datetime(timestamp_ms, unit='ms', utc=True).isoformat()`
- `open/high/low/close/volume`: REAL, as returned by Binance, no transformation except float conversion
- **No fee/slippage columns** - per requirement #10, those live in config, not raw table
- UNIQUE via PRIMARY KEY on `(exchange, symbol, timeframe, timestamp_ms)` ensures **no duplicate rows on repeated fetches** - per requirement #11

**Storage Details:**
- CSV Export: `data/raw/{exchange}_{symbol}_{timeframe}.csv` e.g., `binance_BTC_USDT_1h.csv` - same data as SQLite, for Kronos compatibility (Kronos examples use CSV/DF)
- CSV columns: `timestamp_ms,timestamp_utc,open,high,low,close,volume,exchange,symbol,timeframe`
- CSV sorted by `timestamp_ms ASC` deterministic

### Table 2: `fetch_metadata` - Fetch History & Stats

**Purpose:** Audit trail of data collection, date ranges actually available, missing-data stats per requirement.

```sql
CREATE TABLE IF NOT EXISTS fetch_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    fetch_start_ms INTEGER,              -- Requested start
    fetch_end_ms INTEGER,                -- Requested end (or now)
    candles_fetched INTEGER,             -- From exchange
    candles_inserted INTEGER,            -- New rows inserted (excluding duplicates)
    duplicates_skipped INTEGER,          -- Duplicate timestamps skipped
    missing_candles_detected INTEGER,    -- Gaps detected
    first_timestamp_ms INTEGER,          -- Actual first candle in DB after fetch
    last_timestamp_ms INTEGER,           -- Actual last candle in DB
    first_timestamp_utc TEXT,
    last_timestamp_utc TEXT,
    fetch_duration_s REAL,               -- How long fetch took
    status TEXT,                         -- 'success', 'partial', 'failed'
    error_message TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fetch_meta_symbol_tf ON fetch_metadata (symbol, timeframe, created_at);
```

### Table 3: `validation_reports` - Validation Results

```sql
CREATE TABLE IF NOT EXISTS validation_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    check_type TEXT NOT NULL,            -- 'duplicate', 'missing', 'out_of_order', 'invalid_ohlc', 'timezone', 'pagination'
    is_valid BOOLEAN NOT NULL,
    issues_found INTEGER,
    details TEXT,                        -- JSON with details of issues
    checked_from_ms INTEGER,
    checked_to_ms INTEGER,
    created_at TEXT NOT NULL
);
```

### Table 4 (Optional, NOT in Phase 2 Raw): `ohlcv_processed` - Explicit Preprocessing (Not Used Yet)

Per requirement #9: If interpolation ever used, it must be explicit and preserve original. So we define empty schema but don't populate in Phase 2.

```sql
-- NOT created in Phase 2, only specification:
-- CREATE TABLE ohlcv_processed AS SELECT * FROM ohlcv_raw WHERE 1=0;
-- Plus additional columns: is_interpolated BOOLEAN, original_timestamp_ms INTEGER NULL, processing_config TEXT
-- All filling must be logged here, never overwrite ohlcv_raw
```

---

## 4. Data Validation Specification

Per audit #11 tests required:

### Checks to Implement in `validator.py`:

1. **Duplicate Candles**
   - **Spec:** `(exchange, symbol, timeframe, timestamp_ms)` must be unique
   - **Detection:** `SELECT timestamp_ms, COUNT(*) FROM ohlcv_raw GROUP BY ... HAVING COUNT>1` - should be 0 due to PRIMARY KEY, but also check incoming batch for duplicates before insert
   - **Action:** Skip duplicate, log in `fetch_metadata.duplicates_skipped`, do NOT overwrite
   - **Test:** Insert same timestamp twice, ensure second is skipped, count unchanged

2. **Missing Candles**
   - **Spec:** For given timeframe, timestamps should be contiguous with step = timeframe_ms (e.g., 1h=3600000)
   - **Detection:** Sort by timestamp_ms ASC, compute diff, if diff > timeframe_ms and diff != 0 and diff % timeframe_ms !=0 or diff > timeframe_ms, then missing count = (diff / timeframe_ms) - 1
   - **Report:** Number of gaps and missing count, do NOT fill
   - **Action:** Store in `fetch_metadata.missing_candles_detected` and `validation_reports`
   - **Preserve:** Keep original data as-is, missing remains missing
   - **Test:** Create data with gap of 2*1h, detection should report 1 missing

3. **Out-of-Order Candles**
   - **Spec:** After sorting, timestamps must be strictly increasing
   - **Detection:** If incoming batch not sorted, sort it; if DB query returns unsorted (should not), flag
   - **Action:** Always store sorted ASC, but report if input was out-of-order
   - **Test:** Provide batch [t3, t1, t2] shuffled, ensure stored sorted and out-of-order detected

4. **Invalid OHLC Relationships**
   - **Spec:** high >= max(open, close, low), low <= min(open, close, high), high >= low, open>0, high>0, low>0, close>0, volume>=0, high!=0 etc.
   - **Detection:** For each candle: check `high < low` or `high < open` or `high < close` or `low > open` etc.
   - **Action:** Reject invalid candle, log error, do not insert
   - **Test:** Provide candle high=100 low=200 invalid, ensure rejected

5. **Timezone Conversion**
   - **Spec:** All timestamps normalized to UTC - both ms (already UTC from Binance) and ISO UTC
   - **Detection:** Convert timestamp_ms to UTC ISO via `pd.to_datetime(ms, unit='ms', utc=True)`, ensure resulting string ends with +00:00 or Z, no local timezone
   - **Action:** Always store UTC, no local
   - **Test:** Input ms known to be 2023-01-01 00:00 UTC, ensures iso is 2023-01-01T00:00:00+00:00

6. **Pagination Boundaries**
   - **Spec:** Binance limit 1000 per request. For ranges >1000, need pagination via `since` parameter
   - **Detection:** Test fetcher with limit=2 for small range that needs 3 pages, ensure all candles returned, no gaps at page boundaries, last timestamp of page N < first of page N+1
   - **Action:** Paginate using `since = last_timestamp_ms + timeframe_ms`
   - **Test:** Mock exchange returning 2 per page, fetch 5 candles, ensure pagination logic returns 5, not 4 or duplicate at boundary

7. **Repeated Fetches Producing No Duplicate Rows**
   - **Spec:** Idempotency - fetching same range twice should insert 0 new rows second time
   - **Detection:** Fetch range A, count inserted N. Fetch same range again, count inserted should be 0, duplicates_skipped = N
   - **Action:** Use INSERT OR IGNORE / ON CONFLICT DO NOTHING
   - **Test:** Call `storage.insert_ohlcv(batch)` twice with same batch, second should report inserted=0

8. **Incremental Updates**
   - **Spec:** After initial full history, subsequent fetches should start from last stored timestamp + timeframe_ms, deterministic, no gaps or overlaps
   - **Detection:** Get last timestamp from DB, fetch from last+tf to now, ensure new data contiguous with old
   - **Action:** `get_last_timestamp_ms(symbol, timeframe)` then fetch since that
   - **Test:** Insert 10 candles t1..t10, then incremental fetch t11..t15, ensure DB has t1..t15 contiguous, no duplicate t10

### Additional Validation:

- **Timestamp Normalization:** Ensure timestamp_ms % 1000 == 0 (Binance ms are second-aligned), and minute aligns with timeframe (e.g., 1h candles have minute 0)
- **Volume Sanity:** volume >=0, not NaN, finite
- **Source Tracking:** Every row has `source='binance_ccxt_public'`

#### 9. Database Ordering (Hardening #2)

**Spec:** Do NOT rely on SQLite's natural/table order for chronological data. Every model-data query MUST explicitly use `ORDER BY timestamp_ms ASC`.

**Rationale:** SQLite does not guarantee row order without ORDER BY. Natural order may be insertion order or index order, not chronological. For Kronos (time-series foundation model), chronological order is critical - out-of-order input would cause data leakage and wrong predictions.

**Implementation:**
- All SELECTs in `storage.py` must include `ORDER BY timestamp_ms ASC` (or DESC if explicitly needed)
- Example: `SELECT ... FROM ohlcv_raw WHERE ... ORDER BY timestamp_ms ASC`
- Never use `SELECT * FROM ohlcv_raw` without ORDER BY for model input
- CSV export must be sorted ASC before writing

**Test:** `test_database_ordering_chronological` - Insert candles out-of-order, retrieve via `get_candles`, verify returned list is sorted ASC by timestamp_ms, and that raw SELECT without ORDER BY is NOT used.

#### 10. UTC Daily Candles (Hardening #3)

**Spec:** Document and test that Binance candle timestamps remain exchange-provided UTC timestamps. Do NOT reconstruct daily candles using India/local timezone boundaries.

**Binance Behavior:**
- Binance `1d` klines: `timestamp_ms` is open time at 00:00:00 UTC - e.g., daily candle for 2023-07-22 00:00 UTC has timestamp 2023-07-22 00:00:00 UTC, not 05:30 IST (India).
- `1h` candles: open at UTC hour boundaries (00:00,01:00,02:00... UTC)
- `4h` candles: open at 00:00,04:00,08:00,12:00,16:00,20:00 UTC

**Never Do:**
- Convert timestamp_ms to local IST via `datetime.fromtimestamp()` without tz - would shift daily boundary to 05:30 IST
- Reconstruct daily candles by grouping 1h candles using local day boundaries - must use exchange-provided 1d candles directly, which are already UTC.

**Correct:**
- Keep `timestamp_ms` exactly as returned by Binance (UTC)
- Derive `timestamp_utc` via `datetime.fromtimestamp(ms/1000, tz=timezone.utc).isoformat()` - explicitly UTC
- For any grouping, use UTC boundaries only

**Test:** `test_utc_daily_candles` - Given known daily timestamp `1672531200000` (2023-01-01 00:00 UTC), ensure `ms_to_iso` produces `2023-01-01T00:00:00+00:00` not `2023-01-01T05:30:00+05:30`. Ensure conversion does NOT use local timezone `fromtimestamp()` without tz.

#### 11. Data Leakage Test (Hardening #4 - Critical for Phase 3)

**Spec:** For every prediction timestamp T:
- Model input may contain only timestamps <= T
- Prediction targets must be strictly after T (T+1, T+2, ...)
- No future candle may enter feature/input window

**Definition:**
- Let `input_window = [T-lookback+1, ..., T]` inclusive of T
- Let `target_window = [T+1, ..., T+pred_len]` exclusive of T
- Then `max(input_window) < min(target_window)` and `max(input_window) == T`

**Reusable Validation Function (for Phase 3/backtesting):**
```python
def validate_no_future_leakage(input_timestamps: List[int], target_timestamps: List[int], prediction_time_ms: int) -> bool:
    # input must be <= prediction_time
    # target must be > prediction_time
    # No overlap
```

**Implementation in `validator.py`:**
- `check_no_future_leakage(input_candles, target_candles, prediction_time_ms)` returns bool + details
- Also `validate_backtest_split(train_end_ms, test_start_ms)` ensures no overlap

**Test:** `test_data_leakage` - Create input [t1,t2,t3] with T=t3, target [t4,t5] - valid. Then create leaking case input contains t4 (future) - should be detected as leakage.

#### 12. Multi-Timeframe Alignment (Hardening #5)

**Spec:** Document exactly how prediction at time T selects 1h, 4h, 1d history. Do NOT forward-fill future higher-timeframe candles. Use only candles that were actually closed/available at T.

**Definitions:**
- **T**: Prediction time, timestamp_ms of last closed candle on primary timeframe (1h)
- **1h history**: 1h candles with `timestamp_ms <= T`, sorted ASC, last candle is T
- **4h history**: 4h candles with `timestamp_ms <= T` and closed at or before T. Since 4h candles close every 4h at UTC boundaries (00,04,08,12,16,20), the latest 4h candle available at T is the 4h candle whose open <= T and whose close (open+4h-1ms) <= T? Actually Binance 4h candle with open time O is considered closed at O+4h. For safety, use condition `open_ms + timeframe_ms <= T + timeframe_ms`? Simpler: `open_ms <= T` and `open_ms + timeframe_ms <= T + 1h`? Need precise.
- **1d history**: 1d candles with `open_ms <= T` and closed before T (open +1d <= T+1h)

**Precise Rule (Conservative, No Forward-Fill):**
- For any timeframe TF with ms = tf_ms, a TF candle with open time O is considered available at prediction time T (which is open of next 1h candle? Actually T is last closed 1h candle open) if `O + tf_ms <= T + tf_ms_primary`? Let's define simpler and safer:
- **Available at T if O <= T** - This ensures candle started at or before T. However, if T is in middle of 4h candle (e.g., T=01:00 UTC, 4h candle opened 00:00, closes 04:00), that 4h candle is still forming, not closed. So for training/backtesting, we should use only closed candles.
- **Closed at T if O + tf_ms <= T + tf_ms_primary**? Let's think.

**Simpler Conservative Rule for Phase 2 Spec (Used in Phase 3):**
- **1h history:** candles where `timestamp_ms <= T` (closed, since 1h candle open T is last closed, next would be T+1h)
- **4h history:** candles where `timestamp_ms + 4h <= T + 1h` (i.e., 4h candle closed at or before T's close). Since T is open of last closed 1h, its close is T+1h. So 4h candle must close <= T+1h.
- **1d history:** candles where `timestamp_ms + 1d <= T + 1h`

**Example:**
- T = 2023-07-22 00:00 UTC (open of 1h candle that closed at 01:00? Actually open 00:00, close 01:00, but we consider T as open time of last candle, which is closed when we predict? For training, T is last known open.)
- 1h history: includes candle open 2023-07-22 00:00 (T), plus previous 399 candles
- 4h history: 4h candles open at 2023-07-21 20:00, 16:00, etc. - latest that satisfies O+4h <= T+1h = 2023-07-22 01:00. So candle open 2023-07-21 20:00 closes 00:00 UTC 2023-07-22, which is <=01:00, so available. Candle open 2023-07-22 00:00 closes 04:00, which is >01:00, not yet closed, so NOT available at T. So we use 20:00 4h candle as latest.
- 1d history: daily candle open 2023-07-21 00:00 closes 2023-07-22 00:00, which <=01:00, so available. Daily candle open 2023-07-22 00:00 closes 23:00 next day? Actually daily close is next day 00:00, so 2023-07-22 daily closes 2023-07-23 00:00 >01:00, not available.

**Do NOT Forward-Fill Future:**
- Never use 4h candle open 00:00 at time T=00:00 if it closes at 04:00 - that would be future info (still forming). Only use closed higher TF candles.
- Implementation in Phase 3 will have function `get_aligned_history(symbol, T, lookback)` that queries DB with `WHERE timestamp_ms + tf_ms <= T + primary_tf_ms` for each TF.

**Test:** `test_multi_timeframe_alignment` - At T=2023-07-22 00:00 UTC, with 1h last open T, 4h candles available should be those with open <=2023-07-21 20:00, not 00:00.

#### 13. Incomplete Current Candle (Hardening #6)

**Spec:** Distinguish currently forming candle from closed candle. For training/backtesting: ONLY CLOSED candles unless component explicitly requests otherwise.

**Binance Candle Lifecycle:**
- Candle with open time O, timeframe TF: open at O, close at O+TF-1ms (e.g., 1h candle open 00:00 closes 00:59:59.999)
- At current time `now_ms`, candle with O <= now_ms < O+TF is still forming (incomplete)
- Candle with O+TF <= now_ms is closed (complete)

**Fetcher Handling:**
- `fetch_ohlcv_range` may return incomplete current candle if `until_ms` is now and now is mid-candle
- For historical backtest/training, we should exclude incomplete candle: filter `timestamp_ms + tf_ms <= now_ms`
- Add parameter `include_incomplete: bool = False` default False for training. If True (for live prediction), include incomplete as latest.

**Implementation:**
```python
def is_closed_candle(open_ms, timeframe_ms, now_ms):
    return open_ms + timeframe_ms <= now_ms

def filter_closed(candles, timeframe_ms, now_ms):
    return [c for c in candles if c[0] + timeframe_ms <= now_ms]
```

**Model Pipeline:**
- Training/backtesting data: ONLY closed candles
- Live prediction: may include incomplete current candle as additional info, but must be flagged as incomplete and not used as target

**Test:** `test_incomplete_current_candle` - At now=00:30 UTC, 1h candle open 00:00 is incomplete (close 01:00 > now), should be excluded when include_incomplete=False, included when True.

---

## Updated Test Plan (Hardening Pass)

Previous 13 tests plus new:

14. `test_database_ordering_chronological` - Insert out-of-order, retrieve via get_candles, verify ORDER BY ASC
15. `test_utc_daily_candles` - Known daily timestamp remains UTC, not IST
16. `test_data_leakage` - Input <=T, target >T, no future in input
17. `test_multi_timeframe_alignment` - At T, 4h latest closed is 20:00 previous day, not 00:00 same day
18. `test_incomplete_current_candle` - Incomplete current candle excluded for training

Total 18 tests.


---

## 5. Fetcher Specification (CCXT Public)

**File:** `scripts/data/fetcher.py`

**No API Keys Required:**

```python
import ccxt
exchange = ccxt.binance({
    'enableRateLimit': True,  # CCXT handles rate limit
    # No apiKey/secret - public data only per requirement #3
    'options': {'defaultType': 'spot'}
})
# Public fetch: exchange.fetch_ohlcv('BTC/USDT', '1h', since, limit)
```

**Pagination:**

- Binance `fetch_ohlcv` param `since` is timestamp_ms, `limit` max 1000
- Loop:
  ```
  all_candles = []
  current_since = since_ms
  while current_since < until_ms:
      batch = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
      if not batch: break
      all_candles.extend(batch)
      last_ts = batch[-1][0]
      current_since = last_ts + timeframe_ms
      if len(batch) < 1000: break  # Last page
  ```

**Rate-Limit Handling:**

- `enableRateLimit=True` - CCXT sleeps per exchange's rateLimit (Binance ~ 100ms)
- On `ccxt.errors.RateLimitExceeded` or `DDoSProtection`: exponential backoff retry
- On `NetworkError`: retry 5 times with backoff 1s,2s,4s,8s,16s

**Retries with Exponential Backoff:**

```python
for attempt in range(max_retries):
    try:
        return exchange.fetch_ohlcv(...)
    except (ccxt.NetworkError, ccxt.RateLimitExceeded, ccxt.ExchangeNotAvailable) as e:
        if attempt == max_retries-1: raise
        sleep = backoff_base * (2 ** attempt) + random jitter
        time.sleep(sleep)
```

**Duplicate Detection:**

- Before insert, check batch itself for duplicate timestamp_ms - if duplicate in batch, keep first, log warning
- At DB level, PRIMARY KEY prevents duplicate - use INSERT OR IGNORE

**Timestamp Normalization to UTC:**

- Binance returns timestamp in UTC ms - already UTC
- Store as integer ms + ISO UTC string via `pd.to_datetime(ts, unit='ms', utc=True).isoformat()`
- All internal handling in UTC, no local timezone

**Deterministic Incremental Updates:**

- `storage.get_date_range(symbol, timeframe)` returns (first_ms, last_ms)
- If DB empty: fetch full history (e.g., 2 years per config)
- If DB has data: `since = last_ms + timeframe_ms`, fetch to now
- Deterministic: same inputs produce same DB state, no randomness, no silent filling

**CSV Export:**

- After each successful fetch+insert, export `data/raw/binance_{symbol}_{timeframe}.csv` sorted ASC
- CSV is derived from SQLite, not separate source - ensures consistency

---

## 6. Exchange Fees/Slippage Metadata (Separate from Raw)

Per requirement #10: Include fees/slippage metadata in architecture, but don't mix into raw table.

**Location:** `config/config.yaml` - already has:

```yaml
strategy:
  fee_pct: 0.001  # Binance 0.1% spot
  slippage_pct: 0.001
broker:
  binance:
    fees:
      spot_maker: 0.001
      spot_taker: 0.001
      note: "Actual fees depend on BNB discount and volume - this is conservative assumption for backtesting, not stored in raw OHLCV"
```

Raw table `ohlcv_raw` does NOT contain fee columns. Fees applied later in backtesting/execution layer (Phase 6/7).

---

## 7. Test Plan (Per Requirement #11)

**File:** `tests/test_data_pipeline.py`

Tests use mocked CCXT exchange and in-memory SQLite to avoid network dependency, plus one optional integration test that tries public fetch (skipped if network blocked 451).

**Test Cases:**

1. `test_duplicate_candles` - Insert batch with duplicate timestamp, ensure only 1 inserted, second counted as duplicate
2. `test_missing_candles_detection` - Create DB with gap, run validator, expect missing count >0, report details
3. `test_out_of_order_candles` - Provide batch unsorted, ensure storage sorts and validator flags out-of-order
4. `test_invalid_ohlc_high_low` - High < Low should be rejected
5. `test_invalid_ohlc_high_open` - High < Open should be rejected
6. `test_timezone_conversion` - Known ms to ISO UTC conversion, ensure UTC not local
7. `test_pagination_boundaries` - Mock exchange returning 2 per page, fetch 5, ensure all 5 and no duplicate at page boundary
8. `test_repeated_fetches_no_duplicates` - Insert same batch twice, second insert count 0
9. `test_incremental_updates` - Insert t1-t10, then incremental t11-t15, verify t1-t15 contiguous
10. `test_deterministic_csv_export` - Export twice, ensure CSV identical
11. `test_ohlcv_sanity_volume_negative` - Volume negative rejected
12. `test_timestamp_normalization_utc` - Ensure timestamp_ms unchanged, iso UTC ends with +00:00/Z

**Run:**

```bash
pytest tests/test_data_pipeline.py -v
```

---

## 8. Files to Create in Phase 2

- `config/data_schema.yaml` - machine-readable schema spec
- `docs/DATA_SCHEMA.md` - this file (human-readable)
- `scripts/data/fetcher.py` - CCXT public fetcher with pagination, retries, backoff
- `scripts/data/validator.py` - validation checks per spec
- `scripts/data/storage.py` - SQLite storage, CSV export, incremental logic
- `scripts/data/run_fetch.py` - CLI orchestrator for BTC/ETH 1h/4h/1d
- `tests/test_data_pipeline.py` - tests per requirement #11
- `data/raw/` - CSV exports (generated)
- `data/db/kronos_trading.db` - SQLite DB (generated)

**Not in Phase 2 (deferred per requirement #7):**
- No websocket streaming
- No real-time infra
- No filling/interpolation of missing candles
- No prediction/backtesting

---

## 9. Expected Data Volume (BTC/USDT, ETH/USDT, 1h/4h/1d)

- **1h:** 24 candles/day, 730 days = ~17,520 per asset, 2 assets = 35k rows
- **4h:** 6 candles/day, 730 days = ~4,380 per asset, 2 assets = 8.7k rows
- **1d:** 1 candle/day, 730 days = 730 per asset, 2 assets = 1.4k rows
- **Total:** ~45k rows raw - lightweight, SQLite easily handles

Date range actually available depends on Binance history (BTC from 2017, but we request 730 days per config).

---

## 10. Security & LIVE Guard

- Phase 2 does NOT need API keys - public data only
- LIVE trading completely disabled - guard remains in `trading_mode_guard.py`
- No secrets logged
- Kronos upstream untouched

---

## 11. Final Report Requirements (Per Phase 2 End)

At end of Phase 2, report:

A. files created/modified
B. database schema (actual CREATE TABLE)
C. exact commands
D. number of candles downloaded per asset/timeframe (measured)
E. date range actually available (measured per asset)
F. missing-data statistics (measured gaps)
G. validation results (measured per check)
H. test results (pytest -v)
I. known limitations (e.g., Binance 451 in sandbox, rate limits, etc.)

Do not start Phase 3 yet.
