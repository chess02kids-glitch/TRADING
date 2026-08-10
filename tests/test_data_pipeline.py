#!/usr/bin/env python3
"""
Phase 2 - Tests for Data Pipeline
Per requirement #11:
- duplicate candles
- missing candles
- out-of-order candles
- invalid OHLC relationships
- timezone conversion
- pagination boundaries
- repeated fetches producing no duplicate rows
- incremental updates

No API keys required, uses mocked exchange and in-memory SQLite
"""

import sys
import pytest
from pathlib import Path
import time
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "Kronos"))

from fetcher import BinancePublicFetcher, TIMEFRAME_MS
from validator import DataValidator
from storage import SQLiteStorage

# ========== Helper Mocks ==========

class MockExchange:
    """Mock CCXT exchange for pagination and deterministic tests"""
    def __init__(self, all_candles, limit_default=2):
        # all_candles sorted ASC
        self.all_candles = sorted(all_candles, key=lambda x: x[0])
        self.limit_default = limit_default
        self.calls = []
    
    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls.append((symbol, timeframe, since, limit))
        if since is None:
            since = self.all_candles[0][0] if self.all_candles else 0
        
        # Filter candles >= since
        filtered = [c for c in self.all_candles if c[0] >= since]
        
        # Apply limit
        effective_limit = limit if limit is not None else self.limit_default
        batch = filtered[:effective_limit]
        return batch

# ========== Tests ==========

def test_duplicate_candles():
    """Test duplicate detection - repeated fetches should produce no duplicate rows per #11"""
    print("\n=== Test: duplicate candles ===")
    validator = DataValidator("1h")
    
    candles_with_dup = [
        [1000, 100, 110, 90, 105, 10],
        [1000 + 3600000, 105, 115, 100, 110, 12],
        [1000, 100, 110, 90, 105, 10],  # Duplicate timestamp
    ]
    
    is_valid, dup_count, dups = validator.check_duplicate_candles(candles_with_dup)
    assert not is_valid, "Should detect duplicate"
    assert dup_count == 1, f"Expected 1 duplicate, got {dup_count}"
    assert dups == [1000], f"Expected duplicate ts 1000, got {dups}"
    
    # Test storage duplicate handling - idempotent
    storage = SQLiteStorage(db_path=Path(":memory:"))
    # Use file for in-memory? sqlite :memory: needs special handling - use temp file
    import tempfile
    import os
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()
    storage = SQLiteStorage(db_path=Path(temp_db.name))
    
    candles = [
        [1000, 100, 110, 90, 105, 10],
        [1000 + 3600000, 105, 115, 100, 110, 12],
    ]
    
    inserted1, dups1 = storage.insert_ohlcv("binance", "BTC/USDT", "1h", candles)
    assert inserted1 == 2, f"First insert should insert 2, got {inserted1}"
    
    inserted2, dups2 = storage.insert_ohlcv("binance", "BTC/USDT", "1h", candles)
    assert inserted2 == 0, f"Second insert (repeated fetch) should insert 0, got {inserted2} - must be idempotent per requirement"
    assert dups2 >= 2, f"Second insert should report duplicates skipped"
    
    count = storage.count_candles("BTC/USDT", "1h")
    assert count == 2, f"Count should still be 2 after duplicate insert, got {count}"
    
    storage.close()
    os.unlink(temp_db.name)
    print("✓ duplicate candles test passed - repeated fetches produce no duplicate rows")

def test_missing_candles():
    """Test missing candle detection per #9 - detect and report, no silent filling"""
    print("\n=== Test: missing candles ===")
    validator = DataValidator("1h")
    
    # 1h timeframe = 3600000 ms
    # Create data with gap: t0, t0+1h, t0+3h (missing t0+2h)
    base = 1000000000000
    candles_with_gap = [
        [base, 100, 110, 90, 105, 10],
        [base + 3600000, 105, 115, 100, 110, 12],
        [base + 3*3600000, 110, 120, 105, 115, 15],  # Gap: missing base+2h
    ]
    
    is_valid, missing_count, gaps = validator.check_missing_candles(candles_with_gap)
    assert not is_valid, "Should detect missing"
    assert missing_count == 1, f"Expected 1 missing, got {missing_count}"
    assert len(gaps) == 1, f"Expected 1 gap, got {len(gaps)}"
    assert gaps[0]['missing_count'] == 1
    
    # Ensure no filling - original data preserved
    assert len(candles_with_gap) == 3, "Original data should not be filled silently per #9"
    
    print(f"✓ missing candles test passed - detected {missing_count} missing, no silent filling")

def test_out_of_order_candles():
    """Test out-of-order detection"""
    print("\n=== Test: out-of-order candles ===")
    validator = DataValidator("1h")
    
    base = 1000000000000
    # Out of order: t2, t0, t1
    out_of_order = [
        [base + 2*3600000, 110, 120, 105, 115, 15],
        [base, 100, 110, 90, 105, 10],
        [base + 3600000, 105, 115, 100, 110, 12],
    ]
    
    is_sorted, was_out_of_order = validator.check_out_of_order(out_of_order)
    assert not is_sorted, "Should detect out of order"
    assert was_out_of_order, "Should flag was out of order"
    
    # Storage should store sorted
    import tempfile, os
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()
    storage = SQLiteStorage(db_path=Path(temp_db.name))
    
    inserted, _ = storage.insert_ohlcv("binance", "BTC/USDT", "1h", out_of_order)
    assert inserted == 3
    
    retrieved = storage.get_candles("BTC/USDT", "1h")
    timestamps = [r[3] for r in retrieved]
    assert timestamps == sorted(timestamps), "Storage should return sorted ASC"
    
    storage.close()
    os.unlink(temp_db.name)
    print("✓ out-of-order test passed - detected and stored sorted")

def test_invalid_ohlc_high_low():
    """Test invalid OHLC: high < low"""
    print("\n=== Test: invalid OHLC high < low ===")
    validator = DataValidator("1h")
    
    invalid = [
        [1000, 100, 90, 100, 105, 10],  # high 90 < low 100
    ]
    
    is_valid, invalid_count, details = validator.check_invalid_ohlc(invalid)
    assert not is_valid
    assert invalid_count == 1
    assert "high < low" in details[0]['issues'][0]
    
    print("✓ invalid OHLC high<low test passed")

def test_invalid_ohlc_high_open():
    """Test invalid OHLC: high < open"""
    print("\n=== Test: invalid OHLC high < open ===")
    validator = DataValidator("1h")
    
    invalid = [
        [1000, 110, 100, 90, 105, 10],  # high 100 < open 110
    ]
    
    is_valid, invalid_count, details = validator.check_invalid_ohlc(invalid)
    assert not is_valid
    assert invalid_count == 1
    assert any("high < open" in issue for issue in details[0]['issues'])
    
    print("✓ invalid OHLC high<open test passed")

def test_timezone_conversion():
    """Test timezone conversion to UTC per #11"""
    print("\n=== Test: timezone conversion ===")
    validator = DataValidator("1h")
    
    # Known timestamp: 2023-01-01 00:00:00 UTC = 1672531200000 ms
    known_ms = 1672531200000
    is_valid, iso = validator.check_timezone_conversion(known_ms)
    
    assert is_valid, f"Timezone should be valid, got {iso}"
    assert "2023-01-01" in iso, f"ISO should contain 2023-01-01, got {iso}"
    # Should be UTC
    assert "+00:00" in iso or iso.endswith("Z") or "00:00" in iso
    
    # Test ms_to_iso in fetcher
    fetcher = BinancePublicFetcher()
    iso_from_fetcher = fetcher.ms_to_iso(known_ms)
    assert "2023-01-01" in iso_from_fetcher
    
    print(f"✓ timezone conversion test passed - {known_ms} -> {iso}")

def test_pagination_boundaries():
    """Test pagination boundaries per #11"""
    print("\n=== Test: pagination boundaries ===")
    
    # Create 5 candles
    base = 1000000000000
    all_candles = [
        [base + i*3600000, 100+i, 110+i, 90+i, 105+i, 10+i]
        for i in range(5)
    ]
    
    # Mock exchange returning 2 per page
    mock_exchange = MockExchange(all_candles, limit_default=2)
    
    fetcher = BinancePublicFetcher()
    fetcher.exchange = mock_exchange  # Inject mock
    
    # Fetch with limit 2, should paginate 3 pages: 2+2+1=5
    fetched = fetcher.fetch_ohlcv_range(
        symbol="BTC/USDT",
        timeframe="1h",
        since_ms=base,
        until_ms=base + 5*3600000,
        limit_per_request=2,
        max_retries=1
    )
    
    assert len(fetched) == 5, f"Expected 5 candles via pagination, got {len(fetched)}"
    
    # Check no duplicate at page boundaries
    timestamps = [c[0] for c in fetched]
    assert len(timestamps) == len(set(timestamps)), "Pagination should not produce duplicates at boundaries"
    
    # Check calls - should be 3 pages
    assert len(mock_exchange.calls) == 3, f"Expected 3 pages, got {len(mock_exchange.calls)} calls"
    
    # Check since param advances correctly: page1 since=base, page2 since=base+2h + 1h = base+3h? Let's verify
    # Our logic: since = last_ts + timeframe_ms
    # Page1: since=base, returns base, base+1h, last=base+1h, next since=base+2h
    # Page2: since=base+2h, returns base+2h, base+3h, last=base+3h, next since=base+4h
    # Page3: since=base+4h, returns base+4h
    # So calls should be base, base+2h, base+4h
    expected_since = [base, base+2*3600000, base+4*3600000]
    actual_since = [call[2] for call in mock_exchange.calls]
    assert actual_since == expected_since, f"Pagination since mismatch, expected {expected_since}, got {actual_since}"
    
    print("✓ pagination boundaries test passed - 5 candles via 3 pages, no duplicates at boundaries")

def test_repeated_fetches_no_duplicates():
    """Test repeated fetches produce no duplicate rows per #11"""
    print("\n=== Test: repeated fetches no duplicates ===")
    import tempfile, os
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()
    storage = SQLiteStorage(db_path=Path(temp_db.name))
    
    base = 1000000000000
    candles = [
        [base + i*3600000, 100+i, 110+i, 90+i, 105+i, 10+i]
        for i in range(3)
    ]
    
    # First fetch
    inserted1, dups1 = storage.insert_ohlcv("binance", "BTC/USDT", "1h", candles)
    assert inserted1 == 3
    count1 = storage.count_candles("BTC/USDT", "1h")
    
    # Second fetch same range
    inserted2, dups2 = storage.insert_ohlcv("binance", "BTC/USDT", "1h", candles)
    assert inserted2 == 0, "Repeated fetch should insert 0"
    count2 = storage.count_candles("BTC/USDT", "1h")
    assert count2 == count1 == 3, "Count should remain same after repeated fetch"
    
    storage.close()
    os.unlink(temp_db.name)
    print("✓ repeated fetches no duplicates test passed - idempotent")

def test_incremental_updates():
    """Test incremental updates per #11"""
    print("\n=== Test: incremental updates ===")
    import tempfile, os
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()
    storage = SQLiteStorage(db_path=Path(temp_db.name))
    
    base = 1000000000000
    
    # Initial 10 candles t1..t10
    initial = [
        [base + i*3600000, 100+i, 110+i, 90+i, 105+i, 10+i]
        for i in range(10)
    ]
    inserted, _ = storage.insert_ohlcv("binance", "BTC/USDT", "1h", initial)
    assert inserted == 10
    
    last_ms = storage.get_last_timestamp_ms("BTC/USDT", "1h")
    assert last_ms == base + 9*3600000
    
    # Incremental: t11..t15
    incremental = [
        [base + i*3600000, 100+i, 110+i, 90+i, 105+i, 10+i]
        for i in range(10, 15)
    ]
    
    # Simulate fetcher logic: since = last_ms + timeframe_ms
    tf_ms = TIMEFRAME_MS["1h"]
    since_for_incremental = last_ms + tf_ms
    assert since_for_incremental == base + 10*3600000
    
    inserted_inc, _ = storage.insert_ohlcv("binance", "BTC/USDT", "1h", incremental)
    assert inserted_inc == 5
    
    # Verify DB has t1..t15 contiguous
    all_candles = storage.get_candles("BTC/USDT", "1h")
    assert len(all_candles) == 15
    timestamps = [r[3] for r in all_candles]
    expected = [base + i*3600000 for i in range(15)]
    assert timestamps == expected, f"Incremental should produce contiguous t1..t15, got {timestamps}"
    
    # Check no gaps
    validator = DataValidator("1h")
    # Convert DB rows to candle format for validator
    candles_for_validation = [[r[3], r[5], r[6], r[7], r[8], r[9]] for r in all_candles]
    is_valid_missing, missing_count, _ = validator.check_missing_candles(candles_for_validation)
    assert missing_count == 0, f"After incremental, should have 0 missing, got {missing_count}"
    
    storage.close()
    os.unlink(temp_db.name)
    print("✓ incremental updates test passed - deterministic, contiguous, no gaps")

def test_deterministic_csv_export():
    """Test CSV export deterministic"""
    print("\n=== Test: deterministic CSV export ===")
    import tempfile, os
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()
    temp_dir = tempfile.mkdtemp()
    
    storage = SQLiteStorage(db_path=Path(temp_db.name))
    
    base = 1000000000000
    candles = [
        [base + i*3600000, 100+i, 110+i, 90+i, 105+i, 10+i]
        for i in range(5)
    ]
    storage.insert_ohlcv("binance", "BTC/USDT", "1h", candles)
    
    csv1 = storage.export_csv("BTC/USDT", "1h", output_dir=Path(temp_dir))
    with open(csv1) as f:
        content1 = f.read()
    
    time.sleep(0.1)
    
    csv2 = storage.export_csv("BTC/USDT", "1h", output_dir=Path(temp_dir))
    with open(csv2) as f:
        content2 = f.read()
    
    assert content1 == content2, "CSV export should be deterministic"
    
    storage.close()
    os.unlink(temp_db.name)
    import shutil
    shutil.rmtree(temp_dir)
    print("✓ deterministic CSV export test passed")

def test_volume_negative():
    """Test volume negative rejected"""
    print("\n=== Test: volume negative ===")
    validator = DataValidator("1h")
    
    invalid = [
        [1000, 100, 110, 90, 105, -5],  # volume negative
    ]
    
    is_valid, invalid_count, details = validator.check_invalid_ohlc(invalid)
    assert not is_valid
    assert invalid_count == 1
    
    print("✓ volume negative test passed")

def test_timestamp_normalization_utc():
    """Test timestamp normalization UTC per #11"""
    print("\n=== Test: timestamp normalization UTC ===")
    fetcher = BinancePublicFetcher()
    
    # Binance returns ms already UTC, ensure conversion produces UTC ISO
    test_ms = 1672531200000  # 2023-01-01 00:00 UTC
    iso = fetcher.ms_to_iso(test_ms)
    
    # Should be UTC
    assert "2023-01-01" in iso
    # Convert back
    ms_back = fetcher.iso_to_ms(iso)
    assert ms_back == test_ms, f"Roundtrip ms->iso->ms should preserve, got {ms_back} != {test_ms}"
    
    print(f"✓ timestamp normalization UTC test passed - {test_ms} <-> {iso}")

def test_public_fetch_no_api_keys():
    """Ensure fetcher works without API keys per requirement #3"""
    print("\n=== Test: public fetch without API keys ===")
    try:
        fetcher = BinancePublicFetcher()
        # Should not have apiKey
        assert not hasattr(fetcher.exchange, 'apiKey') or fetcher.exchange.apiKey is None or fetcher.exchange.apiKey == '', "Fetcher should not require API keys for public data"
        
        # Config should not contain apiKey
        # This is check that our fetcher __init__ doesn't set apiKey
        print("✓ public fetch no API keys test passed - CCXT config has no apiKey")
        return True
    except Exception as e:
        print(f"❌ public fetch no keys test failed: {e}")
        raise

# ========== Hardening Pass Tests (Requirements from Final Hardening) ==========

def test_database_ordering_chronological():
    """
    Hardening #2: Every model-data query must explicitly use ORDER BY timestamp_ms ASC
    Test verifies retrieved candles are chronological
    """
    print("\n=== Test: database ordering chronological (Hardening #2) ===")
    import tempfile, os
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()
    storage = SQLiteStorage(db_path=Path(temp_db.name))
    
    base = 1000000000000
    # Insert out-of-order intentionally
    out_of_order = [
        [base + 2*3600000, 110, 120, 105, 115, 15],
        [base, 100, 110, 90, 105, 10],
        [base + 3600000, 105, 115, 100, 110, 12],
        [base + 3*3600000, 115, 125, 110, 120, 20],
    ]
    
    inserted, _ = storage.insert_ohlcv("binance", "BTC/USDT", "1h", out_of_order)
    assert inserted == 4
    
    # Retrieve via get_candles which MUST have ORDER BY ASC
    retrieved = storage.get_candles("BTC/USDT", "1h")
    timestamps = [r[3] for r in retrieved]
    
    # Verify chronological ASC
    assert timestamps == sorted(timestamps), f"Retrieved candles must be chronological ASC due to ORDER BY, got {timestamps}"
    assert timestamps[0] == base
    assert timestamps[-1] == base + 3*3600000
    
    # Also test get_candles_for_model which explicitly documents ASC
    retrieved_model = storage.get_candles_for_model("BTC/USDT", "1h")
    timestamps_model = [r[3] for r in retrieved_model]
    assert timestamps_model == sorted(timestamps_model), "get_candles_for_model must be chronological ASC"
    
    # Validate via validator's database ordering check
    validator = DataValidator("1h")
    is_chrono, out_of_order_idx = validator.check_database_ordering(retrieved)
    assert is_chrono, f"Database ordering check should pass for ORDER BY ASC data, got out-of-order indices {out_of_order_idx}"
    
    storage.close()
    os.unlink(temp_db.name)
    print("✓ database ordering chronological test passed - ORDER BY ASC enforced")

def test_utc_daily_candles():
    """
    Hardening #3: Binance candle timestamps remain UTC, not reconstructed via local timezone
    Document and test that daily candles stay UTC
    """
    print("\n=== Test: UTC daily candles (Hardening #3) ===")
    validator = DataValidator("1d")
    fetcher = BinancePublicFetcher()
    
    # Known Binance daily candle: 2023-01-01 00:00:00 UTC = 1672531200000 ms
    # This timestamp should remain UTC, not become 05:30 IST
    known_daily_ms = 1672531200000
    
    # Test validator's UTC check
    is_valid, iso_utc, _ = validator.check_utc_daily_candles(known_daily_ms)
    assert is_valid, f"UTC daily check should be valid, got {iso_utc}"
    assert "2023-01-01" in iso_utc
    assert "+00:00" in iso_utc, f"ISO should be UTC +00:00, got {iso_utc}"
    
    # Ensure NOT using local timezone: if we used datetime.fromtimestamp without tz, in India it would be 05:30
    # Our method uses timezone.utc explicitly - test difference
    from datetime import datetime
    dt_local_naive = datetime.fromtimestamp(known_daily_ms / 1000)  # No tz - would be local
    dt_utc = datetime.fromtimestamp(known_daily_ms / 1000, tz=timezone.utc)
    
    # They should differ if local is IST (UTC+5:30) - but in sandbox they may be UTC anyway
    # Main check: our iso_utc must be from utc method
    assert dt_utc.tzinfo == timezone.utc
    
    # Test fetcher's ms_to_iso also uses UTC
    iso_from_fetcher = fetcher.ms_to_iso(known_daily_ms)
    assert "+00:00" in iso_from_fetcher or iso_from_fetcher.endswith("Z") or "00:00" in iso_from_fetcher
    assert "2023-01-01" in iso_from_fetcher
    
    # Ensure daily candle hour is 00:00 UTC
    # For 1d timeframe, Binance daily open is 00:00 UTC - our timestamp should preserve that
    # 1672531200000 -> 2023-01-01 00:00 UTC
    dt = datetime.fromtimestamp(known_daily_ms / 1000, tz=timezone.utc)
    assert dt.hour == 0 and dt.minute == 0 and dt.second == 0, f"Daily candle should be 00:00 UTC, got {dt}"
    
    print(f"✓ UTC daily candles test passed - {known_daily_ms} -> {iso_utc} remains UTC, not IST")

def test_data_leakage():
    """
    Hardening #4: Data leakage test - critical for Phase 3 and backtesting
    For every prediction timestamp T:
    - input <= T
    - target > T
    - no future in input
    """
    print("\n=== Test: data leakage (Hardening #4) ===")
    from validator import validate_no_future_leakage
    
    base = 1000000000000
    # T = base + 2h (third candle)
    T = base + 2*3600000
    
    # Valid case: input [t0,t1,t2=T], target [t3,t4]
    input_valid = [base, base+3600000, base+2*3600000]  # t0,t1,T
    target_valid = [base+3*3600000, base+4*3600000]  # t3,t4 > T
    is_valid, details = validate_no_future_leakage(input_valid, target_valid, T)
    assert is_valid, f"Valid case should pass, got issues {details['issues']}"
    assert not details['leakage_found']
    
    # Invalid: input contains future (t3)
    input_leak = [base, base+3600000, base+3*3600000]  # includes t3 which is > T
    target = [base+4*3600000]
    is_valid_leak, details_leak = validate_no_future_leakage(input_leak, target, T)
    assert not is_valid_leak, "Should detect future leakage in input"
    assert details_leak['leakage_found']
    assert any("Future leakage" in issue for issue in details_leak['issues'])
    
    # Invalid: target contains past or T itself
    input_ok = [base, base+3600000, base+2*3600000]
    target_invalid = [base+2*3600000, base+3*3600000]  # includes T itself, should be >T
    is_valid_target, details_target = validate_no_future_leakage(input_ok, target_invalid, T)
    assert not is_valid_target, "Should detect target <= T"
    
    # Invalid: overlap
    input_overlap = [base, base+3600000, base+2*3600000]
    target_overlap = [base+2*3600000, base+3*3600000]  # overlap at T
    is_valid_overlap, details_overlap = validate_no_future_leakage(input_overlap, target_overlap, T)
    assert not is_valid_overlap, "Should detect overlap"
    
    print("✓ data leakage test passed - reusable validation function works for Phase 3")

def test_multi_timeframe_alignment():
    """
    Hardening #5: Multi-timeframe alignment for 1h/4h/1d
    Document how prediction at time T selects history, no forward-fill future higher-TF
    """
    print("\n=== Test: multi-timeframe alignment (Hardening #5) ===")
    from validator import DataValidator, get_aligned_history_for_prediction
    import datetime
    
    # Example: T = 2023-07-22 00:00 UTC
    # 1h history should include T
    # 4h latest should be 2023-07-21 20:00 (closes 00:00), not 2023-07-22 00:00 (closes 04:00, still forming)
    T = 1689984000000  # 2023-07-22 00:00 UTC
    tf_1h = 3600000
    tf_4h = 14400000
    tf_1d = 86400000
    
    validator = DataValidator("1h")
    
    # Create 4h candles: 20:00 previous day, 00:00 same day, etc.
    # 20:00 previous day = T - 4h
    candle_4h_20_prev = [T - tf_4h, 100, 110, 90, 105, 10]  # 2023-07-21 20:00 open, closes 2023-07-22 00:00
    candle_4h_00_same = [T, 105, 115, 100, 110, 12]  # 2023-07-22 00:00 open, closes 04:00 - still forming at T
    
    # 1h candles
    candle_1h_T = [T, 100, 110, 90, 105, 10]
    candle_1h_prev = [T - tf_1h, 99, 109, 89, 104, 9]
    
    # 1d candles
    candle_1d_prev = [T - tf_1d, 95, 105, 85, 100, 100]  # 2023-07-21 00:00 open, closes 2023-07-22 00:00
    candle_1d_same = [T, 100, 110, 90, 105, 100]  # 2023-07-22 00:00 open, closes 2023-07-23 00:00 - not closed yet
    
    # Valid alignment: use 20:00 4h, not 00:00 4h
    valid_4h = [candle_4h_20_prev]
    valid_1d = [candle_1d_prev]
    
    is_valid, details = validator.check_multi_timeframe_alignment(
        prediction_time_ms=T,
        candles_1h=[candle_1h_prev, candle_1h_T],
        candles_4h=valid_4h,
        candles_1d=valid_1d
    )
    assert is_valid, f"Valid alignment should pass, got issues {details['issues']}"
    
    # Invalid: using future 4h candle (00:00 same day) that closes after T+1h
    invalid_4h = [candle_4h_00_same]  # This closes at 04:00, after T+1h=01:00, should be rejected
    is_valid_invalid, details_invalid = validator.check_multi_timeframe_alignment(
        prediction_time_ms=T,
        candles_1h=[candle_1h_prev, candle_1h_T],
        candles_4h=invalid_4h,
        candles_1d=valid_1d
    )
    assert not is_valid_invalid, "Should detect forward-fill of future 4h candle"
    assert any("forward-fill" in issue for issue in details_invalid['issues'])
    
    # Test reusable function get_aligned_history_for_prediction
    all_1h = [candle_1h_prev, candle_1h_T]
    all_4h = [candle_4h_20_prev, candle_4h_00_same]
    all_1d = [candle_1d_prev, candle_1d_same]
    
    aligned = get_aligned_history_for_prediction(
        prediction_time_ms=T,
        all_1h_candles=all_1h,
        all_4h_candles=all_4h,
        all_1d_candles=all_1d,
        lookback_1h=10,
        lookback_4h=10,
        lookback_1d=10
    )
    
    # Should include only closed 4h (20:00 prev), not 00:00 same
    assert len(aligned["4h"]) == 1, f"Aligned 4h should have 1 closed candle, got {len(aligned['4h'])}"
    assert aligned["4h"][0][0] == T - tf_4h, "Latest 4h should be 20:00 previous"
    
    # Should include only closed 1d (prev day), not same day
    assert len(aligned["1d"]) == 1
    assert aligned["1d"][0][0] == T - tf_1d
    
    print("✓ multi-timeframe alignment test passed - no forward-fill, only closed higher-TF")

def test_incomplete_current_candle():
    """
    Hardening #6: Distinguish forming vs closed candle
    For training/backtesting: ONLY CLOSED unless explicitly requested
    """
    print("\n=== Test: incomplete current candle (Hardening #6) ===")
    validator = DataValidator("1h")
    fetcher = BinancePublicFetcher()
    
    # Now = 00:30 UTC, 1h candle open 00:00 is incomplete (closes 01:00 > now)
    now_ms = 1689984000000 + 1800000  # 2023-07-22 00:30 UTC - 30 min after open
    tf_1h_ms = 3600000
    open_00 = 1689984000000  # 2023-07-22 00:00
    
    # Check is_closed_candle
    is_closed = validator.is_closed_candle(open_ms=open_00, timeframe_ms=tf_1h_ms, now_ms=now_ms)
    assert not is_closed, f"Candle open {open_00} with TF 1h should be incomplete at now {now_ms} (00:30)"
    
    # At now = 01:00, it should be closed
    now_closed = open_00 + tf_1h_ms  # 01:00
    is_closed_at_1 = validator.is_closed_candle(open_00, tf_1h_ms, now_closed)
    assert is_closed_at_1, "Candle should be closed at open+TF"
    
    # Test filter_closed_candles
    candles = [
        [open_00 - tf_1h_ms, 100, 110, 90, 105, 10],  # 23:00 previous - closed
        [open_00, 105, 115, 100, 110, 12],  # 00:00 current - incomplete at 00:30
    ]
    
    filtered_closed_only = validator.filter_closed_candles(candles, now_ms=now_ms, include_incomplete=False)
    assert len(filtered_closed_only) == 1, f"Should filter out incomplete, got {len(filtered_closed_only)}"
    assert filtered_closed_only[0][0] == open_00 - tf_1h_ms
    
    filtered_with_incomplete = validator.filter_closed_candles(candles, now_ms=now_ms, include_incomplete=True)
    assert len(filtered_with_incomplete) == 2, "Should include incomplete when explicitly requested"
    
    # Test fetcher's filter
    filtered_fetcher = fetcher.filter_closed_candles(candles, timeframe="1h", now_ms=now_ms, include_incomplete=False)
    assert len(filtered_fetcher) == 1
    
    filtered_fetcher_incl = fetcher.filter_closed_candles(candles, timeframe="1h", now_ms=now_ms, include_incomplete=True)
    assert len(filtered_fetcher_incl) == 2
    
    print("✓ incomplete current candle test passed - training uses only closed, live may include incomplete")

if __name__ == "__main__":
    # Run all tests manually if pytest not available
    test_duplicate_candles()
    test_missing_candles()
    test_out_of_order_candles()
    test_invalid_ohlc_high_low()
    test_invalid_ohlc_high_open()
    test_timezone_conversion()
    test_pagination_boundaries()
    test_repeated_fetches_no_duplicates()
    test_incremental_updates()
    test_deterministic_csv_export()
    test_volume_negative()
    test_timestamp_normalization_utc()
    test_public_fetch_no_api_keys()
    # Hardening pass
    test_database_ordering_chronological()
    test_utc_daily_candles()
    test_data_leakage()
    test_multi_timeframe_alignment()
    test_incomplete_current_candle()
    
    print("\n" + "="*70)
    print("All Phase 2 tests passed (18/18) - trustworthy data foundation + hardening")
    print("="*70)
