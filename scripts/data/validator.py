#!/usr/bin/env python3
"""
Phase 2 - Data Validator
Implements all validation checks per DATA_SCHEMA.md
- No silent filling (#9)
- Detect and report missing candles first
- If interpolation ever used, explicit configurable step preserving original

Tests per requirement #11
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timezone
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

TIMEFRAME_MS = {
    "1h": 3600000,
    "4h": 14400000,
    "1d": 86400000,
}

class DataValidator:
    def __init__(self, timeframe: str):
        if timeframe not in TIMEFRAME_MS:
            raise ValueError(f"Unsupported timeframe {timeframe}")
        self.timeframe = timeframe
        self.timeframe_ms = TIMEFRAME_MS[timeframe]
    
    def check_duplicate_candles(self, candles: List[List[Any]]) -> Tuple[bool, int, List[int]]:
        """
        Check duplicate (exchange, symbol, timeframe, timestamp_ms)
        Returns: (is_valid, duplicate_count, list of duplicate timestamp_ms)
        """
        seen = {}
        duplicates = []
        duplicate_count = 0
        
        for candle in candles:
            ts = candle[0]
            if ts in seen:
                duplicates.append(ts)
                duplicate_count += 1
            else:
                seen[ts] = 1
        
        is_valid = duplicate_count == 0
        if not is_valid:
            logger.warning(f"Duplicate check {self.timeframe}: Found {duplicate_count} duplicates: {duplicates[:5]}")
        else:
            logger.debug(f"Duplicate check {self.timeframe}: No duplicates, {len(candles)} unique")
        
        return is_valid, duplicate_count, duplicates
    
    def check_missing_candles(self, candles: List[List[Any]]) -> Tuple[bool, int, List[Dict[str, Any]]]:
        """
        Detect missing candles - contiguous timestamps should have step = timeframe_ms
        Returns: (is_valid, missing_count, gaps_list)
        Gaps list: [{"from_ms": ..., "to_ms": ..., "missing_count": ...}]
        Does NOT fill - per requirement #9
        """
        if not candles or len(candles) < 2:
            return True, 0, []
        
        sorted_candles = sorted(candles, key=lambda x: x[0])
        missing_count = 0
        gaps = []
        
        for i in range(len(sorted_candles) - 1):
            curr_ts = sorted_candles[i][0]
            next_ts = sorted_candles[i+1][0]
            diff = next_ts - curr_ts
            
            if diff == self.timeframe_ms:
                continue  # OK, contiguous
            elif diff == 0:
                # Duplicate, handled elsewhere
                continue
            elif diff > self.timeframe_ms:
                # Gap - calculate missing
                # Expected timestamps between curr and next: curr+tf, curr+2tf, ... < next
                # Number missing = (diff / tf) -1
                if diff % self.timeframe_ms == 0:
                    gap_missing = (diff // self.timeframe_ms) - 1
                    if gap_missing > 0:
                        missing_count += gap_missing
                        gaps.append({
                            "from_ms": curr_ts,
                            "to_ms": next_ts,
                            "from_iso": self.ms_to_iso(curr_ts),
                            "to_iso": self.ms_to_iso(next_ts),
                            "diff_ms": diff,
                            "missing_count": gap_missing
                        })
                else:
                    # Diff not aligned to timeframe - irregular gap
                    gap_missing = max(0, round(diff / self.timeframe_ms) - 1)
                    missing_count += gap_missing
                    gaps.append({
                        "from_ms": curr_ts,
                        "to_ms": next_ts,
                        "from_iso": self.ms_to_iso(curr_ts),
                        "to_iso": self.ms_to_iso(next_ts),
                        "diff_ms": diff,
                        "missing_count": gap_missing,
                        "note": "diff not aligned to timeframe_ms, irregular"
                    })
            elif diff < self.timeframe_ms and diff > 0:
                # Overlapping or too close - out of order or duplicate timeframe issue
                gaps.append({
                    "from_ms": curr_ts,
                    "to_ms": next_ts,
                    "from_iso": self.ms_to_iso(curr_ts),
                    "to_iso": self.ms_to_iso(next_ts),
                    "diff_ms": diff,
                    "missing_count": 0,
                    "note": f"diff {diff} < timeframe_ms {self.timeframe_ms}, overlapping"
                })
        
        is_valid = missing_count == 0
        if not is_valid:
            logger.warning(f"Missing check {self.timeframe}: Found {len(gaps)} gaps with {missing_count} missing candles")
            for gap in gaps[:3]:
                logger.warning(f"  Gap: {gap['from_iso']} -> {gap['to_iso']}, missing {gap['missing_count']}")
        else:
            logger.debug(f"Missing check {self.timeframe}: No missing candles, {len(candles)} contiguous")
        
        return is_valid, missing_count, gaps
    
    def check_out_of_order(self, candles: List[List[Any]]) -> Tuple[bool, bool]:
        """
        Check if candles are out of order
        Returns: (is_sorted, was_out_of_order_originally)
        """
        if not candles:
            return True, False
        
        # Check if sorted ASC
        timestamps = [c[0] for c in candles]
        sorted_ts = sorted(timestamps)
        
        is_sorted = timestamps == sorted_ts
        was_out_of_order = not is_sorted
        
        if was_out_of_order:
            logger.warning(f"Out-of-order check {self.timeframe}: Input not sorted ASC, {len(candles)} candles")
        else:
            logger.debug(f"Out-of-order check {self.timeframe}: Sorted ASC OK")
        
        return is_sorted, was_out_of_order
    
    def check_invalid_ohlc(self, candles: List[List[Any]]) -> Tuple[bool, int, List[Dict[str, Any]]]:
        """
        Check OHLC relationships:
        high >= low, high >= open, high >= close, low <= open, low <= close, open>0, high>0, low>0, close>0, volume>=0, finite, not NaN
        Returns: (is_valid, invalid_count, invalid_details)
        """
        invalid_count = 0
        invalid_details = []
        
        for idx, candle in enumerate(candles):
            if len(candle) < 6:
                invalid_count += 1
                invalid_details.append({"index": idx, "timestamp_ms": candle[0] if len(candle)>0 else None, "reason": "candle length <6"})
                continue
            
            ts, o, h, l, c, v = candle[0], candle[1], candle[2], candle[3], candle[4], candle[5]
            
            issues = []
            
            # Check NaN, infinite
            try:
                import math
                for val, name in [(o, 'open'), (h, 'high'), (l, 'low'), (c, 'close'), (v, 'volume')]:
                    if pd.isna(val) or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                        issues.append(f"{name} is NaN or Inf: {val}")
            except Exception:
                pass
            
            # Basic >0 checks
            if o is not None and o <= 0:
                issues.append(f"open <=0: {o}")
            if h is not None and h <= 0:
                issues.append(f"high <=0: {h}")
            if l is not None and l <= 0:
                issues.append(f"low <=0: {l}")
            if c is not None and c <= 0:
                issues.append(f"close <=0: {c}")
            if v is not None and v < 0:
                issues.append(f"volume <0: {v}")
            
            # OHLC relationships
            if h is not None and l is not None and h < l:
                issues.append(f"high < low: {h} < {l}")
            if h is not None and o is not None and h < o:
                issues.append(f"high < open: {h} < {o}")
            if h is not None and c is not None and h < c:
                issues.append(f"high < close: {h} < {c}")
            if l is not None and o is not None and l > o:
                issues.append(f"low > open: {l} > {o}")
            if l is not None and c is not None and l > c:
                issues.append(f"low > close: {l} > {c}")
            
            if issues:
                invalid_count += 1
                invalid_details.append({
                    "index": idx,
                    "timestamp_ms": ts,
                    "timestamp_iso": self.ms_to_iso(ts),
                    "candle": candle,
                    "issues": issues
                })
        
        is_valid = invalid_count == 0
        if not is_valid:
            logger.warning(f"Invalid OHLC check {self.timeframe}: Found {invalid_count} invalid candles out of {len(candles)}")
            for detail in invalid_details[:3]:
                logger.warning(f"  Invalid at {detail['timestamp_iso']}: {detail['issues']}")
        else:
            logger.debug(f"Invalid OHLC check {self.timeframe}: All {len(candles)} candles valid")
        
        return is_valid, invalid_count, invalid_details
    
    def check_timezone_conversion(self, timestamp_ms: int) -> Tuple[bool, str]:
        """
        Check timezone conversion to UTC
        Returns: (is_valid, iso_utc)
        """
        try:
            dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            iso = dt.isoformat()
            # Should be UTC
            is_utc = dt.tzinfo is not None and dt.tzinfo == timezone.utc
            # ISO should contain +00:00 or Z or have UTC
            is_valid = is_utc and (iso.endswith('+00:00') or 'Z' in iso or '+00:00' in iso)
            if not is_valid:
                logger.warning(f"Timezone check: timestamp {timestamp_ms} -> {iso} not UTC")
            return is_valid, iso
        except Exception as e:
            logger.error(f"Timezone conversion failed for {timestamp_ms}: {e}")
            return False, str(e)
    
    def validate_all(self, candles: List[List[Any]]) -> Dict[str, Any]:
        """
        Run all validations per requirement #11
        """
        results = {}
        
        # Duplicate
        is_valid_dup, dup_count, dups = self.check_duplicate_candles(candles)
        results['duplicate'] = {"is_valid": is_valid_dup, "count": dup_count, "details": dups[:10]}
        
        # Missing (do not fill, just detect)
        is_valid_miss, miss_count, gaps = self.check_missing_candles(candles)
        results['missing'] = {"is_valid": is_valid_miss, "missing_count": miss_count, "gaps": gaps[:10]}
        
        # Out of order
        is_sorted, was_out_of_order = self.check_out_of_order(candles)
        results['out_of_order'] = {"is_sorted": is_sorted, "was_out_of_order": was_out_of_order}
        
        # Invalid OHLC
        is_valid_ohlc, invalid_count, invalid_details = self.check_invalid_ohlc(candles)
        results['invalid_ohlc'] = {"is_valid": is_valid_ohlc, "invalid_count": invalid_count, "details": invalid_details[:5]}
        
        # Timezone for first and last
        if candles:
            first_ts = candles[0][0]
            last_ts = candles[-1][0]
            tz_valid_first, iso_first = self.check_timezone_conversion(first_ts)
            tz_valid_last, iso_last = self.check_timezone_conversion(last_ts)
            results['timezone'] = {
                "is_valid": tz_valid_first and tz_valid_last,
                "first": {"ms": first_ts, "iso": iso_first, "valid": tz_valid_first},
                "last": {"ms": last_ts, "iso": iso_last, "valid": tz_valid_last}
            }
        else:
            results['timezone'] = {"is_valid": True, "note": "no candles"}
        
        # Overall
        overall_valid = all([
            results['duplicate']['is_valid'],
            results['invalid_ohlc']['is_valid'],
            # Missing is not considered invalid per se, it's reported but not failing - per requirement #9
            # Out of order is warning, not failure if we sort
        ])
        results['overall_valid'] = overall_valid
        results['total_candles'] = len(candles)
        
        logger.info(f"Validation complete for {self.timeframe}: {len(candles)} candles, overall_valid={overall_valid}, duplicates={results['duplicate']['count']}, missing={results['missing']['missing_count']}, invalid_ohlc={results['invalid_ohlc']['invalid_count']}")
        
        return results
    
    # ==================== Hardening Pass Additions ====================

    def check_database_ordering(self, candles_from_db: List[Tuple]) -> Tuple[bool, List[int]]:
        """
        Hardening #2: Ensure retrieved candles are chronological
        Every model-data query must explicitly use ORDER BY timestamp_ms ASC
        This test verifies retrieved list is sorted ASC
        candles_from_db: list of DB rows or list of [ts, ...] - we check ts order
        """
        if not candles_from_db:
            return True, []
        
        # Extract timestamp_ms - handle both tuple row and list candle
        try:
            # If DB row tuple: (exchange, symbol, timeframe, timestamp_ms, ...)
            # If candle list: [timestamp_ms, open, ...]
            timestamps = []
            for row in candles_from_db:
                if isinstance(row, (list, tuple)):
                    # Try to get timestamp_ms - for DB row it's index 3, for candle list it's 0
                    # Heuristic: if len>=10 and isinstance(row[0], str) -> DB row, ts at 3
                    # Else ts at 0
                    if len(row) >= 10 and isinstance(row[0], str):
                        ts = row[3]
                    else:
                        ts = row[0]
                    timestamps.append(ts)
            
            sorted_ts = sorted(timestamps)
            is_chronological = timestamps == sorted_ts
            if not is_chronological:
                logger.warning(f"Database ordering check {self.timeframe}: Retrieved candles NOT chronological ASC!")
                # Find first out-of-order
                out_of_order_indices = []
                for i in range(len(timestamps)-1):
                    if timestamps[i] > timestamps[i+1]:
                        out_of_order_indices.append(i)
                return False, out_of_order_indices
            else:
                logger.debug(f"Database ordering check {self.timeframe}: Chronological ASC OK, {len(timestamps)} candles")
                return True, []
        except Exception as e:
            logger.error(f"Database ordering check failed: {e}")
            return False, []

    def check_utc_daily_candles(self, timestamp_ms: int) -> Tuple[bool, str, str]:
        """
        Hardening #3: Document and test that Binance timestamps remain exchange-provided UTC
        Do not reconstruct daily candles using India/local timezone
        
        Binance:
        - 1d candle timestamp is 00:00:00 UTC (e.g., 1672531200000 = 2023-01-01 00:00 UTC)
        - 1h candles at UTC hour boundaries
        - 4h at 00,04,08,12,16,20 UTC
        
        We must ensure:
        - ms_to_iso uses timezone.utc explicitly, not local
        - Never use datetime.fromtimestamp() without tz (would use local IST)
        """
        try:
            # Correct: uses timezone.utc
            dt_utc = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            iso_utc = dt_utc.isoformat()
            
            # Incorrect would be: datetime.fromtimestamp(ms/1000) without tz - would be local IST in India
            # We test that our method produces UTC, not IST
            
            # Check UTC
            is_utc = dt_utc.tzinfo == timezone.utc
            has_utc_offset = "+00:00" in iso_utc or iso_utc.endswith('Z')
            
            # For daily candles, hour should be 00:00 UTC if timeframe is 1d and timestamp is daily aligned
            # For general, just ensure UTC
            
            is_valid = is_utc and has_utc_offset
            
            if not is_valid:
                logger.warning(f"UTC daily check failed: {timestamp_ms} -> {iso_utc} not UTC")
            
            return is_valid, iso_utc, "UTC check"
        except Exception as e:
            return False, str(e), f"Error: {e}"

    def check_no_future_leakage(
        self,
        input_candles: List[List[Any]],
        target_candles: List[List[Any]],
        prediction_time_ms: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Hardening #4: Data leakage test - reusable for Phase 3 and backtesting
        For every prediction timestamp T:
        - model input may contain only timestamps <= T
        - prediction targets must be strictly after T
        - no future candle may enter feature/input window
        
        This is critical for Phase 3 Kronos inference and backtesting
        
        Returns: (is_valid, details)
        details contains:
        - max_input_ts, min_target_ts, prediction_time, leakage_found
        """
        details = {
            "prediction_time_ms": prediction_time_ms,
            "prediction_time_iso": self.ms_to_iso(prediction_time_ms),
            "input_count": len(input_candles),
            "target_count": len(target_candles),
            "max_input_ts": None,
            "min_target_ts": None,
            "leakage_found": False,
            "issues": []
        }
        
        if not input_candles:
            details["issues"].append("Empty input_candles")
            return False, details
        
        if not target_candles:
            details["issues"].append("Empty target_candles - at least one target required")
            return False, details
        
        input_timestamps = [c[0] for c in input_candles]
        target_timestamps = [c[0] for c in target_candles]
        
        max_input_ts = max(input_timestamps)
        min_target_ts = min(target_timestamps)
        
        details["max_input_ts"] = max_input_ts
        details["max_input_iso"] = self.ms_to_iso(max_input_ts)
        details["min_target_ts"] = min_target_ts
        details["min_target_iso"] = self.ms_to_iso(min_target_ts)
        
        # Rule 1: Input must be <= T
        if max_input_ts > prediction_time_ms:
            details["leakage_found"] = True
            details["issues"].append(f"Future leakage: max_input_ts {max_input_ts} ({self.ms_to_iso(max_input_ts)}) > prediction_time {prediction_time_ms} ({self.ms_to_iso(prediction_time_ms)}) - input contains future data beyond T")
        
        # Rule 2: Target must be > T (strictly after)
        if min_target_ts <= prediction_time_ms:
            details["leakage_found"] = True
            details["issues"].append(f"Target leakage: min_target_ts {min_target_ts} ({self.ms_to_iso(min_target_ts)}) <= prediction_time {prediction_time_ms} - target must be strictly after T")
        
        # Rule 3: No overlap between input and target
        overlap = set(input_timestamps) & set(target_timestamps)
        if overlap:
            details["leakage_found"] = True
            details["issues"].append(f"Overlap: input and target share timestamps {list(overlap)[:5]} - no overlap allowed")
        
        # Rule 4: Input should be sorted and max == T (or <=T, but ideally last input is T)
        # For strict check, we allow max <=T, but warn if max < T - T-1 (gap)
        # Not failing, just info
        
        is_valid = not details["leakage_found"]
        
        if not is_valid:
            logger.warning(f"Data leakage check FAILED at T={self.ms_to_iso(prediction_time_ms)}: {details['issues']}")
        else:
            logger.debug(f"Data leakage check PASSED at T={self.ms_to_iso(prediction_time_ms)}: input max <= T, target min > T")
        
        return is_valid, details

    def check_multi_timeframe_alignment(
        self,
        prediction_time_ms: int,
        candles_1h: List[List[Any]],
        candles_4h: List[List[Any]],
        candles_1d: List[List[Any]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Hardening #5: Multi-timeframe alignment for 1h/4h/1d
        Document how prediction at time T selects history
        
        Rules (conservative, no forward-fill):
        - 1h history: candles with timestamp_ms <= T (closed)
        - 4h history: candles where timestamp_ms + 4h <= T + 1h (closed at or before T's close)
          Simpler: timestamp_ms <= T and timestamp_ms + 4h <= T + 1h
          Example: T=2023-07-22 00:00 (open of 1h candle that closes 01:00), 4h candle open 2023-07-21 20:00 closes 00:00, so 20:00+4h=00:00 <=01:00 => available
                   4h candle open 00:00 2023-07-22 closes 04:00 >01:00 => NOT available (still forming)
        - 1d history: timestamp_ms + 1d <= T + 1h
        - Do NOT forward-fill future higher-TF candles
        - Use only candles actually closed/available at T
        """
        details = {
            "prediction_time_ms": prediction_time_ms,
            "prediction_time_iso": self.ms_to_iso(prediction_time_ms),
            "1h_count": len(candles_1h),
            "4h_count": len(candles_4h),
            "1d_count": len(candles_1d),
            "issues": [],
            "alignment": {}
        }
        
        tf_1h_ms = TIMEFRAME_MS["1h"]
        tf_4h_ms = TIMEFRAME_MS["4h"]
        tf_1d_ms = TIMEFRAME_MS["1d"]
        
        # Check 1h
        if candles_1h:
            max_1h = max(c[0] for c in candles_1h)
            details["alignment"]["1h_max_ts"] = max_1h
            details["alignment"]["1h_max_iso"] = self.ms_to_iso(max_1h)
            if max_1h > prediction_time_ms:
                details["issues"].append(f"1h history contains future: max {max_1h} > T {prediction_time_ms}")
        
        # Check 4h - must be closed at or before T+1h
        if candles_4h:
            latest_4h_open = max(c[0] for c in candles_4h)
            latest_4h_close = latest_4h_open + tf_4h_ms
            details["alignment"]["4h_latest_open"] = latest_4h_open
            details["alignment"]["4h_latest_open_iso"] = self.ms_to_iso(latest_4h_open)
            details["alignment"]["4h_latest_close"] = latest_4h_close
            details["alignment"]["4h_latest_close_iso"] = self.ms_to_iso(latest_4h_close)
            details["alignment"]["T_plus_1h"] = prediction_time_ms + tf_1h_ms
            
            # Conservative rule: 4h candle close <= T+1h
            if latest_4h_close > prediction_time_ms + tf_1h_ms:
                details["issues"].append(
                    f"4h forward-fill: latest 4h open {self.ms_to_iso(latest_4h_open)} closes {self.ms_to_iso(latest_4h_close)} "
                    f"which is after T+1h {self.ms_to_iso(prediction_time_ms + tf_1h_ms)} - should NOT be used, use previous closed 4h"
                )
        
        # Check 1d
        if candles_1d:
            latest_1d_open = max(c[0] for c in candles_1d)
            latest_1d_close = latest_1d_open + tf_1d_ms
            details["alignment"]["1d_latest_open"] = latest_1d_open
            details["alignment"]["1d_latest_open_iso"] = self.ms_to_iso(latest_1d_open)
            details["alignment"]["1d_latest_close"] = latest_1d_close
            details["alignment"]["1d_latest_close_iso"] = self.ms_to_iso(latest_1d_close)
            
            if latest_1d_close > prediction_time_ms + tf_1h_ms:
                details["issues"].append(
                    f"1d forward-fill: latest 1d open {self.ms_to_iso(latest_1d_open)} closes {self.ms_to_iso(latest_1d_close)} "
                    f"after T+1h {self.ms_to_iso(prediction_time_ms + tf_1h_ms)} - should NOT be used"
                )
        
        is_valid = len(details["issues"]) == 0
        return is_valid, details

    def is_closed_candle(self, open_ms: int, timeframe_ms: int, now_ms: int) -> bool:
        """
        Hardening #6: Distinguish forming vs closed candle
        Closed if open + timeframe <= now
        """
        return open_ms + timeframe_ms <= now_ms

    def filter_closed_candles(
        self,
        candles: List[List[Any]],
        now_ms: int,
        include_incomplete: bool = False
    ) -> List[List[Any]]:
        """
        Hardening #6: For training/backtesting, ONLY CLOSED candles unless explicitly requested
        If include_incomplete=False (default for training), filter out currently forming candle
        If True (for live prediction), include incomplete but flag
        """
        if include_incomplete:
            return candles
        
        timeframe_ms = self.timeframe_ms
        closed = [c for c in candles if self.is_closed_candle(c[0], timeframe_ms, now_ms)]
        incomplete_count = len(candles) - len(closed)
        if incomplete_count > 0:
            logger.debug(f"Filtered {incomplete_count} incomplete candles for {self.timeframe} at now={self.ms_to_iso(now_ms)} - training uses only closed")
        return closed

    def ms_to_iso(self, ms: int) -> str:
        try:
            dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
            return dt.isoformat()
        except Exception:
            return str(ms)

# ==================== Standalone reusable functions for Phase 3 ====================

def validate_no_future_leakage(
    input_timestamps: List[int],
    target_timestamps: List[int],
    prediction_time_ms: int
) -> Tuple[bool, Dict[str, Any]]:
    """
    Reusable validation function for Phase 3 and backtesting - Hardening #4
    Ensures no future data leakage
    
    Args:
        input_timestamps: list of timestamp_ms used as model input
        target_timestamps: list of timestamp_ms to predict
        prediction_time_ms: T - time of last known input
    
    Returns:
        (is_valid, details)
    """
    validator = DataValidator("1h")  # timeframe doesn't matter for this check, uses ms_to_iso
    # Convert timestamps to dummy candles format for reuse
    input_candles = [[ts, 0, 0, 0, 0, 0] for ts in input_timestamps]
    target_candles = [[ts, 0, 0, 0, 0, 0] for ts in target_timestamps]
    return validator.check_no_future_leakage(input_candles, target_candles, prediction_time_ms)

def get_aligned_history_for_prediction(
    prediction_time_ms: int,
    all_1h_candles: List[List[Any]],
    all_4h_candles: List[List[Any]],
    all_1d_candles: List[List[Any]],
    lookback_1h: int = 400,
    lookback_4h: int = 100,
    lookback_1d: int = 30
) -> Dict[str, List[List[Any]]]:
    """
    Hardening #5: Get aligned history for prediction at time T
    Uses only closed candles available at T, no forward-fill
    
    Returns dict with '1h', '4h', '1d' aligned lists
    """
    tf_1h = TIMEFRAME_MS["1h"]
    tf_4h = TIMEFRAME_MS["4h"]
    tf_1d = TIMEFRAME_MS["1d"]
    
    # 1h: <= T
    history_1h = [c for c in all_1h_candles if c[0] <= prediction_time_ms]
    history_1h = sorted(history_1h, key=lambda x: x[0])[-lookback_1h:]
    
    # 4h: closed at or before T+1h (conservative)
    history_4h = [c for c in all_4h_candles if c[0] + tf_4h <= prediction_time_ms + tf_1h]
    history_4h = sorted(history_4h, key=lambda x: x[0])[-lookback_4h:]
    
    # 1d: closed at or before T+1h
    history_1d = [c for c in all_1d_candles if c[0] + tf_1d <= prediction_time_ms + tf_1h]
    history_1d = sorted(history_1d, key=lambda x: x[0])[-lookback_1d:]
    
    return {"1h": history_1h, "4h": history_4h, "1d": history_1d}


if __name__ == "__main__":
    # Self-test with sample data
    validator = DataValidator("1h")
    
    # Valid data
    valid_candles = [
        [1000, 100, 110, 90, 105, 10],
        [1000 + 3600000, 105, 115, 100, 110, 12],
        [1000 + 2*3600000, 110, 120, 105, 115, 15],
    ]
    
    print("Testing valid data:")
    result = validator.validate_all(valid_candles)
    print(f"Overall valid: {result['overall_valid']}, missing: {result['missing']['missing_count']}")
    
    # Data with missing
    missing_candles = [
        [1000, 100, 110, 90, 105, 10],
        [1000 + 3*3600000, 110, 120, 105, 115, 15],  # Gap of 2 candles missing
    ]
    print("\nTesting missing data (should detect 2 missing, NOT fill):")
    result = validator.validate_all(missing_candles)
    print(f"Missing count: {result['missing']['missing_count']}, gaps: {result['missing']['gaps']}")
    
    # Invalid OHLC
    invalid_candles = [
        [1000, 100, 90, 90, 105, 10],  # high < low
    ]
    print("\nTesting invalid OHLC:")
    result = validator.validate_all(invalid_candles)
    print(f"Invalid count: {result['invalid_ohlc']['invalid_count']}")
