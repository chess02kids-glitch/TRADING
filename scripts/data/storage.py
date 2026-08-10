#!/usr/bin/env python3
"""
Phase 2 - SQLite Storage with Deterministic Incremental Updates
- No silent filling (#9)
- Fees/slippage NOT in raw table (#10)
- Duplicate detection via PRIMARY KEY
- CSV export derived from SQLite
- Preserve original data
"""

import sqlite3
import logging
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "data.db"
# Per config, actual path is data/db/kronos_trading.db

class SQLiteStorage:
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            # Try config path first
            try:
                import yaml
                config_path = PROJECT_ROOT / "config" / "config.yaml"
                if config_path.exists():
                    with open(config_path) as f:
                        cfg = yaml.safe_load(f)
                    db_path_str = cfg.get('data', {}).get('db_path', 'data/db/kronos_trading.db')
                    db_path = PROJECT_ROOT / db_path_str
                else:
                    db_path = PROJECT_ROOT / "data" / "db" / "kronos_trading.db"
            except Exception:
                db_path = PROJECT_ROOT / "data" / "db" / "kronos_trading.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect with WAL mode for concurrent reads
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        
        self._create_tables()
        logger.info(f"SQLite storage initialized at {self.db_path}, WAL mode")
    
    def _create_tables(self):
        """Create tables per DATA_SCHEMA.md"""
        
        # ohlcv_raw - no fees/slippage per requirement #10
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_raw (
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
        """)
        
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe_ts 
        ON ohlcv_raw (symbol, timeframe, timestamp_ms);
        """)
        
        # fetch_metadata
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS fetch_metadata (
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
        """)
        
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_fetch_meta_symbol_tf 
        ON fetch_metadata (symbol, timeframe, created_at);
        """)
        
        # validation_reports
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS validation_reports (
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
        """)
        
        self.conn.commit()
        logger.debug("Tables created/verified per DATA_SCHEMA.md")
    
    def ms_to_iso(self, ms: int) -> str:
        try:
            dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
            return dt.isoformat()
        except Exception:
            return str(ms)
    
    def iso_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
    
    def insert_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        candles: List[List[Any]],
        source: str = "binance_ccxt_public"
    ) -> Tuple[int, int]:
        """
        Insert OHLCV candles with duplicate detection
        Returns: (inserted_count, duplicates_skipped)
        Deterministic, idempotent per requirement: repeated fetches produce no duplicate rows
        """
        if not candles:
            return 0, 0
        
        # Prepare rows - ensure sorted ASC for deterministic behavior
        sorted_candles = sorted(candles, key=lambda x: x[0])
        
        # Deduplicate within batch first
        seen = set()
        deduped = []
        dups_in_batch = 0
        for c in sorted_candles:
            ts = c[0]
            if ts in seen:
                dups_in_batch += 1
                continue
            seen.add(ts)
            deduped.append(c)
        
        if dups_in_batch > 0:
            logger.warning(f"Insert {symbol} {timeframe}: {dups_in_batch} duplicates within batch skipped")
        
        # Build rows for DB
        created_at = self.iso_now()
        rows = []
        for c in deduped:
            if len(c) < 6:
                logger.warning(f"Skipping invalid candle length <6: {c}")
                continue
            ts_ms, o, h, l, cl, v = c[0], c[1], c[2], c[3], c[4], c[5]
            ts_iso = self.ms_to_iso(ts_ms)
            rows.append((exchange, symbol, timeframe, ts_ms, ts_iso, o, h, l, cl, v, source, created_at))
        
        # Insert with OR IGNORE to handle PRIMARY KEY duplicates - no duplicate rows on repeated fetches
        inserted = 0
        duplicates_skipped = dups_in_batch
        
        # Use transaction
        try:
            cursor = self.conn.cursor()
            # For counting inserted vs ignored, we need to check before and after
            # SQLite INSERT OR IGNORE doesn't tell how many ignored directly in python, so we count via changes
            # Approach: try insert each row and count
            
            # More efficient: use executemany with OR IGNORE and check total_changes
            # But total_changes is for whole connection, need to capture per batch
            
            # We'll do: SELECT existing timestamps in range to compute expected duplicates
            if rows:
                min_ts = min(r[3] for r in rows)
                max_ts = max(r[3] for r in rows)
                cursor.execute("""
                    SELECT timestamp_ms FROM ohlcv_raw 
                    WHERE exchange=? AND symbol=? AND timeframe=? 
                    AND timestamp_ms BETWEEN ? AND ?
                """, (exchange, symbol, timeframe, min_ts, max_ts))
                existing_ts = set(r[0] for r in cursor.fetchall())
                
                # Filter rows to only new
                new_rows = [r for r in rows if r[3] not in existing_ts]
                duplicate_from_db = len(rows) - len(new_rows)
                duplicates_skipped += duplicate_from_db
                
                if new_rows:
                    cursor.executemany("""
                        INSERT OR IGNORE INTO ohlcv_raw 
                        (exchange, symbol, timeframe, timestamp_ms, timestamp_utc, open, high, low, close, volume, source, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, new_rows)
                    inserted = cursor.rowcount
                    # rowcount may be -1 for some sqlite versions, so use len(new_rows) as inserted if rowcount -1
                    if inserted == -1:
                        inserted = len(new_rows)
                    self.conn.commit()
                    logger.info(f"Inserted {inserted} new candles for {symbol} {timeframe}, {duplicates_skipped} duplicates skipped (idempotent)")
                else:
                    logger.info(f"No new candles to insert for {symbol} {timeframe}, all {duplicates_skipped} were duplicates (repeated fetch test)")
                    inserted = 0
            
            return inserted, duplicates_skipped
        
        except sqlite3.IntegrityError as e:
            # Should not happen with OR IGNORE, but log
            logger.error(f"Integrity error inserting {symbol} {timeframe}: {e}")
            self.conn.rollback()
            return 0, len(rows)
        except Exception as e:
            logger.error(f"Error inserting {symbol} {timeframe}: {e}")
            self.conn.rollback()
            raise
    
    def get_last_timestamp_ms(self, symbol: str, timeframe: str, exchange: str = "binance") -> Optional[int]:
        """Get last timestamp for incremental updates - deterministic"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT MAX(timestamp_ms) FROM ohlcv_raw 
            WHERE exchange=? AND symbol=? AND timeframe=?
        """, (exchange, symbol, timeframe))
        result = cursor.fetchone()
        return result[0] if result and result[0] is not None else None
    
    def get_first_timestamp_ms(self, symbol: str, timeframe: str, exchange: str = "binance") -> Optional[int]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT MIN(timestamp_ms) FROM ohlcv_raw 
            WHERE exchange=? AND symbol=? AND timeframe=?
        """, (exchange, symbol, timeframe))
        result = cursor.fetchone()
        return result[0] if result and result[0] is not None else None
    
    def get_date_range(self, symbol: str, timeframe: str, exchange: str = "binance") -> Tuple[Optional[int], Optional[int]]:
        first = self.get_first_timestamp_ms(symbol, timeframe, exchange)
        last = self.get_last_timestamp_ms(symbol, timeframe, exchange)
        return first, last
    
    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        exchange: str = "binance",
        order: str = "ASC"
    ) -> List[Tuple]:
        """
        Retrieve candles sorted - Hardening #2: Every model-data query MUST explicitly use ORDER BY timestamp_ms ASC
        Do NOT rely on SQLite's natural/table order
        """
        # Explicit ORDER BY required per hardening #2 - never rely on natural order
        query = """
            SELECT exchange, symbol, timeframe, timestamp_ms, timestamp_utc, open, high, low, close, volume, source, created_at
            FROM ohlcv_raw
            WHERE exchange=? AND symbol=? AND timeframe=?
        """
        params = [exchange, symbol, timeframe]
        
        if since_ms is not None:
            query += " AND timestamp_ms >= ?"
            params.append(since_ms)
        if until_ms is not None:
            query += " AND timestamp_ms <= ?"
            params.append(until_ms)
        
        # Hardening #2: Explicit ORDER BY timestamp_ms ASC for chronological data - mandatory
        # Validate order param to prevent SQL injection - only allow ASC/DESC
        if order not in ("ASC", "DESC"):
            order = "ASC"
        query += f" ORDER BY timestamp_ms {order}"
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        result = cursor.fetchall()
        
        # Extra safety: verify returned data is actually sorted per ORDER BY
        # This is for debugging if SQLite somehow returns unsorted
        if result and len(result) > 1:
            timestamps = [r[3] for r in result]  # timestamp_ms at index 3
            if order == "ASC":
                if timestamps != sorted(timestamps):
                    logger.warning(f"get_candles returned unsorted despite ORDER BY ASC for {symbol} {timeframe} - data integrity issue")
            else:
                if timestamps != sorted(timestamps, reverse=True):
                    logger.warning(f"get_candles returned unsorted despite ORDER BY DESC for {symbol} {timeframe}")
        
        return result

    def get_candles_for_model(
        self,
        symbol: str,
        timeframe: str,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        exchange: str = "binance"
    ) -> List[Tuple]:
        """
        Hardening #2: Dedicated method for model input - ALWAYS ASC, explicitly documented
        This method MUST be used for any model training/inference to ensure chronological order
        """
        return self.get_candles(symbol, timeframe, since_ms, until_ms, exchange, order="ASC")
    
    def count_candles(self, symbol: str, timeframe: str, exchange: str = "binance") -> int:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM ohlcv_raw
            WHERE exchange=? AND symbol=? AND timeframe=?
        """, (exchange, symbol, timeframe))
        return cursor.fetchone()[0]
    
    def export_csv(self, symbol: str, timeframe: str, exchange: str = "binance", output_dir: Optional[Path] = None) -> Path:
        """
        CSV export derived from SQLite - deterministic, sorted ASC per requirement
        """
        if output_dir is None:
            output_dir = PROJECT_ROOT / "data" / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Underscore symbol for filename: BTC/USDT -> BTC_USDT
        symbol_underscored = symbol.replace("/", "_")
        filename = f"{exchange}_{symbol_underscored}_{timeframe}.csv"
        output_path = output_dir / filename
        
        candles = self.get_candles(symbol, timeframe, exchange=exchange, order="ASC")
        
        if not candles:
            logger.warning(f"No candles to export for {symbol} {timeframe}")
            # Create empty CSV with header
            df = pd.DataFrame(columns=["timestamp_ms","timestamp_utc","open","high","low","close","volume","exchange","symbol","timeframe"])
            df.to_csv(output_path, index=False)
            return output_path
        
        # Convert to DataFrame
        df = pd.DataFrame(candles, columns=["exchange","symbol","timeframe","timestamp_ms","timestamp_utc","open","high","low","close","volume","source","created_at"])
        # Reorder and keep required columns for CSV per schema
        df = df[["timestamp_ms","timestamp_utc","open","high","low","close","volume","exchange","symbol","timeframe"]]
        df = df.sort_values("timestamp_ms", ascending=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} candles for {symbol} {timeframe} to {output_path} (derived from SQLite, deterministic)")
        
        return output_path
    
    def insert_fetch_metadata(self, metadata: Dict[str, Any]) -> int:
        """Insert fetch metadata per schema"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO fetch_metadata 
            (exchange, symbol, timeframe, fetch_start_ms, fetch_end_ms, candles_fetched, candles_inserted, duplicates_skipped, missing_candles_detected, first_timestamp_ms, last_timestamp_ms, first_timestamp_utc, last_timestamp_utc, fetch_duration_s, status, error_message, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.get('exchange'),
            metadata.get('symbol'),
            metadata.get('timeframe'),
            metadata.get('fetch_start_ms'),
            metadata.get('fetch_end_ms'),
            metadata.get('candles_fetched'),
            metadata.get('candles_inserted'),
            metadata.get('duplicates_skipped'),
            metadata.get('missing_candles_detected'),
            metadata.get('first_timestamp_ms'),
            metadata.get('last_timestamp_ms'),
            metadata.get('first_timestamp_utc'),
            metadata.get('last_timestamp_utc'),
            metadata.get('fetch_duration_s'),
            metadata.get('status'),
            metadata.get('error_message'),
            metadata.get('created_at', self.iso_now())
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_validation_report(self, report: Dict[str, Any]) -> int:
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO validation_reports
            (exchange, symbol, timeframe, check_type, is_valid, issues_found, details, checked_from_ms, checked_to_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report.get('exchange'),
            report.get('symbol'),
            report.get('timeframe'),
            report.get('check_type'),
            report.get('is_valid'),
            report.get('issues_found'),
            json.dumps(report.get('details', {})),
            report.get('checked_from_ms'),
            report.get('checked_to_ms'),
            report.get('created_at', self.iso_now())
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_schema_sql(self) -> str:
        """Return actual CREATE TABLE SQL for reporting per requirement B"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name IN ('ohlcv_raw','fetch_metadata','validation_reports')")
        tables_sql = cursor.fetchall()
        return "\n\n".join([row[0] + ";" for row in tables_sql if row[0]])
    
    def close(self):
        if self.conn:
            self.conn.close()

if __name__ == "__main__":
    # Self-test
    storage = SQLiteStorage(db_path=Path("/tmp/test_kronos.db"))
    
    # Test insert
    candles = [
        [1000, 100, 110, 90, 105, 10],
        [1000 + 3600000, 105, 115, 100, 110, 12],
    ]
    
    inserted, dups = storage.insert_ohlcv("binance", "BTC/USDT", "1h", candles)
    print(f"Inserted {inserted}, dups {dups}")
    
    # Test duplicate insert - should be idempotent
    inserted2, dups2 = storage.insert_ohlcv("binance", "BTC/USDT", "1h", candles)
    print(f"Second insert (repeated fetch test): inserted {inserted2}, dups {dups2} - should be 0 inserted")
    
    # Test incremental
    last_ts = storage.get_last_timestamp_ms("BTC/USDT", "1h")
    print(f"Last timestamp: {last_ts}")
    
    # Export CSV
    csv_path = storage.export_csv("BTC/USDT", "1h", output_dir=Path("/tmp"))
    print(f"CSV exported to {csv_path}")
    
    storage.close()
