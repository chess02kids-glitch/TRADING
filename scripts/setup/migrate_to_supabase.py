import os
import sqlite3
import json
from datetime import datetime
import psycopg

SQLITE_DB = "data/db/kronos_trading_verified.db"
SUPABASE_DB_URL = os.environ["SUPABASE_DB_URL"]

BATCH_SIZE = 1000


def parse_jsonb(value):
    if value is None:
        return None
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def migrate_ohlcv(sqlite_conn, pg_conn):
    cur_sqlite = sqlite_conn.cursor()
    cur_pg = pg_conn.cursor()

    cur_sqlite.execute("""
        SELECT exchange, symbol, timeframe, timestamp_ms, timestamp_utc,
               open, high, low, close, volume, source, created_at
        FROM ohlcv_raw
        ORDER BY exchange, symbol, timeframe, timestamp_ms
    """)

    total = 0

    while True:
        rows = cur_sqlite.fetchmany(BATCH_SIZE)
        if not rows:
            break

        cur_pg.executemany("""
            INSERT INTO public.ohlcv_raw (
                exchange, symbol, timeframe, timestamp_ms, timestamp_utc,
                open, high, low, close, volume, source, created_at
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (exchange, symbol, timeframe, timestamp_ms)
            DO UPDATE SET
                timestamp_utc = EXCLUDED.timestamp_utc,
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                source = EXCLUDED.source,
                created_at = EXCLUDED.created_at
        """, rows)

        pg_conn.commit()
        total += len(rows)
        print(f"ohlcv_raw: {total} rows migrated")

    cur_pg.close()
    cur_sqlite.close()
    return total


def migrate_fetch_metadata(sqlite_conn, pg_conn):
    cur_sqlite = sqlite_conn.cursor()
    cur_pg = pg_conn.cursor()

    cur_sqlite.execute("""
        SELECT
            exchange, symbol, timeframe,
            fetch_start_ms, fetch_end_ms,
            candles_fetched, candles_inserted, duplicates_skipped,
            missing_candles_detected,
            first_timestamp_ms, last_timestamp_ms,
            first_timestamp_utc, last_timestamp_utc,
            fetch_duration_s, status, error_message, created_at
        FROM fetch_metadata
        ORDER BY id
    """)

    total = 0

    while True:
        rows = cur_sqlite.fetchmany(BATCH_SIZE)
        if not rows:
            break

        cur_pg.executemany("""
            INSERT INTO public.fetch_metadata (
                exchange, symbol, timeframe,
                fetch_start_ms, fetch_end_ms,
                candles_fetched, candles_inserted, duplicates_skipped,
                missing_candles_detected,
                first_timestamp_ms, last_timestamp_ms,
                first_timestamp_utc, last_timestamp_utc,
                fetch_duration_s, status, error_message, created_at
            )
            VALUES (
                %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s,
                %s, %s,
                %s, %s,
                %s, %s, %s, %s
            )
        """, rows)

        pg_conn.commit()
        total += len(rows)
        print(f"fetch_metadata: {total} rows migrated")

    cur_pg.close()
    cur_sqlite.close()
    return total


def migrate_validation_reports(sqlite_conn, pg_conn):
    cur_sqlite = sqlite_conn.cursor()
    cur_pg = pg_conn.cursor()

    cur_sqlite.execute("""
        SELECT
            exchange, symbol, timeframe,
            check_type, is_valid, issues_found,
            details,
            checked_from_ms, checked_to_ms, created_at
        FROM validation_reports
        ORDER BY id
    """)

    total = 0

    while True:
        rows = cur_sqlite.fetchmany(BATCH_SIZE)
        if not rows:
            break

        converted = []
        for row in rows:
            exchange, symbol, timeframe, check_type, is_valid, issues_found, details, checked_from_ms, checked_to_ms, created_at = row
            converted.append((
                exchange,
                symbol,
                timeframe,
                check_type,
                bool(is_valid),
                issues_found,
                json.dumps(parse_jsonb(details)),
                checked_from_ms,
                checked_to_ms,
                created_at
            ))

        cur_pg.executemany("""
            INSERT INTO public.validation_reports (
                exchange, symbol, timeframe,
                check_type, is_valid, issues_found, details,
                checked_from_ms, checked_to_ms, created_at
            )
            VALUES (
                %s, %s, %s,
                %s, %s, %s, %s::jsonb,
                %s, %s, %s
            )
        """, converted)

        pg_conn.commit()
        total += len(rows)
        print(f"validation_reports: {total} rows migrated")

    cur_pg.close()
    cur_sqlite.close()
    return total


def main():
    print("=" * 70)
    print("KRONOS SQLITE -> SUPABASE MIGRATION")
    print("=" * 70)

    print(f"SQLite source: {SQLITE_DB}")

    sqlite_conn = sqlite3.connect(f"file:{SQLITE_DB}?mode=ro", uri=True)

    try:
        print("Connecting to Supabase...")
        pg_conn = psycopg.connect(SUPABASE_DB_URL, connect_timeout=10)

        try:
            with pg_conn.cursor() as cur:
                cur.execute("SELECT current_database(), current_user")
                print("Connected:", cur.fetchone())

            ohlcv_count = migrate_ohlcv(sqlite_conn, pg_conn)
            metadata_count = migrate_fetch_metadata(sqlite_conn, pg_conn)
            validation_count = migrate_validation_reports(sqlite_conn, pg_conn)

            print("\n" + "=" * 70)
            print("MIGRATION COMPLETE")
            print("=" * 70)
            print(f"ohlcv_raw:          {ohlcv_count}")
            print(f"fetch_metadata:     {metadata_count}")
            print(f"validation_reports: {validation_count}")

        finally:
            pg_conn.close()

    finally:
        sqlite_conn.close()


if __name__ == "__main__":
    main()
