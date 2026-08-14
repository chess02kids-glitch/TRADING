import os
import sqlite3
import hashlib
from datetime import datetime, timezone
import psycopg

SQLITE_DB = "data/db/kronos_trading_verified.db"


def canonical_timestamp(value):
    if value is None:
        return ""

    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat()

    text = str(value)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat()
    except ValueError:
        return text


def canonical_row(row):
    values = []

    for index, value in enumerate(row):
        # timestamp_utc is column 4
        if index == 4:
            values.append(canonical_timestamp(value))
        else:
            values.append("" if value is None else str(value))

    return "|".join(values)


def digest(rows):
    h = hashlib.sha256()

    for row in rows:
        h.update((canonical_row(row) + "\n").encode("utf-8"))

    return h.hexdigest()


def sqlite_rows():
    conn = sqlite3.connect(SQLITE_DB)

    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT exchange, symbol, timeframe, timestamp_ms,
                   timestamp_utc, open, high, low, close, volume, source
            FROM ohlcv_raw
            ORDER BY exchange, symbol, timeframe, timestamp_ms
        """)
        return cur.fetchall()
    finally:
        conn.close()


def supabase_rows(conn):
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT exchange, symbol, timeframe, timestamp_ms,
                   timestamp_utc, open, high, low, close, volume, source
            FROM public.ohlcv_raw
            ORDER BY exchange, symbol, timeframe, timestamp_ms
        """)
        return cur.fetchall()
    finally:
        cur.close()


def main():
    print("=" * 70)
    print("SUPABASE ↔ SQLITE CANONICAL PARITY CHECK")
    print("=" * 70)

    sqlite_data = sqlite_rows()

    print(f"SQLite rows:   {len(sqlite_data)}")
    print(f"SQLite SHA256: {digest(sqlite_data)}")

    conn = psycopg.connect(
        os.environ["SUPABASE_DB_URL"],
        connect_timeout=10
    )

    try:
        pg_data = supabase_rows(conn)

        print(f"Supabase rows:   {len(pg_data)}")
        print(f"Supabase SHA256: {digest(pg_data)}")

        if len(sqlite_data) != len(pg_data):
            print("PARITY: FAIL — row counts differ")
            return 1

        sqlite_hash = digest(sqlite_data)
        supabase_hash = digest(pg_data)

        if sqlite_hash == supabase_hash:
            print("PARITY: PASS")
            print("SQLite and Supabase contain identical canonical OHLCV data.")
            return 0

        print("PARITY: FAIL — canonical hashes differ")

        for i, (sqlite_row, pg_row) in enumerate(zip(sqlite_data, pg_data)):
            if canonical_row(sqlite_row) != canonical_row(pg_row):
                print(f"\nFirst actual data mismatch at row {i}")
                print("SQLite :", sqlite_row)
                print("Supabase:", pg_row)
                return 1

        return 1

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
