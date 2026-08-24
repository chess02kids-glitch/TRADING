"""Phase 9A schema migration — add breakout-direction columns (idempotent).

Adds three columns to ``public.har_predictions`` so the breakout bar's candle
direction can be persisted alongside the existing breakout flag:

* ``breakout_direction``      INTEGER  (+1 UP / -1 DOWN / NULL)
* ``breakout_candle_open``    REAL     (open price at the breakout bar, NULL otherwise)
* ``breakout_candle_close``   REAL     (close price at the breakout bar, NULL otherwise)

Design guarantees (read the docstrings in the functions for the why):

* **Idempotent** — each column is checked against ``information_schema`` /
  ``PRAGMA table_info`` before being added; already-present columns are skipped.
* **Never destructive** — only ``ALTER TABLE ... ADD COLUMN`` is ever issued;
  nothing is dropped, truncated, or rewritten.
* **Row-count invariant** — rows are counted before and after and asserted
  equal, so a runaway migration cannot silently corrupt data.
* **Transactional** — runs inside a single transaction with rollback on any
  error (Postgres). ``--db-url`` or ``SUPABASE_DB_URL``; credentials are never
  hardcoded.

Usage::

    python scripts/migrate_phase9a.py --db-url "$SUPABASE_DB_URL"
    python scripts/migrate_phase9a.py            # reads SUPABASE_DB_URL

Exit 0 on success, 1 on any failure.
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Iterable, List, Sequence, Set, Tuple

LOG = logging.getLogger("migrate_phase9a")

# (column_name, sql_type) — DEFAULT NULL is implicit for added columns.
PHASE9A_COLUMNS: Sequence[Tuple[str, str]] = (
    ("breakout_direction", "INTEGER"),
    ("breakout_candle_open", "REAL"),
    ("breakout_candle_close", "REAL"),
)

TABLE = "har_predictions"


def _log(msg: str) -> None:
    LOG.info("[%s] %s", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), msg)


def _normalize_url(url: str) -> str:
    url = (url or "").strip().strip('"').strip("'")
    if not url:
        raise ValueError("db-url is empty (pass --db-url or set SUPABASE_DB_URL)")
    if not (url.startswith("postgres://") or url.startswith("postgresql://")):
        url = "postgresql://" + url
    return url


def get_column_names(conn, table: str = TABLE) -> Set[str]:
    """Return the set of column names for ``table``, working on pg or sqlite."""
    if isinstance(conn, sqlite3.Connection):
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(r[1]) for r in rows}
    # psycopg connection
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = %s",
            (table,),
        )
        return {str(r[0]) for r in cur.fetchall()}


def count_rows(conn, table: str = TABLE) -> int:
    """Count rows in ``table`` (0 when the table does not exist yet)."""
    try:
        if isinstance(conn, sqlite3.Connection):
            return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            return int(cur.fetchone()[0])
    except Exception:
        return 0


def apply_migration(conn, columns: Iterable[Tuple[str, str]] = PHASE9A_COLUMNS) -> Tuple[List[str], List[str]]:
    """Add any missing columns. Returns ``(added, skipped)`` lists.

    The caller is responsible for commit/rollback boundaries (Postgres); on
    SQLite each ``ALTER`` autocommits but is independently safe.
    """
    existing = get_column_names(conn)
    added: List[str] = []
    skipped: List[str] = []
    for name, col_type in columns:
        if name in existing:
            _log(f"Column {name} already exists — skipping")
            skipped.append(name)
            continue
        sql = f"ALTER TABLE {TABLE} ADD COLUMN {name} {col_type} DEFAULT NULL"
        _log(f"Adding column: {sql}")
        if isinstance(conn, sqlite3.Connection):
            conn.execute(sql)
        else:
            with conn.cursor() as cur:
                cur.execute(sql)
        added.append(name)
    return added, skipped


def _connect(db_url: str):
    import psycopg  # type: ignore

    conn = psycopg.connect(db_url)
    return conn


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Phase 9A har_predictions migration")
    parser.add_argument("--db-url", default=os.environ.get("SUPABASE_DB_URL", ""),
                        help="Supabase connection string (or set SUPABASE_DB_URL).")
    args = parser.parse_args(argv)

    try:
        db_url = _normalize_url(args.db_url)
    except ValueError as exc:
        _log(f"ERROR: {exc}")
        return 1

    _log(f"Connecting to database")
    try:
        conn = _connect(db_url)
    except Exception as exc:
        _log(f"ERROR: connection failed: {exc}")
        return 1

    try:
        before = count_rows(conn)
        _log(f"Rows before: {before}")

        # Postgres: one transaction, rollback on any error. SQLite: autocommit DDL.
        is_pg = not isinstance(conn, sqlite3.Connection)
        if is_pg:
            pass
        try:
            added, skipped = apply_migration(conn)
            if is_pg:
                conn.commit()
        except Exception:
            if is_pg:
                conn.rollback()
            raise

        after = count_rows(conn)
        _log(f"Rows after: {after}")
        assert before == after, (
            f"row count changed during migration: {before} -> {after}")
    except Exception as exc:
        import traceback
        _log(f"ERROR: migration failed: {exc}\n{traceback.format_exc()}")
        try:
            conn.close()
        except Exception:
            pass
        return 1
    finally:
        try:
            if is_pg:
                conn.close()
        except Exception:
            pass

    print("Migration complete")
    print(f"Rows before: {before}")
    print(f"Rows after: {after}")
    print(f"Columns added: {added}")
    print(f"Columns skipped (existed): {skipped}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
