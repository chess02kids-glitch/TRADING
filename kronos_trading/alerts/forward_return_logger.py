"""Phase 9A forward-return tracking logger.

For every breakout event, this records three rows (horizons 1/2/3) that track
what price does in the next 1/2/3 bars, then fills in the realised return once
each target bar closes. This is the data the standalone ``phase9a/`` analysis
module consumes (via :func:`get_phase9a_data` → CSV → analysis).

New table ``public.phase9a_forward_returns``::

    id INTEGER PRIMARY KEY
    breakout_timestamp   TEXT   (ISO8601 UTC, the breakout bar open time)
    asset                TEXT
    breakout_direction   INTEGER (+1/-1, the breakout bar's candle direction)
    horizon              INTEGER (1, 2, 3)
    target_timestamp     TEXT   (breakout_timestamp + horizon hours)
    forward_return       REAL   (NULL until the target bar closes)
    forward_direction    INTEGER (NULL until filled; +1 if return >= 0 else -1)
    breakout_close_price REAL   (close AT the breakout bar — return baseline)
    created_at           TEXT
    UNIQUE(breakout_timestamp, asset, horizon)

Architectural rules honoured here:

* Every function takes the ``conn`` — the *caller* owns the connection (same
  pattern as the rest of ``prediction_logger``). No connection strings, no
  global state.
* Works on both ``sqlite3`` (tests) and ``psycopg`` (production Supabase):
  DDL, ``INSERT OR IGNORE`` vs ``ON CONFLICT … DO NOTHING``, and ``?`` vs
  ``%s`` placeholders are adapted to the connection type.
* Idempotent inserts (UNIQUE constraint) — re-tracking a breakout is a no-op.
* Look-ahead safe: forward returns are only filled once
  ``target_timestamp <= now``.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

TABLE = "phase9a_forward_returns"
HORIZONS = (1, 2, 3)
_HOUR_MS = 3_600_000

_RETURN_COLUMNS = [
    "breakout_timestamp", "asset", "breakout_direction", "horizon",
    "target_timestamp", "forward_return", "forward_direction",
    "breakout_close_price",
]


# --------------------------------------------------------------------------- #
# Connection helpers (sqlite3 vs psycopg)
# --------------------------------------------------------------------------- #
def _is_sqlite(conn) -> bool:
    return isinstance(conn, sqlite3.Connection)


def _adapt(conn, sql: str) -> str:
    """``?`` placeholders for sqlite, ``%s`` for psycopg."""
    return sql if _is_sqlite(conn) else sql.replace("?", "%s")


def _exec(conn, sql: str, params: Any = ()) -> None:
    if _is_sqlite(conn):
        conn.execute(sql, params)
        conn.commit()
    else:
        with conn.cursor() as cur:
            cur.execute(_adapt(conn, sql), params)


def _executemany(conn, sql: str, rows) -> None:
    if _is_sqlite(conn):
        conn.executemany(sql, rows)
        conn.commit()
    else:
        with conn.cursor() as cur:
            cur.executemany(_adapt(conn, sql), rows)


def _fetchall(conn, sql: str, params: Any = ()):
    if _is_sqlite(conn):
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    with conn.cursor() as cur:
        cur.execute(_adapt(conn, sql), params)
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, r)) for r in cur.fetchall()]


# --------------------------------------------------------------------------- #
# Timestamp helpers
# --------------------------------------------------------------------------- #
def _iso_to_ms(ts: Any) -> Optional[int]:
    if ts is None:
        return None
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        v = float(ts)
        v = v if abs(v) >= 1e12 else v * 1000.0
        return int(round(v))
    try:
        s = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.astimezone(timezone.utc).timestamp() * 1000)
    except (ValueError, TypeError):
        return None


def _ms_to_iso(ms: Optional[int]) -> Optional[str]:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_iso(ts: Any) -> str:
    ms = _iso_to_ms(ts)
    return _ms_to_iso(ms) if ms is not None else str(ts)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
_SCHEMA_SQLITE = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    breakout_timestamp TEXT NOT NULL,
    asset TEXT NOT NULL,
    breakout_direction INTEGER NOT NULL,
    horizon INTEGER NOT NULL,
    target_timestamp TEXT NOT NULL,
    forward_return REAL,
    forward_direction INTEGER,
    breakout_close_price REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(breakout_timestamp, asset, horizon)
)
"""

_SCHEMA_PG = _SCHEMA_SQLITE.replace(
    "id INTEGER PRIMARY KEY AUTOINCREMENT", "id SERIAL PRIMARY KEY"
)


def create_phase9a_table(conn) -> None:
    """Create the forward-returns table if it does not exist (idempotent)."""
    schema = _SCHEMA_SQLITE if _is_sqlite(conn) else _SCHEMA_PG
    if _is_sqlite(conn):
        conn.execute(schema)
        conn.commit()
    else:
        with conn.cursor() as cur:
            cur.execute(schema)


# --------------------------------------------------------------------------- #
# Logging a breakout for tracking
# --------------------------------------------------------------------------- #
def log_breakout_for_tracking(
    conn,
    breakout_timestamp: str,
    asset: str,
    direction: int,
    close_price: float,
) -> None:
    """Insert 3 tracking rows (horizons 1/2/3) for one breakout event.

    ``target_timestamp = breakout_timestamp + horizon hours``. Both return
    fields start NULL. Re-tracking the same breakout is a silent no-op (the
    UNIQUE constraint on ``(breakout_timestamp, asset, horizon)`` is honoured).
    """
    create_phase9a_table(conn)
    bt_ms = _iso_to_ms(breakout_timestamp)
    base_iso = _ms_to_iso(bt_ms) or breakout_timestamp
    now_iso = _ms_to_iso(int(datetime.now(timezone.utc).timestamp() * 1000))
    rows = []
    for h in HORIZONS:
        target_ms = (bt_ms + h * _HOUR_MS) if bt_ms is not None else None
        target_iso = _ms_to_iso(target_ms) or breakout_timestamp
        rows.append((base_iso, str(asset), int(direction), int(h),
                     target_iso, None, None, float(close_price), now_iso))

    if _is_sqlite(conn):
        sql = (
            f"INSERT OR IGNORE INTO {TABLE} "
            "(breakout_timestamp, asset, breakout_direction, horizon, "
            "target_timestamp, forward_return, forward_direction, "
            "breakout_close_price, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
    else:
        sql = (
            f"INSERT INTO {TABLE} "
            "(breakout_timestamp, asset, breakout_direction, horizon, "
            "target_timestamp, forward_return, forward_direction, "
            "breakout_close_price, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (breakout_timestamp, asset, horizon) DO NOTHING"
        )
    _executemany(conn, sql, rows)


# --------------------------------------------------------------------------- #
# Filling realised forward returns
# --------------------------------------------------------------------------- #
def _build_close_lookup(df: Optional[pd.DataFrame]) -> Dict[int, float]:
    """``{epoch_ms: close}`` from a candle DataFrame (any timestamp format)."""
    if df is None or len(df) == 0:
        return {}
    if "close" not in df.columns or "timestamp" not in df.columns:
        return {}
    out: Dict[int, float] = {}
    for ts, close in zip(df["timestamp"], df["close"]):
        ms = _iso_to_ms(ts)
        if ms is not None:
            try:
                out[ms] = float(close)
            except (TypeError, ValueError):
                continue
    return out


def update_forward_returns(
    conn,
    now_timestamp: str,
    candles_by_asset: Dict[str, pd.DataFrame],
) -> int:
    """Fill realised returns for every target bar that has now closed.

    For rows where ``forward_return IS NULL`` and ``target_timestamp <= now``,
    look up the close at ``target_timestamp`` in ``candles_by_asset[asset]``
    and set ``forward_return = close_target / breakout_close_price - 1`` and
    ``forward_direction`` accordingly. Rows whose candle is not yet available
    are skipped silently (retried on the next cycle). Returns the count of rows
    updated.
    """
    create_phase9a_table(conn)
    now_iso = _normalize_iso(now_timestamp)

    pending = _fetchall(
        conn,
        f"SELECT id, asset, target_timestamp, breakout_close_price FROM {TABLE} "
        "WHERE forward_return IS NULL AND target_timestamp <= ? ORDER BY id",
        (now_iso,),
    )

    lookups = {
        str(asset): _build_close_lookup(df)
        for asset, df in (candles_by_asset or {}).items()
    }

    updated = 0
    for row in pending:
        asset = str(row["asset"])
        target_ms = _iso_to_ms(row["target_timestamp"])
        lk = lookups.get(asset, {})
        close = lk.get(target_ms) if target_ms is not None else None
        if close is None:
            continue  # candle not available yet — retry next hour
        base = float(row["breakout_close_price"])
        if base <= 0:
            continue
        forward_return = close / base - 1.0
        forward_direction = 1 if forward_return >= 0 else -1
        _exec(
            conn,
            f"UPDATE {TABLE} SET forward_return = ?, forward_direction = ? WHERE id = ?",
            (forward_return, forward_direction, row["id"]),
        )
        updated += 1
    return updated


# --------------------------------------------------------------------------- #
# Reading for analysis
# --------------------------------------------------------------------------- #
def get_phase9a_data(conn) -> pd.DataFrame:
    """All rows with a filled forward_return, ordered oldest breakout first."""
    cols_sql = ", ".join(_RETURN_COLUMNS)
    rows = _fetchall(
        conn,
        f"SELECT {cols_sql} FROM {TABLE} WHERE forward_return IS NOT NULL "
        "ORDER BY breakout_timestamp ASC",
    )
    if not rows:
        return pd.DataFrame(columns=_RETURN_COLUMNS)
    return pd.DataFrame(rows, columns=_RETURN_COLUMNS)
