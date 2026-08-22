"""Paper position tracking in a local SQLite database.

Module-level API (the db path is set via :func:`initialize_db` and defaults to
``execution/paper_positions.db``). Deliberately local SQLite — **not** Supabase
— kept separate from the ``har_predictions`` research table.

Schema ``paper_positions`` mirrors the Phase 9A execution contract:

    id, timestamp, asset, direction, entry_price, size_base, size_usd,
    har_predicted_range, regime, status, exit_price, pnl_usd, pnl_pct,
    exit_timestamp, created_at

PnL convention:

* Long  (``direction = +1``): ``pnl_usd = (exit - entry) * size_base``
* Short (``direction = -1``): ``pnl_usd = (entry - exit) * size_base``

i.e. ``pnl_usd = direction * (exit - entry) * size_base`` and
``pnl_pct = direction * (exit - entry) / entry * 100``.
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = str(Path(__file__).resolve().parent / "paper_positions.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    asset TEXT NOT NULL,
    direction INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    size_base REAL NOT NULL,
    size_usd REAL NOT NULL,
    har_predicted_range REAL NOT NULL,
    regime TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    exit_price REAL,
    pnl_usd REAL,
    pnl_pct REAL,
    exit_timestamp TEXT,
    created_at TEXT NOT NULL
)
"""

# Module-level db path (set by initialize_db; tests point this at tmp_path).
_DB_PATH: str = DEFAULT_DB_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def initialize_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create the schema (idempotent) and set the active db path."""
    global _DB_PATH
    _DB_PATH = str(db_path)
    Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(_DB_PATH)) as conn:
        conn.execute(_SCHEMA)
        conn.commit()


def _connect():
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _rows_to_dicts(rows) -> List[Dict[str, Any]]:
    return [{k: r[k] for k in r.keys()} for r in rows]


def open_position(
    asset: str,
    direction: int,
    entry_price: float,
    size_base: float,
    size_usd: float,
    har_predicted_range: float,
    regime: str,
) -> int:
    """Open a paper position. Returns the new row id."""
    initialize_db(_DB_PATH)
    now = _now_iso()
    with closing(_connect()) as conn:
        cur = conn.execute(
            """INSERT INTO paper_positions
               (timestamp, asset, direction, entry_price, size_base, size_usd,
                har_predicted_range, regime, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)""",
            (now, asset, int(direction), float(entry_price), float(size_base),
             float(size_usd), float(har_predicted_range), str(regime), now),
        )
        conn.commit()
        return int(cur.lastrowid)


def close_position(position_id: int, exit_price: float) -> bool:
    """Close an open position at ``exit_price``; returns whether a row updated."""
    initialize_db(_DB_PATH)
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT direction, entry_price, size_base, status "
            "FROM paper_positions WHERE id = ?",
            (int(position_id),),
        ).fetchone()
        if row is None or row["status"] == "closed":
            return False
        direction = int(row["direction"])
        entry = float(row["entry_price"])
        size_base = float(row["size_base"])
        pnl_usd = direction * (float(exit_price) - entry) * size_base
        pnl_pct = direction * (float(exit_price) - entry) / entry * 100.0 if entry else 0.0
        cur = conn.execute(
            """UPDATE paper_positions
               SET status = 'closed', exit_price = ?, pnl_usd = ?, pnl_pct = ?,
                   exit_timestamp = ?
               WHERE id = ? AND status = 'open'""",
            (float(exit_price), float(pnl_usd), float(pnl_pct), _now_iso(),
             int(position_id)),
        )
        conn.commit()
        return cur.rowcount > 0


def get_open_positions() -> List[Dict[str, Any]]:
    initialize_db(_DB_PATH)
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM paper_positions WHERE status = 'open' ORDER BY id"
        ).fetchall()
    return _rows_to_dicts(rows)


def get_closed_positions() -> List[Dict[str, Any]]:
    initialize_db(_DB_PATH)
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM paper_positions WHERE status = 'closed' ORDER BY id"
        ).fetchall()
    return _rows_to_dicts(rows)


def compute_paper_pnl() -> Dict[str, Any]:
    """Aggregate paper PnL over closed positions + open-position counts."""
    initialize_db(_DB_PATH)
    with closing(_connect()) as conn:
        closed = conn.execute(
            "SELECT pnl_usd, pnl_pct FROM paper_positions WHERE status = 'closed'"
        ).fetchall()
        n_open = conn.execute(
            "SELECT COUNT(*) AS c FROM paper_positions WHERE status = 'open'"
        ).fetchone()["c"]
        n_total = conn.execute(
            "SELECT COUNT(*) AS c FROM paper_positions"
        ).fetchone()["c"]
    pnls = [float(r["pnl_usd"]) for r in closed if r["pnl_usd"] is not None]
    pcts = [float(r["pnl_pct"]) for r in closed if r["pnl_pct"] is not None]
    n_closed = len(closed)
    return {
        "total_trades": int(n_total),
        "open_trades": int(n_open),
        "closed_trades": int(n_closed),
        "total_pnl_usd": float(sum(pnls)) if pnls else 0.0,
        "win_rate": float(sum(1 for p in pnls if p > 0) / len(pnls)) if pnls else 0.0,
        "avg_pnl_pct": float(sum(pcts) / len(pcts)) if pcts else 0.0,
    }
