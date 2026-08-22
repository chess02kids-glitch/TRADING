"""Paper position tracking in a local SQLite database.

Records every paper position opened by the execution layer and computes
realized PnL on close. Deliberately local SQLite (not Supabase): this is
paper-only bookkeeping, kept separate from the ``har_predictions`` research
table. The schema mirrors the pre-registered contract exactly.

PnL convention:

* Long (``direction = +1``): ``pnl = (exit_price - entry_price) * size``
* Short (``direction = -1``): ``pnl = (entry_price - exit_price) * size``

Equivalently ``pnl = direction * (exit_price - entry_price) * size``. ``size``
is in base currency (e.g. BTC), so pnl is in quote currency (USDT).
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from execution.order_manager import OrderParams

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = str(Path(__file__).resolve().parent / "paper_positions.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    asset TEXT NOT NULL,
    direction INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    size REAL NOT NULL,
    har_predicted_range REAL,
    regime TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    exit_price REAL,
    pnl REAL,
    exit_timestamp TEXT
)
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PositionTracker:
    """SQLite-backed tracker for open/closed paper positions."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = str(db_path)
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return {k: row[k] for k in row.keys()}

    def open_position(self, order_params: OrderParams, fill_price: float) -> int:
        """Open a paper position at ``fill_price``. Returns the new row id."""
        direction = int(order_params.direction)
        if direction == 0:
            direction = 1 if str(order_params.side).lower() == "buy" else -1
        with closing(self._connect()) as conn:
            cur = conn.execute(
                """INSERT INTO paper_positions
                   (timestamp, asset, direction, entry_price, size,
                    har_predicted_range, regime, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'open')""",
                (_now_iso(), order_params.symbol, direction, float(fill_price),
                 float(order_params.size), float(order_params.har_predicted_range),
                 order_params.regime),
            )
            conn.commit()
            return int(cur.lastrowid)

    def close_position(self, position_id: int, exit_price: float) -> bool:
        """Close an open position at ``exit_price``; returns whether a row updated."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT direction, entry_price, size, status "
                "FROM paper_positions WHERE id = ?",
                (int(position_id),),
            ).fetchone()
            if row is None:
                logger.warning("close_position: no position id=%s", position_id)
                return False
            if row["status"] == "closed":
                logger.warning("close_position: id=%s already closed", position_id)
                return False
            direction = int(row["direction"])
            pnl = direction * (float(exit_price) - float(row["entry_price"])) * float(row["size"])
            cur = conn.execute(
                """UPDATE paper_positions
                   SET status = 'closed', exit_price = ?, pnl = ?, exit_timestamp = ?
                   WHERE id = ? AND status = 'open'""",
                (float(exit_price), float(pnl), _now_iso(), int(position_id)),
            )
            conn.commit()
            return cur.rowcount > 0

    def get_open_positions(self) -> List[Dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM paper_positions WHERE status = 'open' ORDER BY id"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_position_history(self) -> List[Dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM paper_positions ORDER BY id"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def compute_paper_pnl(self) -> Dict[str, Any]:
        """Aggregate realized PnL from closed positions + open-position counts."""
        with closing(self._connect()) as conn:
            closed = conn.execute(
                "SELECT pnl FROM paper_positions WHERE status = 'closed'"
            ).fetchall()
            n_open = conn.execute(
                "SELECT COUNT(*) AS c FROM paper_positions WHERE status = 'open'"
            ).fetchone()["c"]
        pnls = [float(r["pnl"]) for r in closed if r["pnl"] is not None]
        realized = float(sum(pnls)) if pnls else 0.0
        return {
            "realized_pnl": realized,
            "n_closed": int(len(pnls)),
            "n_open": int(n_open),
            "avg_realized_pnl": float(realized / len(pnls)) if pnls else 0.0,
            "win_rate": float(sum(1 for p in pnls if p > 0) / len(pnls)) if pnls else 0.0,
        }
