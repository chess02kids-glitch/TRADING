"""Execution audit log — signals, order attempts, results, and skips.

A completely separate SQLite store from ``har_predictions``. Every signal the
execution layer sees, every order it tries to place, every outcome and every
skip is recorded here with a UTC timestamp and an ``event_type`` tag, so the
full paper-trading decision trail is replayable. Nothing in this module
touches the live HAR bot or the research DB.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = str(Path(__file__).resolve().parent / "execution_log.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS execution_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    asset TEXT,
    direction INTEGER,
    details TEXT,
    reason TEXT,
    created_at TEXT NOT NULL
)
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    return obj


class ExecutionLogger:
    """Append-only audit log for the paper execution layer."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _write(self, event_type: str, asset: str = "", direction: int = 0,
               details: Any = None, reason: str = "") -> None:
        details_json = json.dumps(_to_jsonable(details), default=str) if details is not None else None
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO execution_log
                   (timestamp, event_type, asset, direction, details, reason, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (_now_iso(), event_type, asset, int(direction or 0),
                 details_json, reason, _now_iso()),
            )
            conn.commit()

    # -- typed events ------------------------------------------------------

    def log_signal(self, signal_input: Any) -> None:
        """Record an incoming signal (dataclass or dict-like)."""
        sig = _to_jsonable(signal_input)
        asset = sig.get("asset", "") if isinstance(sig, dict) else ""
        direction = sig.get("direction", 0) if isinstance(sig, dict) else 0
        self._write("signal", asset=asset, direction=direction, details=sig)

    def log_order_attempt(self, signal: Any, params: Any) -> None:
        """Record an order attempt (signal + the built OrderParams)."""
        sig = _to_jsonable(signal)
        asset = sig.get("asset", "") if isinstance(sig, dict) else ""
        direction = sig.get("direction", 0) if isinstance(sig, dict) else 0
        self._write("order_attempt", asset=asset, direction=direction,
                    details={"signal": sig, "order_params": _to_jsonable(params)})

    def log_order_result(self, result: Any) -> None:
        """Record an order outcome (CCXT order dict or status dict)."""
        res = _to_jsonable(result)
        asset = ""
        if isinstance(res, dict):
            asset = res.get("symbol", "") or res.get("asset", "")
        self._write("order_result", asset=asset, details=res)

    def log_skip(self, reason: str) -> None:
        """Record a skipped trade and the reason."""
        self._write("skip", reason=str(reason))

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Return all log rows (oldest first), parsing the JSON ``details``."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM execution_log ORDER BY id"
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = {k: r[k] for k in r.keys()}
            if d.get("details"):
                try:
                    d["details"] = json.loads(d["details"])
                except (TypeError, ValueError):
                    pass
            out.append(d)
        return out
