"""
Results logging for the research_generations table.

Primary backend: Supabase Postgres (SUPABASE_DB_URL env var) using the
schema in supabase/007_lab_schema.sql (CREATE TABLE IF NOT EXISTS +
ADD COLUMN IF NOT EXISTS so nothing is ever dropped).

Fallback backend: when SUPABASE_DB_URL is not reachable (as in this
sandboxed run) every row is inserted into a local SQLite mirror with
identical columns AND appended to a JSONL file, so no result is ever
lost and the rows can be replayed into Supabase later with
`python agent/replay_to_supabase.py`.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict

HERE = os.path.dirname(os.path.abspath(__file__))
LAB_ROOT = os.path.dirname(HERE)
SQLITE_PATH = os.path.join(LAB_ROOT, "data", "research_generations.sqlite")
JSONL_PATH = os.path.join(LAB_ROOT, "research", "results", "log.jsonl")

COLUMNS = [
    ("generation", "INTEGER"), ("genome_id", "TEXT"), ("genome", "TEXT"),
    ("signal_type", "TEXT"), ("asset", "TEXT"), ("total_trades", "INTEGER"),
    ("profit_factor", "REAL"), ("sharpe_ratio", "REAL"), ("max_drawdown", "REAL"),
    ("total_return_pct", "REAL"), ("passed_all_gates", "BOOLEAN"),
    ("gate_failed", "TEXT"), ("failure_reason", "TEXT"),
    ("oos_sharpe", "REAL"), ("oos_positive_splits", "INTEGER"),
    ("concentration_score", "REAL"), ("robustness_score", "REAL"),
    ("stability_score", "REAL"), ("created_at", "TIMESTAMPTZ"),
    ("seed", "INTEGER"), ("parent_genome_ids", "TEXT"),
]


class ResultsLogger:
    def __init__(self):
        self.mode = None  # "supabase" | "sqlite"
        self.supabase_conn = None
        db_url = os.environ.get("SUPABASE_DB_URL")
        if db_url:
            try:
                import psycopg  # noqa
                self.supabase_conn = psycopg.connect(db_url, prepare_threshold=None)
                self._ensure_supabase_schema()
                self.mode = "supabase"
            except Exception as e:  # noqa: BLE001
                print(f"[ResultsLogger] Supabase unavailable ({e}); falling back to SQLite")
                self.supabase_conn = None
        if self.mode is None:
            os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
            os.makedirs(os.path.dirname(JSONL_PATH), exist_ok=True)
            con = sqlite3.connect(SQLITE_PATH)
            cols_sql = ",\n  ".join(f"{c} {t}" for c, t in COLUMNS)
            con.execute(f"CREATE TABLE IF NOT EXISTS research_generations (\n  {cols_sql}\n)")
            con.commit()
            con.close()
            self.mode = "sqlite"

    def _ensure_supabase_schema(self):
        sql_path = os.path.join(LAB_ROOT, "supabase", "007_lab_schema.sql")
        with open(sql_path) as f:
            sql = f.read()
        with self.supabase_conn.cursor() as cur:
            cur.execute(sql)
        self.supabase_conn.commit()

    def insert_genome_result(self, row: Dict):
        """Insert one row immediately (crash-safe, one by one)."""
        row = dict(row)
        row.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        if row.get("parent_genome_ids") is None:
            row["parent_genome_ids"] = None
        elif isinstance(row["parent_genome_ids"], (list, tuple)):
            row["parent_genome_ids"] = "{" + ",".join(map(str, row["parent_genome_ids"])) + "}"

        if self.mode == "supabase":
            try:
                placeholders = ", ".join(["%s"] * len(COLUMNS))
                names = ", ".join(c for c, _ in COLUMNS)
                with self.supabase_conn.cursor() as cur:
                    cur.execute(
                        f"INSERT INTO research_generations ({names}) VALUES ({placeholders})",
                        [row.get(c) for c, _ in COLUMNS],
                    )
                self.supabase_conn.commit()
                return
            except Exception as e:  # noqa: BLE001
                print(f"[ResultsLogger] Supabase insert failed ({e}); row kept in SQLite+JSONL")

        con = sqlite3.connect(SQLITE_PATH)
        placeholders = ", ".join(["?"] * len(COLUMNS))
        names = ", ".join(c for c, _ in COLUMNS)
        con.execute(
            f"INSERT INTO research_generations ({names}) VALUES ({placeholders})",
            [row.get(c) for c, _ in COLUMNS],
        )
        con.commit()
        con.close()
        with open(JSONL_PATH, "a") as f:
            f.write(json.dumps({c: row.get(c) for c, _ in COLUMNS}, default=str) + "\n")


def notify(message: str):
    """Telegram alert if configured (no-op otherwise)."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": message}, timeout=10,
        )
    except Exception:  # noqa: BLE001
        pass
