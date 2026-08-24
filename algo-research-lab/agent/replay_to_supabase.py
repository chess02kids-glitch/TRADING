"""Replay the SQLite/JSONL results into Supabase research_generations
once SUPABASE_DB_URL is available (this sandbox could not reach Supabase).

Usage:
  export SUPABASE_DB_URL=postgresql://...
  python agent/replay_to_supabase.py            # replay all rows
  python agent/replay_to_supabase.py --dry-run  # count only
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAB_ROOT = os.path.dirname(HERE)
SQLITE_PATH = os.path.join(LAB_ROOT, "data", "research_generations.sqlite")

COLS = [
    "generation", "genome_id", "genome", "signal_type", "asset", "total_trades",
    "profit_factor", "sharpe_ratio", "max_drawdown", "total_return_pct",
    "passed_all_gates", "gate_failed", "failure_reason", "oos_sharpe",
    "oos_positive_splits", "concentration_score", "robustness_score",
    "stability_score", "created_at", "seed", "parent_genome_ids",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(SQLITE_PATH)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        f"SELECT {', '.join(COLS)} FROM research_generations ORDER BY rowid")]
    con.close()
    print(f"{len(rows)} rows in SQLite mirror")

    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        print("SUPABASE_DB_URL not set - nothing to do (dry run).")
        return
    if args.dry_run:
        return

    import psycopg
    sql_schema = os.path.join(LAB_ROOT, "supabase", "007_lab_schema.sql")
    with psycopg.connect(db_url, prepare_threshold=None) as conn:
        with conn.cursor() as cur:
            cur.execute(open(sql_schema).read())
        conn.commit()
        names = ", ".join(COLS)
        placeholders = ", ".join(["%s"] * len(COLS))
        for r in rows:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO research_generations ({names}) VALUES ({placeholders})",
                    [r[c] for c in COLS],
                )
            conn.commit()
        with conn.cursor() as cur:
            cur.execute("SELECT generation, COUNT(*) FROM research_generations GROUP BY generation")
            print("Supabase rows by generation:", cur.fetchall())
    print("Replay complete.")


if __name__ == "__main__":
    sys.exit(main())
