"""Backfill ``breakout_direction`` / candle open-close for historical breakouts.

For every existing ``har_predictions`` row where ``breakout_flag = 1`` but
``breakout_direction IS NULL``, fetch the matching 1h candle from KuCoin and
fill in:

* ``breakout_direction``    = +1 if ``close >= open`` else -1
* ``breakout_candle_open``  = the breakout bar's open
* ``breakout_candle_close`` = the breakout bar's close

Safety rules (enforced in code):

* **Never overwrite** a row that already has ``breakout_direction`` set — the
  SELECT and the UPDATE both gate on ``breakout_direction IS NULL``.
* **Never crash on a single row** — every row is wrapped in its own
  ``try/except``; failures are counted and skipped.
* **Exact timestamp match** — the breakout bar's ISO8601 UTC open time is
  matched exactly against the candle's open time (1h-aligned, UTC).
* Rate-limited (0.5 s sleep between KuCoin fetches); ``--dry-run`` logs but
  writes nothing.

Usage::

    python scripts/backfill_breakout_direction.py --db-url "$SUPABASE_DB_URL"
    python scripts/backfill_breakout_direction.py --db-url "$SUPABASE_DB_URL" --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

LOG = logging.getLogger("backfill")

TIMEFRAME = "1h"
RATE_LIMIT_SLEEP = 0.5


def _log(msg: str) -> None:
    LOG.info("[%s] %s", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), msg)


def _iso_to_ms(ts: str) -> Optional[int]:
    """ISO8601 UTC string -> epoch-millis (None if unparseable)."""
    if not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.astimezone(timezone.utc).timestamp() * 1000)
    except (ValueError, TypeError):
        return None


def compute_direction(candle_open: float, candle_close: float) -> int:
    """+1 if close >= open (UP), else -1 (DOWN). Pure, no I/O."""
    return 1 if float(candle_close) >= float(candle_open) else -1


def find_candle(rows: Sequence[Sequence[float]], ts_ms: int) -> Optional[Tuple[float, float]]:
    """Find the OHLCV row whose open time == ``ts_ms``; return ``(open, close)``.

    ``rows`` is a list of CCXT ``[ts, o, h, l, c, v]`` tuples. Returns ``None``
    when no exact timestamp match exists.
    """
    for r in rows:
        if len(r) < 5:
            continue
        if int(r[0]) == int(ts_ms):
            return float(r[1]), float(r[4])
    return None


def _default_fetcher(exchange, asset: str, ts_ms: int) -> List[List[float]]:
    """Fetch a small OHLCV window around ``ts_ms`` via CCXT (KuCoin setup)."""
    import ccxt  # type: ignore

    if exchange is None:
        exchange = ccxt.kucoin({"enableRateLimit": True,
                                "options": {"defaultType": "spot"}})
    # since = ts_ms, limit 2 to robustly capture the target bar.
    return exchange.fetch_ohlcv(asset, TIMEFRAME, since=ts_ms, limit=2) or []


# --- DB helpers (work on both sqlite3 and psycopg connections) ---------------

def _is_sqlite(conn) -> bool:
    return isinstance(conn, sqlite3.Connection)


def _exec(conn, sql: str, params: Sequence[Any] = ()) -> None:
    if _is_sqlite(conn):
        conn.execute(sql, params)
        conn.commit()
    else:
        with conn.cursor() as cur:
            cur.execute(sql.replace("?", "%s"), params)


def _fetchall(conn, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    if _is_sqlite(conn):
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    with conn.cursor() as cur:
        cur.execute(sql.replace("?", "%s"), params)
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def fetch_pending_breakouts(conn) -> List[Dict[str, Any]]:
    """Rows needing backfill: breakout, direction unset, actual known."""
    return _fetchall(
        conn,
        'SELECT id, "timestamp", asset FROM har_predictions '
        "WHERE breakout_flag = 1 AND breakout_direction IS NULL "
        "AND actual_range IS NOT NULL ORDER BY \"timestamp\"",
    )


def update_breakout_direction(
    conn, row_id: int, direction: int, candle_open: float, candle_close: float,
) -> bool:
    """Fill the three columns for one row. Never overwrites a set direction."""
    if _is_sqlite(conn):
        cur = conn.execute(
            "UPDATE har_predictions SET breakout_direction = ?, "
            "breakout_candle_open = ?, breakout_candle_close = ? "
            "WHERE id = ? AND breakout_direction IS NULL",
            (int(direction), float(candle_open), float(candle_close), int(row_id)),
        )
        conn.commit()
        return cur.rowcount > 0
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE har_predictions SET breakout_direction = %s, "
            "breakout_candle_open = %s, breakout_candle_close = %s "
            "WHERE id = %s AND breakout_direction IS NULL".replace("%s", "%s"),
            (int(direction), float(candle_open), float(candle_close), int(row_id)),
        )
        return cur.rowcount > 0


def run_backfill(
    conn,
    exchange=None,
    dry_run: bool = False,
    fetcher: Callable = _default_fetcher,
    rate_limit_sleep: float = RATE_LIMIT_SLEEP,
) -> Dict[str, int]:
    """Backfill all pending breakout rows. Returns a counts summary dict.

    ``fetcher(exchange, asset, ts_ms) -> list[[ts,o,h,l,c,v]]`` is injectable so
    tests can supply deterministic candles without hitting the network.
    """
    counts = {"found": 0, "updated": 0, "not_found": 0, "db_errors": 0,
              "skipped_existing": 0}

    rows = fetch_pending_breakouts(conn)
    counts["found"] = len(rows)
    _log(f"Breakout rows found: {len(rows)}")

    for row in rows:
        asset = row["asset"]
        ts = row["timestamp"]
        ts_ms = _iso_to_ms(ts)
        if ts_ms is None:
            _log(f"WARNING: unparseable timestamp for id={row['id']} — skipping")
            counts["db_errors"] += 1
            continue
        try:
            ohlcv = fetcher(exchange, asset, ts_ms)
            match = find_candle(ohlcv, ts_ms)
        except Exception as exc:
            _log(f"WARNING: fetch failed for {asset} {ts}: {exc} — skipping")
            counts["not_found"] += 1
            continue
        if match is None:
            _log(f"WARNING: candle not found for {asset} {ts} — skipping")
            counts["not_found"] += 1
            continue
        candle_open, candle_close = match
        direction = compute_direction(candle_open, candle_close)
        if dry_run:
            _log(f"DRY-RUN: would update {asset} {ts} -> dir={direction}")
            counts["updated"] += 1
        else:
            try:
                ok = update_breakout_direction(
                    conn, row["id"], direction, candle_open, candle_close)
                if ok:
                    _log(f"Updated {asset} {ts} -> dir={direction}")
                    counts["updated"] += 1
                else:
                    counts["skipped_existing"] += 1
            except Exception as exc:
                _log(f"ERROR: DB update failed for id={row['id']}: {exc}")
                counts["db_errors"] += 1
        if rate_limit_sleep > 0 and not dry_run:
            time.sleep(rate_limit_sleep)

    _log(f"Summary: {counts}")
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Backfill breakout_direction")
    parser.add_argument("--db-url", default=os.environ.get("SUPABASE_DB_URL", ""),
                        help="Supabase connection string (or set SUPABASE_DB_URL).")
    parser.add_argument("--dry-run", action="store_true", help="Log but do not write.")
    args = parser.parse_args(argv)

    url = (args.db_url or "").strip().strip('"').strip("'")
    if not url:
        _log("ERROR: --db-url / SUPABASE_DB_URL not set")
        return 1
    if not (url.startswith("postgres://") or url.startswith("postgresql://")):
        url = "postgresql://" + url

    try:
        import psycopg  # type: ignore
        conn = psycopg.connect(url)
    except Exception as exc:
        _log(f"ERROR: connection failed: {exc}")
        return 1

    try:
        counts = run_backfill(conn, exchange=None, dry_run=args.dry_run)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    print(f"Breakout rows found: {counts['found']}")
    print(f"Successfully updated: {counts['updated']}")
    print(f"Candle not found: {counts['not_found']}")
    print(f"DB errors: {counts['db_errors']}")
    print(f"Already filled (skipped): {counts['skipped_existing']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
