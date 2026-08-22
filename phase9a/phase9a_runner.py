"""Phase 9A orchestrator — ties direction, continuation and DM into one report.

Reads completed breakout rows (``breakout_flag == 1``) from the
``har_predictions`` table, pulls the matching candle history from KuCoin via
CCXT, and runs the full pre-registered experiment:

1. compute breakout-bar direction (past-only)
2. compute 1/2/3-bar forward returns
3. compute hit rates (overall / per-asset / per-regime)
4. run the one-sided Diebold-Mariano test vs a random baseline
5. run the G1–G6 gate checks
6. print the results table and (unless ``--dry-run``) write
   ``phase9a/PHASE9A_RESULTS.md``

This module **never writes** to ``har_predictions`` and **never places
trades** — it is a read-only statistical analysis. The two real I/O providers
(:func:`fetch_breakout_rows`, :func:`fetch_candle_history`) are isolated from
the pure :func:`run_analysis` core so the whole experiment can be unit-tested
with synthetic data and no network.

CLI::

    python -m phase9a.phase9a_runner --db-url "$SUPABASE_DB_URL" \\
        --asset both --horizon 1 --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Dict, Iterable, List, Optional

import pandas as pd

from phase9a.continuation_tester import (
    compute_hit_rate,
    compute_temporal_stability,
    merge_direction_returns,
    run_gate_checks,
)
from phase9a.direction_calculator import (
    compute_breakout_direction,
    compute_forward_returns,
)
from phase9a.dm_test import compute_dm_statistic

logger = logging.getLogger(__name__)

DEFAULT_CANDLE_HISTORY = 3000          # ~125 days of 1h candles
TIMEFRAME = "1h"
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "PHASE9A_RESULTS.md")
_BAR_MS = 3_600_000


# ---------------------------------------------------------------------------
# Pure analysis core (fully testable, no I/O)
# ---------------------------------------------------------------------------

def run_analysis(
    breakout_rows: pd.DataFrame,
    candles_by_asset: Dict[str, pd.DataFrame],
    horizon: int = 1,
) -> Dict[str, object]:
    """Run the full Phase 9A experiment on in-memory data.

    Args:
        breakout_rows: completed breakout rows (``breakout_flag == 1``) for one
            or both assets. Must contain ``timestamp``, ``asset``,
            ``har_predicted_range``.
        candles_by_asset: ``{asset: candle DataFrame}`` covering every breakout
            timestamp (and a few bars after for forward returns).
        horizon: gate/analysis horizon (1, 2 or 3). Canonical gates use 1.

    Returns a dict with ``direction_df``, ``returns_df``, ``hit_rate``,
    ``dm``, ``temporal``, ``gates``, ``horizon``, ``assets``, ``n_events``.
    """
    horizon = int(horizon)
    if breakout_rows is None or len(breakout_rows) == 0:
        logger.warning("run_analysis: no breakout rows supplied")
        empty = {
            "hit_rate": {"hit_rate": 0.0, "n_events": 0, "n_correct": 0,
                         "by_asset": {}, "by_asset_n": {}, "by_regime": {}},
            "dm": {"dm_stat": 0.0, "p_value": 1.0, "n_obs": 0, "conclusion": "NO DATA"},
            "temporal": {"older_hit_rate": 0.0, "middle_hit_rate": 0.0,
                         "recent_hit_rate": 0.0, "is_stable": False},
            "gates": run_gate_checks(
                {"hit_rate": 0.0, "by_asset": {}, "by_asset_n": {}},
                {"older_hit_rate": 0.0, "middle_hit_rate": 0.0, "recent_hit_rate": 0.0},
                0, dm_dict=None),
            "horizon": horizon, "assets": [], "n_events": 0,
        }
        return empty

    assets = [str(a) for a in breakout_rows["asset"].unique()]
    direction_parts: List[pd.DataFrame] = []
    returns_parts: List[pd.DataFrame] = []
    for asset in assets:
        sub = breakout_rows[breakout_rows["asset"].astype(str) == asset]
        cands = candles_by_asset.get(asset)
        if cands is None or len(cands) == 0:
            logger.warning("run_analysis: no candle history for %s — skipping", asset)
            continue
        direction_parts.append(compute_breakout_direction(cands, sub))
        returns_parts.append(compute_forward_returns(cands, sub, horizons=(1, 2, 3)))

    direction_df = pd.concat(direction_parts, ignore_index=True) if direction_parts else pd.DataFrame()
    returns_df = pd.concat(returns_parts, ignore_index=True) if returns_parts else pd.DataFrame()

    hit_rate = compute_hit_rate(direction_df, returns_df, horizon)
    temporal = compute_temporal_stability(direction_df, returns_df, horizon)

    # Paired actual/predicted directions at the analysed horizon for the DM test.
    merged = merge_direction_returns(direction_df, returns_df, horizon)
    if merged.empty:
        dm = {"dm_stat": 0.0, "p_value": 1.0, "n_obs": 0, "conclusion": "NO DATA"}
    else:
        dm = compute_dm_statistic(
            merged["forward_direction"].to_numpy(),
            merged["breakout_direction"].to_numpy(),
        )

    gates = run_gate_checks(
        hit_rate, temporal, hit_rate["n_events"], dm_dict=dm, horizon=horizon)

    return {
        "direction_df": direction_df,
        "returns_df": returns_df,
        "hit_rate": hit_rate,
        "dm": dm,
        "temporal": temporal,
        "gates": gates,
        "horizon": horizon,
        "assets": assets,
        "n_events": int(hit_rate["n_events"]),
    }


def _fmt_pct(x: object) -> str:
    """Format a hit-rate-like value as ``XX.X%`` or ``N/A``."""
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "N/A"
        return f"{float(x) * 100.0:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_pvalue(p: object) -> str:
    """Format a p-value, always in scientific-ish notation."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "N/A"
    if p == 0.0:
        return "0.000e+00"
    return f"{p:.3e}"


def format_results(results: Dict[str, object]) -> str:
    """Render the results dict as the fixed Phase 9A report text."""
    hit = results.get("hit_rate", {}) or {}
    dm = results.get("dm", {}) or {}
    temporal = results.get("temporal", {}) or {}
    gates = results.get("gates", {}) or {}
    horizon = int(results.get("horizon", 1))
    assets = results.get("assets", []) or []

    by_asset = hit.get("by_asset", {}) or {}
    by_regime = hit.get("by_regime", {}) or {}
    n_events = int(hit.get("n_events", 0))

    if len(assets) > 1:
        asset_label = "both (" + ", ".join(assets) + ")"
    elif assets:
        asset_label = assets[0]
    else:
        asset_label = "none"

    def _gate(letter: str, label: str) -> str:
        ok = bool(gates.get(letter, False))
        return f"  G{letter[-1]} ({label}): {'PASS' if ok else 'FAIL'}"

    lines: List[str] = []
    lines.append("=" * 27)
    lines.append("PHASE 9A RESULTS")
    lines.append("=" * 27)
    lines.append(f"Asset: {asset_label}")
    lines.append(f"Breakout events: {n_events}")
    lines.append(f"Horizon: t+{horizon}")
    lines.append("")
    lines.append(f"Hit rate (overall): {_fmt_pct(hit.get('hit_rate'))}")
    lines.append(f"Hit rate (BTC): {_fmt_pct(by_asset.get('BTC/USDT'))}")
    lines.append(f"Hit rate (ETH): {_fmt_pct(by_asset.get('ETH/USDT'))}")
    lines.append(f"Hit rate (high regime): {_fmt_pct(by_regime.get('high'))}")
    lines.append(f"Hit rate (low regime): {_fmt_pct(by_regime.get('low'))}")
    lines.append("")
    dm_stat = dm.get("dm_stat", 0.0)
    dm_stat_str = f"{dm_stat:.2f}" if isinstance(dm_stat, (int, float)) else "N/A"
    lines.append(f"DM statistic: {dm_stat_str}")
    lines.append(f"p-value: {_fmt_pvalue(dm.get('p_value'))}")
    lines.append("")
    lines.append("Temporal stability:")
    lines.append(f"  Older: {_fmt_pct(temporal.get('older_hit_rate'))}")
    lines.append(f"  Middle: {_fmt_pct(temporal.get('middle_hit_rate'))}")
    lines.append(f"  Recent: {_fmt_pct(temporal.get('recent_hit_rate'))}")
    lines.append("")
    lines.append("Gate results:")
    lines.append(_gate("G1", "hit rate > 55%"))
    lines.append(_gate("G2", "DM p < 0.05   "))
    lines.append(_gate("G3", "both assets   "))
    lines.append(_gate("G4", "stable windows"))
    lines.append(_gate("G5", "no degradation"))
    lines.append(_gate("G6", "n >= 30       "))
    lines.append("")
    lines.append(f"VERDICT: {gates.get('verdict', 'CLOSED')}")
    lines.append("")
    lines.append(f"DM conclusion: {dm.get('conclusion', 'N/A')}")
    return "\n".join(lines)


def save_results(text: str, path: str = RESULTS_PATH) -> str:
    """Write the report to ``path`` as markdown. Returns the path."""
    import datetime as _dt
    header = (
        f"# Phase 9A Results\n\n"
        f"_Pre-registered hypothesis:_ breakout-bar candle direction persists "
        f"into the next 1/2/3 bars.\n\n"
        f"_Generated:_ {_dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n\n"
        f"```\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + text + "\n```\n")
    return path


# ---------------------------------------------------------------------------
# I/O providers (lazy-imported, isolated from the pure core)
# ---------------------------------------------------------------------------

def fetch_breakout_rows(db_url: str, assets: Iterable[str]) -> pd.DataFrame:
    """Fetch completed breakout rows (``breakout_flag = 1``) from Supabase.

    Returns a DataFrame with the ``har_predictions`` columns. Only rows with a
    non-null ``actual_range`` (i.e. the bar has closed) are returned — Phase 9A
    never reasons about bars that have not closed yet.
    """
    try:
        import psycopg  # type: ignore
        from psycopg.rows import dict_row  # type: ignore
    except ImportError as exc:  # pragma: no cover - postgres is optional in CI
        raise RuntimeError(
            "psycopg is required to read from Supabase "
            "(pip install 'psycopg[binary]')") from exc

    url = (db_url or "").strip().strip('"').strip("'")
    if not url:
        raise ValueError("db_url is empty")
    if not (url.startswith("postgres://") or url.startswith("postgresql://")):
        url = "postgresql://" + url

    assets = [str(a).upper() for a in assets]
    if not assets:
        return pd.DataFrame()

    query = (
        'SELECT id, "timestamp", asset, timeframe, har_predicted_range, '
        "coef_b0, coef_b1, coef_b2, coef_b3, n_obs, regime, actual_range, "
        "prediction_error, abs_prediction_error, breakout_flag, created_at "
        "FROM public.har_predictions "
        "WHERE breakout_flag = 1 AND actual_range IS NOT NULL "
        "AND asset = ANY(%s) "
        'ORDER BY "timestamp" ASC'
    )
    conn = psycopg.connect(url, row_factory=dict_row)
    try:
        with conn.cursor() as cur:
            cur.execute(query, (assets,))
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame(rows)


def fetch_candle_history(
    asset: str,
    timeframe: str = TIMEFRAME,
    n: int = DEFAULT_CANDLE_HISTORY,
    exchange=None,
    now_ms: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch the latest ``n`` *closed* candles for ``asset`` via CCXT.

    Returns a frame with columns ``[timestamp, open, high, low, close,
    volume]`` where ``timestamp`` is ISO8601 UTC. The still-forming bar is
    dropped so no future/incomplete data enters the analysis.
    """
    import ccxt  # type: ignore

    if exchange is None:
        exchange = ccxt.kucoin({"enableRateLimit": True,
                                "options": {"defaultType": "spot"}})
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    bar_ms = {"1h": _BAR_MS, "4h": 14_400_000, "1d": 86_400_000}.get(timeframe, _BAR_MS)

    import datetime as _dt
    raw = exchange.fetch_ohlcv(asset, timeframe, limit=n) or []
    rows = []
    for r in raw:
        if len(r) < 6:
            continue
        ts = int(r[0])
        if ts + bar_ms > now_ms:
            continue  # still forming
        iso = _dt.datetime.fromtimestamp(ts / 1000.0, tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append({"timestamp": iso, "open": float(r[1]), "high": float(r[2]),
                     "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])})
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phase9a",
        description="Phase 9A breakout-direction continuation experiment.",
    )
    p.add_argument("--db-url", default=os.environ.get("SUPABASE_DB_URL", ""),
                   help="Supabase connection string (or set SUPABASE_DB_URL).")
    p.add_argument("--asset", default="both", choices=["BTC/USDT", "ETH/USDT", "both"],
                   help="Asset to analyse (default: both).")
    p.add_argument("--horizon", type=int, default=1, choices=[1, 2, 3],
                   help="Forward horizon in bars (default: 1 = t+1).")
    p.add_argument("--dry-run", action="store_true",
                   help="Run without writing PHASE9A_RESULTS.md.")
    p.add_argument("--candles", type=int, default=DEFAULT_CANDLE_HISTORY,
                   help="Number of candles to fetch per asset.")
    return p


def main(argv: Optional[List[str]] = None) -> Dict[str, object]:
    """CLI entry point. Returns the analysis results dict."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    assets = ["BTC/USDT", "ETH/USDT"] if args.asset == "both" else [args.asset]
    breakout_rows = fetch_breakout_rows(args.db_url, assets)
    candles_by_asset = {a: fetch_candle_history(a, TIMEFRAME, args.candles) for a in assets}

    results = run_analysis(breakout_rows, candles_by_asset, horizon=args.horizon)
    text = format_results(results)
    print(text)
    if not args.dry_run:
        try:
            saved = save_results(text)
            print(f"\nResults saved to {saved}")
        except OSError as exc:
            logger.error("Could not write results file: %s", exc)
    return results


if __name__ == "__main__":  # pragma: no cover
    main()
