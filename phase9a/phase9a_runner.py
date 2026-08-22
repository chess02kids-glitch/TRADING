"""Phase 9A CLI — load a CSV of forward-return data and print the gate report.

This runner is DB-free: it reads a CSV exported from
``forward_return_logger.get_phase9a_data()`` and runs the full pre-registered
experiment (hit rates → Diebold-Mariano → temporal stability → G1–G6 gates).

Usage::

    python -m phase9a.phase9a_runner --data-file phase9a_data.csv \\
        --asset both --horizon 1 --output phase9a/PHASE9A_RESULTS.md

CLI args:
    --data-file   path to the CSV of Phase 9A forward-return data (required)
    --asset       "BTC/USDT", "ETH/USDT", or "both" (default: both)
    --horizon     1, 2, or 3 (default: 1, the primary gate horizon)
    --output      optional path to save the report as markdown

Flow: load → filter by asset → check G6 (≥30 events/asset) → analyze → print
the results box → optionally save. If G6 fails it prints "Not enough data yet"
and exits cleanly (the system is still collecting data).
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

import pandas as pd

from phase9a.continuation_tester import (
    BOTH_ASSETS,
    PRIMARY_HORIZON,
    compute_temporal_stability,
    run_all_gate_checks,
)
from phase9a.direction_calculator import compute_hit_rate

MIN_EVENTS_PER_ASSET = 30


def load_data(path: str) -> pd.DataFrame:
    """Load the Phase 9A CSV into a DataFrame."""
    return pd.read_csv(path)


def filter_asset(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Keep rows for ``asset`` ('both' keeps BTC + ETH)."""
    if asset == "both":
        return df[df["asset"].isin(list(BOTH_ASSETS))].copy() if "asset" in df.columns else df.copy()
    return df[df["asset"] == asset].copy() if "asset" in df.columns else df.copy()


def _per_asset_counts(df: pd.DataFrame, horizon: int) -> Dict[str, int]:
    sub = df.copy()
    if "horizon" in sub.columns:
        sub = sub[sub["horizon"] == horizon]
    sub = sub.dropna(subset=["breakout_direction", "forward_direction"])
    if sub.empty:
        return {}
    return {str(k): int(v) for k, v in sub.groupby("asset").size().items()}


def _pct(x) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_report(results: Dict[str, object]) -> str:
    """Render the fixed-width Phase 9A results box."""
    W = 30

    def row(content: str) -> str:
        content = content[:W]
        return "║" + content + " " * (W - len(content)) + "║"

    def divider() -> str:
        return "╠" + "═" * W + "╣"

    hit = results["hit_rate"]
    dm = results["dm"]
    temporal = results["temporal"]
    gates = results["gates"]
    by_asset = hit.get("by_asset", {})
    horizon = results["horizon"]

    pf = lambda b: "PASS" if b else "FAIL"
    lines: List[str] = []
    lines.append("╔" + "═" * W + "╗")
    lines.append(row("   PHASE 9A RESULTS"))
    lines.append(divider())
    lines.append(row(f" Asset:           {results['asset_label']}"))
    lines.append(row(f" Breakout events: {hit.get('n_events', 0)}"))
    lines.append(row(f" Horizon:         t+{horizon}"))
    lines.append(divider())
    lines.append(row(f" Hit rate (all):  {_pct(hit.get('overall_hit_rate'))}"))
    lines.append(row(f" Hit rate (BTC):  {_pct(by_asset.get('BTC/USDT'))}"))
    lines.append(row(f" Hit rate (ETH):  {_pct(by_asset.get('ETH/USDT'))}"))
    lines.append(divider())
    dm_stat = dm.get("dm_stat", 0.0)
    dm_stat_s = f"{dm_stat:.2f}" if isinstance(dm_stat, (int, float)) else "N/A"
    lines.append(row(f" DM statistic:    {dm_stat_s}"))
    try:
        lines.append(row(f" p-value:         {float(dm.get('p_value', 1.0)):.3e}"))
    except (TypeError, ValueError):
        lines.append(row(" p-value:         N/A"))
    lines.append(divider())
    lines.append(row(" Temporal windows:"))
    lines.append(row(f"   Older:   {_pct(temporal.get('older'))}"))
    lines.append(row(f"   Middle:  {_pct(temporal.get('middle'))}"))
    lines.append(row(f"   Recent:  {_pct(temporal.get('recent'))}"))
    lines.append(divider())
    lines.append(row(" GATES:"))
    lines.append(row(f" G1 (hit > 55%):   {pf(gates.get('G1'))}"))
    lines.append(row(f" G2 (DM p<0.05):   {pf(gates.get('G2'))}"))
    lines.append(row(f" G3 (both assets): {pf(gates.get('G3'))}"))
    lines.append(row(f" G4 (stability):   {pf(gates.get('G4'))}"))
    lines.append(row(f" G5 (no degrade):  {pf(gates.get('G5'))}"))
    lines.append(row(f" G6 (n >= 30):     {pf(gates.get('G6'))}"))
    lines.append(divider())
    lines.append(row(f" VERDICT: {gates.get('verdict', 'CLOSED')}"))
    lines.append("╚" + "═" * W + "╝")
    return "\n".join(lines)


def analyze(df: pd.DataFrame, asset: str, horizon: int) -> Dict[str, object]:
    """Run the full analysis on an in-memory DataFrame (testable core)."""
    sub = filter_asset(df, asset)
    asset_label = "both (BTC/USDT + ETH/USDT)" if asset == "both" else asset
    gates = run_all_gate_checks(sub, horizon=horizon)
    return {
        "asset_label": asset_label,
        "horizon": horizon,
        "hit_rate": gates["details"]["hit_rate"],
        "dm": gates["details"]["dm"],
        "temporal": gates["details"]["temporal"],
        "gates": gates,
        "g6_ok": all(v >= MIN_EVENTS_PER_ASSET
                     for v in _per_asset_counts(sub, horizon).values())
        if _per_asset_counts(sub, horizon) else False,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 9A breakout-direction analysis")
    parser.add_argument("--data-file", required=True, help="CSV of Phase 9A data.")
    parser.add_argument("--asset", default="both",
                        choices=["BTC/USDT", "ETH/USDT", "both"],
                        help="Asset to analyse (default: both).")
    parser.add_argument("--horizon", type=int, default=PRIMARY_HORIZON,
                        choices=[1, 2, 3], help="Forward horizon (default: 1).")
    parser.add_argument("--output", default=None, help="Optional markdown output path.")
    args = parser.parse_args(argv)

    df = load_data(args.data_file)
    results = analyze(df, args.asset, args.horizon)

    # G6 pre-check: still collecting data?
    counts = _per_asset_counts(filter_asset(df, args.asset), args.horizon)
    needed = [a for a in (list(BOTH_ASSETS) if args.asset == "both" else [args.asset])
              if counts.get(a, 0) < MIN_EVENTS_PER_ASSET]
    if needed:
        worst = min((counts.get(a, 0) for a in needed), default=0)
        print(f"Not enough data yet ({worst} of {MIN_EVENTS_PER_ASSET} needed for "
              f"{', '.join(needed)}).")
        if args.output:
            _save(args.output, format_report(results))
        return 0

    text = format_report(results)
    print(text)
    if args.output:
        _save(args.output, text)
    return 0


def _save(path: str, text: str) -> None:
    import datetime as _dt
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Phase 9A Results\n\n")
        fh.write(f"_Generated:_ {_dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n\n")
        fh.write("```\n" + text + "\n```\n")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
