"""Pattern research runner (Agent 1 sandbox CLI).

Runs the full validation pipeline for one or all pattern families on KuCoin
data (``1h``/``4h``/``1d`` bars — see ``--timeframe``) and writes an honest
markdown report — including patterns that fail, and patterns that are skipped
for having fewer than 50 occurrences.

Usage::

    python -m sandbox.pattern_research.run_pattern_research \\
        --pattern all --asset both --horizon 1 --timeframe 1h \\
        --output sandbox/pattern_research/results

    # fade (mean-reversion) reading of Pattern 1 — deliberately NOT in "all"
    python -m sandbox.pattern_research.run_pattern_research \\
        --pattern momentum_fade --asset both --horizon 1 --timeframe 4h

    # fully offline (no exchange egress): point at saved CSVs
    python -m sandbox.pattern_research.run_pattern_research --pattern momentum \\
        --csv "BTC/USDT=cache/BTCUSDT_1h_730d.csv,ETH/USDT=cache/ETHUSDT_1h_730d.csv"

Never touches the main codebase. Never touches Supabase. No secrets needed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

try:  # allow both `python -m sandbox.pattern_research.run_pattern_research`
    from sandbox.pattern_research import data_loader, validator
    from sandbox.pattern_research.patterns import (
        candlestick, momentum, time_of_day, volume_spike)
    from sandbox.pattern_research.patterns.momentum import compute_forward_return
except ImportError:  # ...and `python run_pattern_research.py` from this folder
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pattern_research import data_loader, validator  # type: ignore
    from pattern_research.patterns import (  # type: ignore
        candlestick, momentum, time_of_day, volume_spike)
    from pattern_research.patterns.momentum import compute_forward_return  # type: ignore

logger = logging.getLogger(__name__)

PATTERN_CHOICES = ["momentum", "momentum_fade", "candlestick", "time", "volume", "all"]
ASSET_CHOICES = ["BTC/USDT", "ETH/USDT", "both"]
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SignalFunc = Callable[[pd.DataFrame], pd.Series]


# ---------------------------------------------------------------------------
# signal catalogue
# ---------------------------------------------------------------------------
def _static_signals(pattern: str) -> List[Tuple[str, str, SignalFunc]]:
    """(family, signal name, function) triples that need no fitting."""
    catalogue: Dict[str, List[Tuple[str, str, SignalFunc]]] = {
        "momentum": [
            ("momentum", "higher_high_higher_low (+1)", momentum.detect_higher_high_higher_low),
            ("momentum", "lower_low_lower_high (-1)", momentum.detect_lower_low_lower_high),
            ("momentum", "combined HH/HL + LL/LH", momentum.detect_momentum_combined),
        ],
        # Fade (mean-reversion) reading of the combined momentum signal.
        # Deliberately NOT part of "all": it scores the *same events* as
        # "momentum: combined" (with flipped signs), so bundling the two into
        # one report would let a single experiment read as two independent
        # findings. It runs through the exact same evaluate_signal path —
        # same compute_forward_return, same DM test, same G1-G6 gates, same
        # walk-forward — but only when explicitly requested via
        # --pattern momentum_fade.
        "momentum_fade": [
            ("momentum_fade", "combined fade (inverse of momentum)",
             momentum.detect_momentum_fade_combined),
        ],
        "candlestick": [
            ("candlestick", "bullish_engulfing (+1)", candlestick.detect_bullish_engulfing),
            ("candlestick", "bearish_engulfing (-1)", candlestick.detect_bearish_engulfing),
            ("candlestick", "doji (+1, tested as long)", candlestick.detect_doji),
            ("candlestick", "hammer (+1)", candlestick.detect_hammer),
        ],
        "volume": [
            ("volume", "volume_spike (+1/-1)", volume_spike.detect_volume_spike),
        ],
    }
    if pattern == "all":
        out: List[Tuple[str, str, SignalFunc]] = []
        for key in ("momentum", "candlestick", "volume"):
            out.extend(catalogue[key])
        return out
    return catalogue.get(pattern, [])


# ---------------------------------------------------------------------------
# horizon wording
# ---------------------------------------------------------------------------

# Singular/plural wording for the horizon label of each timeframe. A horizon
# is ALWAYS counted in bars; these strings translate bars -> clock time.
_HORIZON_UNIT = {"1h": "hour", "4h": "hour", "1d": "day"}
_HORIZON_BARS_PER_UNIT = {"1h": 1, "4h": 4, "1d": 1}


def horizon_label(horizon: int, timeframe: str) -> str:
    """Human wording for a horizon, e.g. ``"t+1 = 4 hours forward"``.

    A horizon is always counted in **bars** of the selected timeframe:
    4h bars with horizon 1 = 4 hours forward; 1d bars with horizon 1 = one day
    forward. Unknown timeframes fall back to neutral bar wording.
    """
    n = int(horizon)
    if timeframe in _HORIZON_UNIT:
        amount = n * _HORIZON_BARS_PER_UNIT[timeframe]
        unit = _HORIZON_UNIT[timeframe]
        return f"t+{n} = {amount} {unit}{'s' if amount != 1 else ''} forward"
    return f"t+{n} = {n} bar{'s' if n != 1 else ''} forward"


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def evaluate_signal(
    candles_by_asset: Dict[str, pd.DataFrame],
    signal_func: SignalFunc,
    horizon: int,
    n_splits: int = 3,
) -> Dict[str, object]:
    """Score one signal across the supplied assets.

    Returns a dict with the pooled results frame, per-asset occurrence counts,
    the G1–G6 gate report and a per-asset walk-forward. Assets with fewer than
    :data:`validator.MIN_OCCURRENCES` occurrences are reported but flagged.
    """
    frames: List[pd.DataFrame] = []
    counts: Dict[str, int] = {}
    walk: Dict[str, object] = {}

    for asset, candles in candles_by_asset.items():
        signal = signal_func(candles)
        events = compute_forward_return(candles, signal, horizon=horizon)
        events = events.copy()
        events["asset"] = asset
        events["timestamp"] = events.index
        events = events.reset_index(drop=True)
        counts[asset] = int(len(events))
        frames.append(events)
        walk[asset] = validator.run_walk_forward(
            candles, signal_func, n_splits=n_splits, horizon=horizon)

    results_df = (pd.concat(frames).sort_values("timestamp")
                  if frames else pd.DataFrame(columns=["signal", "forward_return",
                                                       "correct", "asset", "timestamp"]))
    total = int(len(results_df))
    skipped = total < validator.MIN_OCCURRENCES

    gates = None if skipped else validator.run_gate_checks(results_df)
    return {
        "results_df": results_df,
        "counts": counts,
        "n_events": total,
        "skipped": skipped,
        "skip_reason": (
            f"only {total} occurrences in the sample (< {validator.MIN_OCCURRENCES} "
            "minimum) — not worth testing; no gates were run (sandbox rule 7)"
            if skipped else None),
        "gates": gates,
        "walk_forward": walk,
    }


def evaluate_time_of_day(
    candles_by_asset: Dict[str, pd.DataFrame],
    horizon: int,
    min_win_rate: float = time_of_day.DEFAULT_MIN_WIN_RATE,
    train_frac: float = 0.5,
    n_splits: int = 3,
) -> Dict[str, object]:
    """Time-of-day bias, selected in-sample-free.

    The "best hours" are learned on the **first ``train_frac``** of each asset's
    history and then traded on the remaining, unseen bars. Learning the hours on
    the whole sample and scoring them on the same sample would be look-ahead
    (data snooping) and is deliberately not done.
    """
    frames: List[pd.DataFrame] = []
    counts: Dict[str, int] = {}
    walk: Dict[str, object] = {}
    per_asset: Dict[str, object] = {}

    for asset, candles in candles_by_asset.items():
        cut = int(len(candles) * train_frac)
        train, test = candles.iloc[:cut], candles.iloc[cut:]
        hourly_train = time_of_day.compute_hourly_bias(train)
        best_hours = time_of_day.find_best_hours(hourly_train, min_win_rate=min_win_rate)

        per_asset[asset] = {
            "hourly_full": time_of_day.compute_hourly_bias(candles),
            "daily_full": time_of_day.compute_daily_bias(candles),
            "hourly_train": hourly_train,
            "best_hours_train": best_hours,
            "train_end": str(train.index[-1]) if len(train) else None,
            "test_start": str(test.index[0]) if len(test) else None,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        }

        func: SignalFunc = lambda c, _h=best_hours: time_of_day.build_hour_signal(c, _h)
        signal = func(test)
        events = compute_forward_return(test, signal, horizon=horizon)
        events = events.copy()
        events["asset"] = asset
        events["timestamp"] = events.index
        events = events.reset_index(drop=True)
        counts[asset] = int(len(events))
        frames.append(events)
        walk[asset] = validator.run_walk_forward(test, func, n_splits=n_splits,
                                                 horizon=horizon)

    results_df = (pd.concat(frames).sort_values("timestamp")
                  if frames else pd.DataFrame(columns=["signal", "forward_return",
                                                       "correct", "asset", "timestamp"]))
    total = int(len(results_df))
    skipped = total < validator.MIN_OCCURRENCES
    gates = None if skipped else validator.run_gate_checks(results_df)
    return {
        "results_df": results_df,
        "counts": counts,
        "n_events": total,
        "skipped": skipped,
        "skip_reason": (
            f"only {total} out-of-sample occurrences (< {validator.MIN_OCCURRENCES}); "
            "no hour cleared the selection filter on the training half, or the test "
            "window is too short (sandbox rule 7)" if skipped else None),
        "gates": gates,
        "walk_forward": walk,
        "per_asset": per_asset,
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _pct(x) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_box(title: str, evaluation: Dict[str, object], horizon: int) -> str:
    """Phase 9A-style fixed-width results box for one signal."""
    W = 58  # wide enough for the longest signal title, e.g. the fade reading

    def row(content: str = "") -> str:
        content = content[:W]
        return "|" + content + " " * (W - len(content)) + "|"

    def divider() -> str:
        return "+" + "-" * W + "+"

    lines = [divider(), row(f" {title}"), divider()]
    counts = evaluation["counts"]
    lines.append(row(f" Horizon:          t+{horizon}"))
    lines.append(row(f" Occurrences:      {evaluation['n_events']}"))
    for asset, n in counts.items():
        lines.append(row(f"   {asset:<12} {n}"))

    if evaluation["skipped"]:
        lines.append(divider())
        lines.append(row(" STATUS: SKIPPED (< 50 occurrences)"))
        lines.append(divider())
        return "\n".join(lines)

    gates = evaluation["gates"]
    details = gates["details"]
    hit, dm, temporal = details["hit_rate"], details["dm"], details["temporal"]
    by_asset = hit["by_asset"]
    lines.append(divider())
    lines.append(row(f" Hit rate (all):   {_pct(hit['overall_hit_rate'])}"))
    for asset in validator.BOTH_ASSETS:
        if asset in by_asset:
            lines.append(row(f" Hit rate ({asset.split('/')[0]}):   {_pct(by_asset[asset])}"))
    mean_ret = float(evaluation["results_df"]["forward_return"].mean())
    lines.append(row(f" Mean fwd return:  {mean_ret * 100:+.4f}%"))
    lines.append(divider())
    dm_stat = dm.get("dm_stat", 0.0)
    lines.append(row(f" DM statistic:     {dm_stat:.3f}" if isinstance(dm_stat, (int, float))
                     else " DM statistic:     N/A"))
    try:
        lines.append(row(f" p-value:          {float(dm['p_value']):.4g}"))
    except (TypeError, ValueError):
        lines.append(row(" p-value:          N/A"))
    lines.append(divider())
    lines.append(row(" Temporal windows (pooled events):"))
    lines.append(row(f"   Older:   {_pct(temporal['older'])}"))
    lines.append(row(f"   Middle:  {_pct(temporal['middle'])}"))
    lines.append(row(f"   Recent:  {_pct(temporal['recent'])}"))
    lines.append(divider())
    pf = lambda b: "PASS" if b else "FAIL"
    lines.append(row(" GATES:"))
    lines.append(row(f" G1 (hit > 55%, both):   {pf(gates['G1'])}"))
    lines.append(row(f" G2 (DM p < 0.05):       {pf(gates['G2'])}"))
    lines.append(row(f" G3 (both assets > 50%): {pf(gates['G3'])}"))
    lines.append(row(f" G4 (stability):         {pf(gates['G4'])}"))
    lines.append(row(f" G5 (no degradation):    {pf(gates['G5'])}"))
    lines.append(row(f" G6 (n >= 30 per asset): {pf(gates['G6'])}"))
    lines.append(divider())
    lines.append(row(f" VERDICT: {gates['verdict']}"))
    lines.append(divider())
    return "\n".join(lines)


def _walk_forward_md(walk: Dict[str, object]) -> str:
    rows = ["| Asset | Older | Middle | Recent | Stable | Degrading |",
            "|---|---|---|---|---|---|"]
    for asset, wf in walk.items():
        rows.append(
            f"| {asset} | {_pct(wf['older'])} | {_pct(wf['middle'])} | "
            f"{_pct(wf['recent'])} | {'yes' if wf['is_stable'] else 'no'} | "
            f"{'yes' if wf['degrading'] else 'no'} |")
    return "\n".join(rows)


def build_report(
    pattern: str, asset_label: str, horizon: int, days: int,
    evaluations: List[Tuple[str, Dict[str, object]]],
    data_summary: Dict[str, object],
    time_context: Optional[Dict[str, object]] = None,
    source: str = "CCXT KuCoin public API (spot)",
    timeframe: str = "1h",
) -> str:
    """Assemble the full markdown report (honest: failures included)."""
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    md: List[str] = []
    md.append(f"# Pattern Research Results — {pattern}")
    md.append("")
    md.append(f"_Generated:_ {now}  ")
    md.append(f"_Source:_ {source}, {timeframe} OHLCV, last {days} days  ")
    md.append(f"_Assets:_ {asset_label}  ")
    md.append(f"_Timeframe:_ {timeframe}  ")
    md.append(f"_Horizon:_ {horizon_label(horizon, timeframe)}  ")
    md.append("_Sandbox:_ `sandbox/pattern_research` — no DB, no secrets, "
              "no contact with the production system.")
    md.append("")
    md.append("## Data")
    md.append("")
    md.append("| Asset | Bars | Bar spacing | First bar (UTC) | Last bar (UTC) |")
    md.append("|---|---|---|---|---|")
    for asset, info in data_summary.items():
        spacing = info.get("spacing") or "unknown"
        md.append(f"| {asset} | {info['n']} | {spacing} | {info['start']} | {info['end']} |")
    md.append("")

    # A --csv file can be saved at a different bar spacing than --timeframe
    # claims; that is possible only in the offline path (a live fetch always
    # requests the flag's timeframe), so warn loudly when detected.
    mismatched = sorted(
        asset for asset, info in data_summary.items()
        if info.get("spacing") not in (None, timeframe))
    if mismatched:
        detail = ", ".join(
            f"{a}: {data_summary[a]['spacing']}" for a in mismatched)
        md.append(f"> **Warning:** the loaded data's bar spacing ({detail}) "
                  f"contradicts `--timeframe {timeframe}`. This is possible "
                  "with `--csv`; all horizons below are counted in **bars of "
                  "the loaded data**, and the clock-time wording in the "
                  "horizon label follows the flag, not the file. Re-export "
                  "the CSV at the right spacing or fix `--timeframe`.")
        md.append("")
    md.append("## Method")
    md.append("")
    md.append("* Every detector is `.shift(1)`-ed: `signal[t]` reflects a pattern that "
              "**completed at bar t-1**, so it is known before bar `t` opens.")
    md.append(f"* `forward_return[t] = close[t+{horizon}]/close[t] - 1` — entry at the "
              "close of the signal bar, exit `horizon` bars later. No look-ahead. "
              "**A horizon is always counted in bars** of the selected timeframe.")
    md.append("* `correct = 1` when `sign(forward_return) == sign(signal)`.")
    md.append("* Diebold-Mariano: one-sided vs a 50/50 coin flip, Newey-West HAC with "
              "3 lags (identical arithmetic to Phase 9A `dm_test.py`).")
    md.append("* Gates G1–G6 as pre-registered in Phase 9A; all-or-nothing verdict.")
    md.append(f"* Patterns with fewer than {validator.MIN_OCCURRENCES} occurrences are "
              "skipped, not tested (rule 6/7).")
    md.append("")

    if time_context:
        md.append("## Time-of-day context (descriptive)")
        md.append("")
        for asset, ctx in time_context.items():
            hourly = ctx["hourly_full"].sort_values("win_rate", ascending=False)
            md.append(f"### {asset} — hourly bias (full sample, sorted by win rate)")
            md.append("")
            md.append("| Hour (UTC) | Mean return | Win rate | N |")
            md.append("|---|---|---|---|")
            for _, r in hourly.iterrows():
                md.append(f"| {int(r['hour']):02d} | {float(r['mean_return']) * 100:+.4f}% | "
                          f"{_pct(r['win_rate'])} | {int(r['n_observations'])} |")
            md.append("")
            daily = ctx["daily_full"]
            md.append(f"### {asset} — day-of-week bias (full sample)")
            md.append("")
            md.append("| Day | Mean return | Win rate | N |")
            md.append("|---|---|---|---|")
            for _, r in daily.iterrows():
                md.append(f"| {r['day_name']} | {float(r['mean_return']) * 100:+.4f}% | "
                          f"{_pct(r['win_rate'])} | {int(r['n_observations'])} |")
            md.append("")
            md.append(f"Best hours learned on the training half "
                      f"(bars up to {ctx['train_end']}, n={ctx['n_train']}): "
                      f"`{ctx['best_hours_train']}` — evaluated only on the held-out "
                      f"bars from {ctx['test_start']} onwards (n={ctx['n_test']}).")
            md.append("")

    md.append("## Results")
    md.append("")
    for title, ev in evaluations:
        md.append(f"### {title}")
        md.append("")
        md.append("```")
        md.append(format_box(title, ev, horizon))
        md.append("```")
        md.append("")
        if ev["skipped"]:
            md.append(f"**SKIPPED** — {ev['skip_reason']}")
            md.append("")
            continue
        gates = ev["gates"]
        details = gates["details"]
        md.append(f"DM conclusion: {details['dm']['conclusion']}")
        md.append("")
        md.append("Walk-forward (signal re-run inside each chronological third):")
        md.append("")
        md.append(_walk_forward_md(ev["walk_forward"]))
        md.append("")
        for note in gates.get("notes", []):
            md.append(f"> Note: {note}")
        if gates.get("notes"):
            md.append("")

    md.append("## Summary")
    md.append("")
    md.append("| Signal | N | Hit rate | DM p | Verdict |")
    md.append("|---|---|---|---|---|")
    for title, ev in evaluations:
        if ev["skipped"]:
            md.append(f"| {title} | {ev['n_events']} | — | — | SKIPPED (<50) |")
            continue
        d = ev["gates"]["details"]
        md.append(f"| {title} | {ev['n_events']} | {_pct(d['hit_rate']['overall_hit_rate'])} | "
                  f"{float(d['dm']['p_value']):.4g} | {ev['gates']['verdict']} |")
    md.append("")
    passing = [t for t, e in evaluations
               if not e["skipped"] and e["gates"]["all_pass"]]
    if passing:
        md.append(f"**{len(passing)} signal(s) cleared every gate:** "
                  + ", ".join(f"`{p}`" for p in passing) + ".")
    else:
        md.append("**No signal cleared all six gates.** Every pattern tested above is "
                  "reported as CLOSED. This is the expected outcome for simple public "
                  f"patterns on liquid {timeframe} crypto data and is documented here "
                  "rather than buried — negative results are results.")
    md.append("")
    return "\n".join(md)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_csv_arg(raw: Optional[str]) -> Dict[str, str]:
    """``"BTC/USDT=a.csv,ETH/USDT=b.csv"`` → mapping (or ``{}``)."""
    if not raw:
        return {}
    mapping: Dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("--csv expects ASSET=path[,ASSET=path]")
        asset, path = part.split("=", 1)
        mapping[asset.strip().upper()] = path.strip()
    return mapping


def load_assets(assets: List[str], timeframe: str, days: int, csv_map: Dict[str, str],
                use_cache: bool, cache_dir: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for asset in assets:
        out[asset] = data_loader.load_candles(
            asset, timeframe=timeframe, days=days, csv=csv_map.get(asset),
            use_cache=use_cache, cache_dir=cache_dir)
    return out


def run(pattern: str, assets: List[str], horizon: int, candles_by_asset: Dict[str, pd.DataFrame],
        days: int, min_win_rate: float = time_of_day.DEFAULT_MIN_WIN_RATE,
        n_splits: int = 3,
        source: str = "CCXT KuCoin public API (spot)",
        timeframe: str = "1h",
        ) -> Tuple[str, List[Tuple[str, Dict[str, object]]]]:
    """Core (testable) run: returns ``(markdown_report, evaluations)``."""
    evaluations: List[Tuple[str, Dict[str, object]]] = []
    for family, name, func in _static_signals(pattern):
        title = f"{family}: {name}"
        evaluations.append((title, evaluate_signal(candles_by_asset, func, horizon,
                                                   n_splits=n_splits)))

    time_context = None
    if pattern in ("time", "all"):
        ev = evaluate_time_of_day(candles_by_asset, horizon, min_win_rate=min_win_rate,
                                  n_splits=n_splits)
        time_context = ev.pop("per_asset")
        evaluations.append(("time_of_day: best-hours (out-of-sample)", ev))

    data_summary = {
        asset: {"n": int(len(c)),
                "start": str(c.index[0]) if len(c) else "—",
                "end": str(c.index[-1]) if len(c) else "—",
                "spacing": data_loader.infer_timeframe(c)}
        for asset, c in candles_by_asset.items()
    }
    for asset, info in data_summary.items():
        spacing = info["spacing"]
        if spacing is not None and spacing != timeframe:
            logger.warning(
                "%s: loaded data's bar spacing looks like %s but --timeframe %s "
                "was requested (possible with --csv); horizons are counted in "
                "bars of the loaded data", asset, spacing, timeframe)
    asset_label = "both (BTC/USDT + ETH/USDT)" if len(assets) > 1 else assets[0]
    report = build_report(pattern, asset_label, horizon, days, evaluations,
                          data_summary, time_context, source=source,
                          timeframe=timeframe)
    return report, evaluations


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pattern research sandbox — KuCoin public data, no DB, no secrets.")
    parser.add_argument("--pattern", default="all", choices=PATTERN_CHOICES,
                        help="Pattern family to test (default: all).")
    parser.add_argument("--asset", default="both", choices=ASSET_CHOICES,
                        help="Asset to test (default: both).")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 2, 3],
                        help="Forward horizon in bars (default: 1).")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                        help="Results directory for the markdown report.")
    parser.add_argument("--days", type=int, default=data_loader.DEFAULT_DAYS,
                        help="History length in days (default: 730).")
    parser.add_argument("--timeframe", default=data_loader.DEFAULT_TIMEFRAME,
                        choices=list(data_loader.SUPPORTED_TIMEFRAMES),
                        help="Candle timeframe (default: 1h). A horizon is "
                             "always counted in bars of this timeframe.")
    parser.add_argument("--csv", default=None,
                        help="Offline mode: 'BTC/USDT=path.csv,ETH/USDT=path.csv'.")
    parser.add_argument("--cache-dir", default=data_loader.CACHE_DIR,
                        help="Where fetched candles are cached as CSV.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore and do not write the CSV cache.")
    parser.add_argument("--min-win-rate", type=float, default=time_of_day.DEFAULT_MIN_WIN_RATE,
                        help="Threshold for find_best_hours (default: 0.55).")
    parser.add_argument("--splits", type=int, default=3,
                        help="Walk-forward splits (default: 3).")
    parser.add_argument("--quiet", action="store_true", help="Do not print the report.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    assets = list(data_loader.DEFAULT_ASSETS) if args.asset == "both" else [args.asset]
    csv_map = _parse_csv_arg(args.csv)
    try:
        candles_by_asset = load_assets(assets, args.timeframe, args.days, csv_map,
                                       use_cache=not args.no_cache,
                                       cache_dir=args.cache_dir)
    except Exception as exc:  # network / data problems must not crash with a traceback
        print(f"Data load failed: {exc}", file=sys.stderr)
        print("Hint: this sandbox needs public egress to api.kucoin.com. Offline? "
              "Use --csv 'BTC/USDT=path.csv,ETH/USDT=path.csv'.", file=sys.stderr)
        return 2

    source = ("local CSV file(s) — verify their provenance"
              if csv_map else "CCXT KuCoin public API (spot)")
    report, _ = run(args.pattern, assets, args.horizon, candles_by_asset, args.days,
                    min_win_rate=args.min_win_rate, n_splits=args.splits,
                    source=source, timeframe=args.timeframe)

    os.makedirs(args.output, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    asset_slug = "both" if len(assets) > 1 else assets[0].replace("/", "")
    path = os.path.join(
        args.output,
        f"pattern_research_{args.pattern}_{asset_slug}_{args.timeframe}"
        f"_h{args.horizon}_{stamp}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report)

    if not args.quiet:
        print(report)
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
