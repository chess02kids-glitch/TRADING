"""CLI: build, backtest, walk-forward, grade and sweep the Fear & Greed strategy.

Usage:
    .venv/bin/python -m fear_greed_strategy.run_backtest --asset BTC \
        --fear-threshold 25 --greed-threshold 75 --stop-loss 0.05 --max-hold 14
"""
from __future__ import annotations

import argparse
import itertools
import time
from datetime import datetime, timezone
from pathlib import Path

from . import data_fetcher as dfl
from . import signal_generator as sg
from .backtester import backtest, walk_forward
from .grader import grade_strategy, period_consistent

PACKAGE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PACKAGE_DIR / "results"

SWEEP_FEAR = [15, 20, 25, 30]
SWEEP_GREED = [70, 75, 80, 85]
SWEEP_STOP = [0.03, 0.05, 0.07, 0.10]


def line(width: int = 54, ch: str = "═") -> str:
    return ch * width


def box(rows: list[tuple[str, str]], width: int = 54, title: str | None = None) -> str:
    out = ["╔" + line(width, "═") + "╗"]
    if title:
        pad = width - len(title)
        out.append("║" + " " * (pad // 2) + title + " " * (pad - pad // 2) + "║")
        out.append("╠" + line(width, "═") + "╣")
    for k, v in rows:
        kv = f" {k:<17}{v}"
        out.append("║" + kv.ljust(width)[:width] + "║")
    out.append("╚" + line(width, "═") + "╝")
    return "\n".join(out)


def run_one(df, fear, greed, stop, max_hold, fee):
    sig = sg.generate_signals(df, fear_threshold=fear, greed_threshold=greed)
    stats = backtest(sig, fee_one_way=fee, stop_loss_pct=stop, max_hold_days=max_hold)
    wf = walk_forward(sig, 3, fee_one_way=fee, stop_loss_pct=stop, max_hold_days=max_hold)
    grade = grade_strategy(stats, wf)
    return stats, wf, grade


def main() -> None:
    ap = argparse.ArgumentParser(description="Fear & Greed Contrarian backtest")
    ap.add_argument("--asset", default="BTC", choices=["BTC", "ETH"])
    ap.add_argument("--fear-threshold", type=float, default=25)
    ap.add_argument("--greed-threshold", type=float, default=75)
    ap.add_argument("--stop-loss", type=float, default=0.05)
    ap.add_argument("--max-hold", type=float, default=14)
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--fng-limit", type=int, default=1000,
                    help="days of F&G to fetch (1000 covers the 730d price window)")
    ap.add_argument("--no-sweep", action="store_true", help="skip the parameter sweep")
    args = ap.parse_args()

    asset = args.asset.upper()
    t0 = time.time()

    # 1. fetch all data ---------------------------------------------------- #
    print(f"=== Fear & Greed Contrarian - {asset} ===")
    fng = dfl.fetch_fear_greed(limit=args.fng_limit)
    dom = dfl.fetch_btc_dominance_history(days=730)
    prices = dfl.load_price_data(asset)
    df = dfl.merge_all_data(prices, fng, dom)
    dom_note = "ON" if df["dominance_pct"].notna().any() else "OFF (no free history)"

    # 2/3/4. signals, backtest, walk-forward (base parameters) -------------- #
    stats, wf, grade = run_one(
        df, args.fear_threshold, args.greed_threshold,
        args.stop_loss, args.max_hold, args.fee,
    )

    # 5/6. grade + report --------------------------------------------------- #
    all_ok = all(period_consistent(s) for s in wf)
    n_ok = sum(period_consistent(s) for s in wf)

    rows = [
        ("Asset:", f"{asset}/USDT"),
        ("Period:", f"{stats['start']} to {stats['end']}"),
        ("Fear threshold:", f"< {args.fear_threshold:g}"),
        ("Greed threshold:", f"> {args.greed_threshold:g}"),
        ("Dominance filter:", dom_note),
    ]
    rows += [
        ("", ""),
        ("FULL PERIOD RESULTS:", ""),
        ("Strategy return:", f"{stats['total_return_pct']:.1f}%"),
        ("Buy-hold return:", f"{stats['buy_hold_return_pct']:.1f}%"),
        ("Beat buy-hold:", "YES" if stats["beat_buy_hold"] else "NO"),
        ("Sharpe ratio:", f"{stats['annualized_sharpe']:.2f}"),
        ("Max drawdown:", f"{stats['max_drawdown_pct']:.1f}%"),
        ("Profit factor:", f"{stats['profit_factor']:.2f}"),
        ("Win rate:", f"{stats['win_rate_pct']:.1f}%"),
        ("Total trades:", str(stats["num_trades"])),
        ("Avg hold (days):", f"{stats['avg_hold_days']:.1f}"),
        ("Best trade:", f"{stats['best_trade_pct']:.1f}%"),
        ("Worst trade:", f"{stats['worst_trade_pct']:.1f}%"),
        ("Exposure:", f"{stats['exposure_pct']:.1f}% of bars"),
    ]
    rows += [("", ""), ("WALK-FORWARD:", "")]
    for k, s in enumerate(wf):
        label = ["Period 1 (older)", "Period 2 (mid)  ", "Period 3 (recent)"][k]
        rows.append((label, f"{s['total_return_pct']:6.1f}%  PF={s['profit_factor']:.2f}  "
                            f"trades={s['num_trades']}  {'OK' if period_consistent(s) else 'FAIL'}"))
    rows += [("Consistent:", f"{'YES' if all_ok else 'NO'} ({n_ok}/3 periods profitable)")]

    report = box(rows, title="FEAR & GREED CONTRARIAN STRATEGY")
    grade_block = box(
        [
            ("GRADE:", grade["grade"]),
            ("SCORE:", f"{grade['score']}/100"),
            ("", ""),
            ("RECOMMENDATION:", grade["recommendation"]),
        ],
        title="VERDICT",
    )
    print()
    print(report)
    print()
    print(grade_block)
    print()
    print("PASSED:")
    for p in grade["pass_criteria"]:
        print(f"  [x] {p}")
    print("FAILED:")
    for f_ in grade["fail_criteria"]:
        print(f"  [ ] {f_}")

    # 7. parameter sweep ---------------------------------------------------- #
    sweep_lines: list[str] = []
    best = None
    if not args.no_sweep:
        print(f"\nSweeping {len(SWEEP_FEAR)*len(SWEEP_GREED)*len(SWEEP_STOP)} combinations ...")
        cache: dict[tuple[float, float], object] = {}
        for fear, greed in itertools.product(SWEEP_FEAR, SWEEP_GREED):
            sig = sg.generate_signals(df, fear_threshold=fear, greed_threshold=greed)
            cache[(fear, greed)] = sig
        for fear, greed, stop in itertools.product(SWEEP_FEAR, SWEEP_GREED, SWEEP_STOP):
            sig = cache[(fear, greed)]
            s = backtest(sig, fee_one_way=args.fee, stop_loss_pct=stop,
                         max_hold_days=args.max_hold)
            w = walk_forward(sig, 3, fee_one_way=args.fee, stop_loss_pct=stop,
                             max_hold_days=args.max_hold)
            g = grade_strategy(s, w)
            ok = sum(period_consistent(x) for x in w)
            sweep_lines.append(
                f"| {fear} | {greed} | {stop:.2f} | {s['total_return_pct']:.1f}% | "
                f"{s['buy_hold_return_pct']:.1f}% | {s['profit_factor']:.2f} | "
                f"{s['annualized_sharpe']:.2f} | {s['max_drawdown_pct']:.1f}% | "
                f"{s['num_trades']} | {ok}/3 | {g['grade']} | {g['score']} |"
            )
            cand = (g["score"], g["grade"] != "F" and g["grade"] != "D", s["total_return_pct"])
            if best is None or cand > best[0]:
                best = (cand, fear, greed, stop, g, s)
        if best:
            _, bf, bg, bs, bg_, bs_ = best
            print()
            print(f"BEST PARAMS: fear<{bf} greed>{bg} stop={bs:.0%}")
            print(f"             Grade: {bg_['grade']} Score: {bg_['score']}/100 "
                  f"(return {bs_['total_return_pct']:.1f}%, PF {bs_['profit_factor']:.2f})")

    # save results ---------------------------------------------------------- #
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"RESULTS_{asset}_{stamp}.md"
    with out_path.open("w") as fh:
        fh.write(f"# Fear & Greed Contrarian - {asset}/USDT\n\n")
        fh.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
        fh.write("```\n" + report + "\n\n" + grade_block + "\n```\n\n")
        fh.write(f"**GRADE {grade['grade']} - score {grade['score']}/100 - "
                 f"{grade['recommendation']}**\n\n")
        fh.write(f"Dominance filter: {dom_note}\n\n")
        fh.write("## Grading detail\n\nPassed:\n")
        fh.write("".join(f"- [x] {p}\n" for p in grade["pass_criteria"]))
        fh.write("\nFailed:\n")
        fh.write("".join(f"- [ ] {f_}\n" for f_ in grade["fail_criteria"]))
        fh.write("\n## Exit reasons\n\n")
        fh.write("```json\n" + str(stats["exit_reasons"]) + "\n```\n\n")
        if sweep_lines:
            fh.write("\n## Parameter sweep\n\n")
            fh.write("| fear< | greed> | stop | return | buy-hold | PF | sharpe | maxDD | trades | WF | grade | score |\n")
            fh.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            fh.write("\n".join(sweep_lines) + "\n")
        fh.write(f"\n_Fees: {args.fee * 2:.2%} round trip. Buy-hold charged identical fees. "
                 f"F&G values lagged +1 day (no lookahead), signals shifted +1 bar._\n")
    print(f"\nResults saved to: {out_path}")
    print(f"[done in {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
