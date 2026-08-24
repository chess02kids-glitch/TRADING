"""Grading system for the Fear & Greed Contrarian strategy.

grade_strategy(stats, wf_stats) -> dict with grade (A-F), score (0-100),
pass_criteria, fail_criteria and a recommendation string.
"""
from __future__ import annotations

import math


def period_consistent(s: dict) -> bool:
    """A walk-forward period counts as consistent when the strategy was
    actually profitable in it (positive return AND profit factor > 1)."""
    if s.get("error") or s.get("num_trades", 0) == 0:
        return False
    return s["total_return_pct"] > 0 and s["profit_factor"] > 1.0


def _finite(x: float) -> float:
    return x if math.isfinite(x) else 99.0


def grade_strategy(stats: dict, wf_stats: list[dict]) -> dict:
    pf = _finite(stats.get("profit_factor", 0.0))
    sharpe = stats.get("annualized_sharpe", 0.0)
    beat = bool(stats.get("beat_buy_hold", False))
    mdd = stats.get("max_drawdown_pct", -100.0)
    total_ret = stats.get("total_return_pct", -100.0)
    n_trades = stats.get("num_trades", 0)

    ok = [period_consistent(s) for s in wf_stats]
    n_consistent = sum(ok)
    n_periods = max(len(ok), 1)

    passed: list[str] = []
    failed: list[str] = []

    def check(cond: bool, label_ok: str, label_bad: str) -> bool:
        (passed if cond else failed).append(label_ok if cond else label_bad)
        return cond

    # ---- hard fail conditions ------------------------------------------- #
    f_fail = pf < 1.0
    neg_ret = total_ret < 0
    deep_dd = mdd < -40.0

    check(not f_fail, f"profit factor {pf:.2f} >= 1.0", f"profit factor {pf:.2f} < 1.0 (loses money after fees)")
    check(not neg_ret, "total return positive", f"total return negative ({total_ret:.1f}%)")
    check(not deep_dd, f"max drawdown {mdd:.1f}% > -40%", f"max drawdown {mdd:.1f}% < -40% (blow-up risk)")
    check(beat, "beats buy-and-hold", "does NOT beat buy-and-hold")
    check(pf > 1.5, f"profit factor {pf:.2f} > 1.5", f"profit factor {pf:.2f} <= 1.5")
    check(sharpe > 1.0, f"sharpe {sharpe:.2f} > 1.0", f"sharpe {sharpe:.2f} <= 1.0")
    check(n_consistent == n_periods,
          f"consistent in all {n_periods} walk-forward periods",
          f"consistent in only {n_consistent}/{n_periods} walk-forward periods")
    check(mdd > -20.0, f"max drawdown {mdd:.1f}% > -20%", f"max drawdown {mdd:.1f}% <= -20%")
    check(pf > 1.2, f"profit factor {pf:.2f} > 1.2", f"profit factor {pf:.2f} <= 1.2")
    check(sharpe > 0.5, f"sharpe {sharpe:.2f} > 0.5", f"sharpe {sharpe:.2f} <= 0.5")
    check(n_consistent >= n_periods - 1,
          f"consistent in >= {n_periods - 1} of {n_periods} periods",
          "inconsistent across periods")
    check(pf > 1.05, f"profit factor {pf:.2f} > 1.05", f"profit factor {pf:.2f} <= 1.05")
    check(sharpe > 0.0, f"sharpe {sharpe:.2f} > 0", f"sharpe {sharpe:.2f} <= 0")

    # ---- grade (checked best-to-worst, F conditions can override) -------- #
    if f_fail or neg_ret or deep_dd:
        grade = "F"
    elif pf > 1.5 and sharpe > 1.0 and beat and n_consistent == n_periods and mdd > -20.0:
        grade = "A"
    elif pf > 1.2 and sharpe > 0.5 and beat and n_consistent >= n_periods - 1:
        grade = "B"
    elif pf > 1.05 and sharpe > 0.0 and beat:
        grade = "C"
    elif pf > 1.0:
        grade = "D"
    else:
        grade = "F"

    # ---- numeric score --------------------------------------------------- #
    score = 0.0
    score += min(25.0, pf * 10.0)                                   # profit factor
    score += max(-5.0, min(20.0, sharpe * 10.0))                    # sharpe
    score += 15.0 if beat else 0.0                                  # benchmark
    score += {True: 20.0}.get(n_consistent == n_periods,
                              13.0 if n_consistent == n_periods - 1 else
                              7.0 if n_consistent == 1 else 0.0)    # consistency
    score += 10.0 if mdd > -10 else 7.0 if mdd > -20 else 4.0 if mdd > -30 else 0.0
    score += 5.0 if n_trades >= 10 else 3.0 if n_trades >= 5 else 1.0 if n_trades > 0 else 0.0
    wr = stats.get("win_rate_pct", 0.0)
    score += 5.0 if wr >= 50 else 3.0 if wr >= 40 else 1.0
    score = max(0.0, min(100.0, round(score)))

    recommendation = {
        "A": "PAPER TRADE THIS",
        "B": "PAPER TRADE THIS",
        "C": "OPTIMIZE PARAMETERS",
        "D": "CLOSED - DO NOT USE",
        "F": "CLOSED - DO NOT USE",
    }[grade]

    return {
        "grade": grade,
        "score": score,
        "pass_criteria": passed,
        "fail_criteria": failed,
        "recommendation": recommendation,
        "n_consistent_periods": n_consistent,
        "n_periods": n_periods,
        "period_ok": ok,
    }
