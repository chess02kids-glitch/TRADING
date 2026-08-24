"""Bar-by-bar backtester with stop-loss state tracking, plus walk-forward split.

The simulation is deliberately NOT vectorised: stop-loss / max-hold exits need
per-trade state (entry price, entry time, block-until-reset after a stop).

Fees: charged on BOTH legs (entry and exit), fee_one_way each side.
Buy-hold benchmark is charged the same round-trip fees for an honest comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .grader import period_consistent  # noqa: F401  (re-exported for convenience)


def backtest(
    df: pd.DataFrame,
    fee_one_way: float = 0.001,
    stop_loss_pct: float = 0.05,
    max_hold_days: float = 14.0,
) -> dict:
    """Run the simulation on a frame produced by signal_generator.generate_signals."""
    close = df["close"].to_numpy(dtype=float)
    pos = df["position"].to_numpy(dtype=int)
    size_arr = df["size"].to_numpy(dtype=float)
    force_exit = df["force_exit"].to_numpy(dtype=bool)
    ts = df.index

    n = len(df)
    equity = 1.0
    cash = 1.0
    units = 0.0
    entry_price = 0.0
    entry_time = None
    entry_commit = 0.0
    in_pos = False
    blocked = False  # set after a stop-loss exit; cleared when the raw signal resets

    equity_curve = np.empty(n)
    trades: list[dict] = []

    for i in range(n):
        price = close[i]

        if in_pos:
            exit_reason = None
            if force_exit[i]:
                exit_reason = "greed"
            elif price <= entry_price * (1.0 - stop_loss_pct):
                exit_reason = "stop_loss"
            elif (ts[i] - entry_time).total_seconds() / 86400.0 > max_hold_days:
                exit_reason = "max_hold"

            if exit_reason is not None:
                proceeds = units * price * (1.0 - fee_one_way)
                cash += proceeds
                ret = proceeds / entry_commit - 1.0
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": ts[i],
                        "entry_price": entry_price,
                        "exit_price": price,
                        "return_pct": ret * 100.0,
                        "hold_days": (ts[i] - entry_time).total_seconds() / 86400.0,
                        "exit_reason": exit_reason,
                        "size": entry_commit,
                    }
                )
                equity = cash
                units = 0.0
                in_pos = False
                if exit_reason == "stop_loss":
                    blocked = True  # no same-episode re-entry after a stop-out
        else:
            if blocked and pos[i] != 1:
                blocked = False  # signal reset - allow fresh entries
            if pos[i] == 1 and not blocked:
                commit = equity * size_arr[i] if size_arr[i] > 0 else 0.0
                if commit > 0:
                    entry_commit = commit
                    entry_price = price
                    entry_time = ts[i]
                    units = commit * (1.0 - fee_one_way) / price
                    cash -= commit
                    in_pos = True

        equity_curve[i] = cash + units * price if in_pos else cash
        equity = equity_curve[i]

    # close any position still open at the end of data
    if in_pos:
        price = close[-1]
        proceeds = units * price * (1.0 - fee_one_way)
        cash += proceeds
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": ts[-1],
                "entry_price": entry_price,
                "exit_price": price,
                "return_pct": (proceeds / entry_commit - 1.0) * 100.0,
                "hold_days": (ts[-1] - entry_time).total_seconds() / 86400.0,
                "exit_reason": "end_of_data",
                "size": entry_commit,
            }
        )
        equity = cash
        equity_curve[-1] = cash

    # ---- metrics --------------------------------------------------------- #
    total_return_pct = (equity - 1.0) * 100.0
    bh = (close[-1] / close[0]) * (1.0 - fee_one_way) ** 2 - 1.0  # same fees, honest
    buy_hold_return_pct = bh * 100.0

    rets = np.diff(equity_curve) / equity_curve[:-1]
    if len(rets) and rets.std(ddof=0) > 0:
        sharpe = float(rets.mean() / rets.std(ddof=0) * np.sqrt(24 * 365))
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(equity_curve)
    dd = equity_curve / peak - 1.0
    max_drawdown_pct = float(dd.min() * 100.0)

    wins = [t for t in trades if t["return_pct"] > 0]
    losses = [t for t in trades if t["return_pct"] <= 0]
    gross_win = sum(t["return_pct"] * t["size"] for t in wins)
    gross_loss = abs(sum(t["return_pct"] * t["size"] for t in losses))
    profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_win > 0 else 0.0
    )

    hold_series = pd.Series([t["hold_days"] for t in trades])
    reason_counts = (
        pd.Series([t["exit_reason"] for t in trades]).value_counts().to_dict()
        if trades else {}
    )
    exposure = float(np.mean(pos == 1) * 100.0)

    return {
        "total_return_pct": total_return_pct,
        "buy_hold_return_pct": buy_hold_return_pct,
        "beat_buy_hold": bool(total_return_pct > buy_hold_return_pct),
        "annualized_sharpe": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "num_trades": len(trades),
        "win_rate_pct": (len(wins) / len(trades) * 100.0) if trades else 0.0,
        "avg_hold_days": float(hold_series.mean()) if len(hold_series) else 0.0,
        "best_trade_pct": max((t["return_pct"] for t in trades), default=0.0),
        "worst_trade_pct": min((t["return_pct"] for t in trades), default=0.0),
        "profit_factor": profit_factor,
        "exposure_pct": exposure,
        "exit_reasons": reason_counts,
        "start": str(ts[0].date()),
        "end": str(ts[-1].date()),
        "trades": trades,
        "equity_curve": equity_curve,
    }


def walk_forward(
    df: pd.DataFrame,
    n_periods: int = 3,
    fee_one_way: float = 0.001,
    stop_loss_pct: float = 0.05,
    max_hold_days: float = 14.0,
) -> list[dict]:
    """Split into n contiguous periods and backtest each independently."""
    stats_list = []
    n = len(df)
    bounds = [round(i * n / n_periods) for i in range(n_periods + 1)]
    for k in range(n_periods):
        chunk = df.iloc[bounds[k]:bounds[k + 1]]
        if len(chunk) < 24 * 30:  # need at least ~a month
            stats_list.append({"error": "period too short", "total_return_pct": 0.0,
                               "profit_factor": 0.0, "num_trades": 0,
                               "start": str(chunk.index[0].date()),
                               "end": str(chunk.index[-1].date())})
            continue
        s = backtest(chunk, fee_one_way=fee_one_way,
                     stop_loss_pct=stop_loss_pct, max_hold_days=max_hold_days)
        s["period"] = k + 1
        s.pop("trades", None)
        s.pop("equity_curve", None)
        stats_list.append(s)
    return stats_list
