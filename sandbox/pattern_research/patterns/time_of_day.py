"""Pattern 3 — Time-of-day / day-of-week bias (Phase 9D).

Descriptive tables first (:func:`compute_hourly_bias`,
:func:`compute_daily_bias`), then a tradable signal builder
(:func:`build_hour_signal`) that fires *ahead* of the selected hours.

Bar return convention
---------------------
``ret[t] = close[t] / close[t-1] - 1`` is attributed to the **hour (UTC) of bar
t** — the hour during which that return was earned.

No look-ahead
-------------
The bias tables summarise already-realised bars (they are statistics, not
forecasts); the runner is what must avoid look-ahead, and it does so by
selecting the "best hours" on a *training slice only* and evaluating them on
later, unseen bars.

The signal itself contains **zero market data**: it is pure calendar
arithmetic. To bet on hour ``H`` under the sandbox convention
(``signal[t]`` → trade from ``close[t]`` to ``close[t+horizon]``), the signal
must fire on the bar *before* ``H``:

    signal[t] = +1  iff  (hour(t) + 1) % 24 in best_hours

Nothing from bar ``t`` or later is used, which is a strictly stronger
guarantee than the ``.shift(1)`` rule applied to the price-based patterns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .momentum import _require_columns, compute_forward_return  # noqa: F401

MIN_OBS_PER_HOUR = 100
DEFAULT_MIN_WIN_RATE = 0.55
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _bar_returns(candles: pd.DataFrame) -> pd.Series:
    _require_columns(candles, ("close",))
    if not isinstance(candles.index, pd.DatetimeIndex):
        raise TypeError("candles must be indexed by a DatetimeIndex (UTC)")
    return candles["close"].astype(float).pct_change().dropna()


def _bias_table(returns: pd.Series, key: pd.Series, key_name: str,
                full_index) -> pd.DataFrame:
    grouped = returns.groupby(key)
    table = pd.DataFrame({
        key_name: pd.Series(full_index, index=full_index),
        "mean_return": grouped.mean(),
        "win_rate": grouped.apply(lambda s: float((s > 0).mean()) if len(s) else np.nan),
        "n_observations": grouped.size(),
    }).reindex(full_index)
    table[key_name] = full_index
    table["n_observations"] = table["n_observations"].fillna(0).astype(int)
    table.index.name = key_name
    return table


def compute_hourly_bias(candles: pd.DataFrame) -> pd.DataFrame:
    """Per-UTC-hour return statistics, sorted by ``win_rate`` descending.

    Returns a DataFrame indexed by hour ``0-23`` with columns
    ``hour, mean_return, win_rate, n_observations``. Hours with no data are
    present with ``n_observations = 0`` and ``NaN`` statistics.
    """
    returns = _bar_returns(candles)
    hours = pd.Index(range(24), name="hour")
    if returns.empty:
        empty = pd.DataFrame({"hour": hours, "mean_return": np.nan,
                              "win_rate": np.nan, "n_observations": 0}, index=hours)
        return empty
    table = _bias_table(returns, returns.index.hour, "hour", hours)
    return table.sort_values("win_rate", ascending=False, na_position="last")


def compute_daily_bias(candles: pd.DataFrame) -> pd.DataFrame:
    """Per-weekday return statistics, indexed ``0-6`` (Mon..Sun, chronological).

    Columns: ``day, day_name, mean_return, win_rate, n_observations``.
    """
    returns = _bar_returns(candles)
    days = pd.Index(range(7), name="day")
    if returns.empty:
        out = pd.DataFrame({"day": days, "mean_return": np.nan,
                            "win_rate": np.nan, "n_observations": 0}, index=days)
        out.insert(1, "day_name", DAY_NAMES)
        return out
    table = _bias_table(returns, returns.index.dayofweek, "day", days)
    table.insert(1, "day_name", DAY_NAMES)
    return table


def find_best_hours(hourly_df: pd.DataFrame,
                    min_win_rate: float = DEFAULT_MIN_WIN_RATE) -> list:
    """Hours whose ``win_rate > min_win_rate`` **and** ``n_observations > 100``.

    Returned in descending win-rate order (ties broken by hour ascending).
    """
    if hourly_df is None or hourly_df.empty:
        return []
    df = hourly_df.copy()
    if "hour" not in df.columns:
        df["hour"] = df.index
    # index is also named 'hour' -> drop it so sorting is unambiguous
    df = df.reset_index(drop=True)
    sel = df[(df["win_rate"] > float(min_win_rate))
             & (df["n_observations"] > MIN_OBS_PER_HOUR)]
    sel = sel.sort_values(["win_rate", "hour"], ascending=[False, True])
    return [int(h) for h in sel["hour"].tolist()]


def build_hour_signal(candles: pd.DataFrame, hours) -> pd.Series:
    """``+1`` on the bar *preceding* each hour in ``hours``, ``0`` elsewhere.

    Pure calendar signal — see the module docstring. With the sandbox
    convention this means the trade is held *through* the selected hour.
    """
    if not isinstance(candles.index, pd.DatetimeIndex):
        raise TypeError("candles must be indexed by a DatetimeIndex (UTC)")
    target = {int(h) % 24 for h in (hours or [])}
    if candles.empty or not target:
        return pd.Series(0, index=candles.index, dtype=int, name="hour_bias")
    next_hour = (candles.index.hour + 1) % 24
    raw = np.isin(next_hour, list(target)).astype(int)
    return pd.Series(raw, index=candles.index, dtype=int, name="hour_bias")
