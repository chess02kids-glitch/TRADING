"""
Shared research data context (loaded once, reused by every genome).

Builds:
  Window A (2017-08-17 .. 2019-11-04, Binance spot 1h):
    btc_A, eth_A OHLCV frames on a common hourly grid
    spread_A = log(BTC) - log(ETH)
    HAR predicted daily range + tercile regime labels (TWO variants:
    har_window=5 uses RV-daily + RV-weekly components, har_window=22
    adds the RV-monthly component) - rolling OLS, strictly no lookahead
  Window B (2019-12-01 .. 2023-12-31, Bitstamp BTC/USD 1h price +
  Binance USDT-M BTCUSDT funding rates):
    btc_B OHLCV frame
    funding_ev: 8h funding events (decimal)
    funding_h:  funding merged onto hourly bars and SHIFTED BY ONE BAR
                (no lookahead: a bar can only use funding settled at or
                before the previous bar's open)

The HAR model is the one validated signal in the parent project
(p ~ 2.15e-26 for volatility MAGNITUDE). It is recomputed here from
raw OHLCV with the same standard HAR-RV methodology because Supabase
is not reachable from this environment (documented in the report).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.dirname(HERE), "data", "cache")


def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(CACHE, name))


def _har_variants(btc: pd.DataFrame) -> Dict[str, pd.Series]:
    """Rolling HAR-RV one-day-ahead forecasts -> predicted daily range.

    Strictly out-of-sample: the regression for day t is fitted only on
    days < t (trailing 90 obs), and only past daily RVs enter the design.
    """
    close = btc["close"]
    logret = np.log(close).diff()
    r = logret.groupby(close.index.date).apply(lambda s: (s ** 2).sum())
    r.index = pd.to_datetime(r.index)
    r.name = "rv"
    rv = pd.DataFrame(r)

    rv["rv_d"] = rv["rv"]
    rv["rv_w"] = rv["rv"].rolling(5).mean().shift(1)
    rv["rv_m"] = rv["rv"].rolling(22).mean().shift(1)
    rv["rv_d_lag"] = rv["rv"].shift(1)

    idx = btc.index
    out = {}
    for variant, cols in (("5", ["rv_d_lag", "rv_w"]),
                          ("22", ["rv_d_lag", "rv_w", "rv_m"])):
        preds = {}
        series = rv.dropna(subset=cols + ["rv"])
        vals = series[cols].values
        targets = series["rv"].values
        for i in range(90, len(series)):
            X = np.column_stack([np.ones(90), vals[i - 90:i]])
            y = targets[i - 90:i]
            try:
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                pred = max(float(np.dot(np.r_[1.0, vals[i]], beta)), 1e-12)
            except Exception:
                pred = float(np.mean(y))
            preds[series.index[i]] = np.sqrt(pred)  # daily range forecast
        s = pd.Series(preds)
        out[variant] = s
    return out


def _map_daily_to_hourly(daily: pd.Series, hourly_index: pd.DatetimeIndex) -> pd.Series:
    s = daily.copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize("UTC")
    mapped = s.reindex(hourly_index.normalize()).ffill()
    mapped.index = hourly_index  # guarantee identical labels to the hourly frame
    return mapped


def build_context() -> dict:
    btc_a = _load("btc_1h_A.parquet")
    eth_a = _load("eth_1h_A.parquet")
    btc_b = _load("btc_1h_B.parquet")
    fund_ev = _load("funding_BTC.parquet")["funding_rate"]

    # Window B funding on hourly grid, shifted one bar (no lookahead):
    # reindex(method="ffill") = asof join of each hourly bar to the most
    # recent funding settlement at or before that bar's open time.
    funding_h = fund_ev.reindex(btc_b.index, method="ffill").shift(1)

    # spread (log price ratio)
    spread = np.log(btc_a["close"]) - np.log(eth_a["close"])

    har = _har_variants(btc_a)
    har_range = {}
    har_regime = {}
    for variant, s in har.items():
        rng_h = _map_daily_to_hourly(s, btc_a.index)
        har_range[variant] = rng_h
        # tercile regime computed on trailing 90 days of predicted range
        q33 = rng_h.rolling(90 * 24, min_periods=30 * 24).quantile(1 / 3)
        q67 = rng_h.rolling(90 * 24, min_periods=30 * 24).quantile(2 / 3)
        regime = pd.Series("medium", index=rng_h.index)
        regime[rng_h < q33] = "low"
        regime[rng_h > q67] = "high"
        har_regime[variant] = regime

    ctx = {
        "btc_A": btc_a, "eth_A": eth_a, "spread_A": spread,
        "har_range_A": har_range, "har_regime_A": har_regime,
        "btc_B": btc_b, "funding_ev_B": fund_ev, "funding_h_B": funding_h,
        "windows": {"A": (btc_a.index[0], btc_a.index[-1]),
                    "B": (btc_b.index[0], btc_b.index[-1])},
    }
    return ctx


class FundingMA:
    """Memoised funding moving averages merged onto Window B hourly bars
    (event-level rolling mean, then one-bar shift)."""

    def __init__(self, ctx: dict):
        self.ctx = ctx
        self._cache: Dict[int, pd.Series] = {}

    def __call__(self, window_events: int) -> pd.Series:
        if window_events not in self._cache:
            ev = self.ctx["funding_ev_B"]
            ma = ev.rolling(window_events, min_periods=1).mean()
            hourly = ma.reindex(self.ctx["btc_B"].index, method="ffill").shift(1)
            self._cache[window_events] = hourly
        return self._cache[window_events]


def get_funding_ma(ctx: dict) -> FundingMA:
    if "funding_ma" not in ctx:
        ctx["funding_ma"] = FundingMA(ctx)
    return ctx["funding_ma"]
