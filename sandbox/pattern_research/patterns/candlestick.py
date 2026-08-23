"""Pattern 2 — Candlestick patterns (Phase 9C).

Four single/two-bar candlestick patterns on 1h candles. Every detector follows
the sandbox timing convention: the raw condition uses only bars ``<= t``, then
the series is ``.shift(1)``-ed so ``signal[t]`` means "pattern completed at
bar ``t-1``" and the trade is measured from ``close[t]``. No look-ahead.

Degenerate bars (``high == low``, i.e. zero range) are treated as "no pattern"
rather than dividing by zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .momentum import _finalize, _require_columns, compute_forward_return  # noqa: F401

DOJI_THRESHOLD = 0.1


def _parts(candles: pd.DataFrame):
    """Body / range / shadow components used by several detectors."""
    o = candles["open"].astype(float)
    h = candles["high"].astype(float)
    low = candles["low"].astype(float)
    c = candles["close"].astype(float)
    body = (c - o).abs()
    rng = h - low
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - low
    return o, h, low, c, body, rng, upper, lower


def detect_bullish_engulfing(candles: pd.DataFrame) -> pd.Series:
    """``+1`` where a bullish engulfing completes, ``0`` elsewhere.

    Rule: the current candle is bullish (``close > open``), the previous candle
    is bearish (``close < open``), and the current *body* completely engulfs the
    previous *body* (``open[t] <= close[t-1]`` and ``close[t] >= open[t-1]``,
    with at least one strict inequality so a repeat of the same body is not a
    pattern).
    """
    _require_columns(candles, ("open", "high", "low", "close"))
    if candles.empty:
        return pd.Series(dtype=int, index=candles.index, name="bullish_engulfing")

    o = candles["open"].astype(float)
    c = candles["close"].astype(float)
    prev_o, prev_c = o.shift(1), c.shift(1)

    cur_bull = c > o
    prev_bear = prev_c < prev_o
    engulfs = (o <= prev_c) & (c >= prev_o) & ((o < prev_c) | (c > prev_o))
    raw = (cur_bull & prev_bear & engulfs).fillna(False).astype(int)
    return _finalize(raw, "bullish_engulfing")


def detect_bearish_engulfing(candles: pd.DataFrame) -> pd.Series:
    """``-1`` where a bearish engulfing completes, ``0`` elsewhere.

    Mirror image of :func:`detect_bullish_engulfing`: current candle bearish,
    previous bullish, current body engulfs the previous body.
    """
    _require_columns(candles, ("open", "high", "low", "close"))
    if candles.empty:
        return pd.Series(dtype=int, index=candles.index, name="bearish_engulfing")

    o = candles["open"].astype(float)
    c = candles["close"].astype(float)
    prev_o, prev_c = o.shift(1), c.shift(1)

    cur_bear = c < o
    prev_bull = prev_c > prev_o
    engulfs = (o >= prev_c) & (c <= prev_o) & ((o > prev_c) | (c < prev_o))
    raw = -(cur_bear & prev_bull & engulfs).fillna(False).astype(int)
    return _finalize(raw, "bearish_engulfing")


def detect_doji(candles: pd.DataFrame, threshold: float = DOJI_THRESHOLD) -> pd.Series:
    """``1`` where a doji completes, ``0`` elsewhere.

    Rule: ``|close - open| / (high - low) < threshold`` (default 0.1).
    Zero-range bars are excluded (undefined ratio).

    Note: a doji is a *non-directional* pattern. It is emitted as ``+1`` so the
    same machinery can score it, which effectively tests "does a doji predict an
    up move?". That interpretation is stated in the results report — a doji
    failing the gates says nothing about a doji-as-reversal-context reading.
    """
    _require_columns(candles, ("open", "high", "low", "close"))
    if candles.empty:
        return pd.Series(dtype=int, index=candles.index, name="doji")
    if not (0.0 < float(threshold) <= 1.0):
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    o, h, low, c, body, rng, _u, _l = _parts(candles)
    ratio = pd.Series(np.where(rng > 0, body / rng.replace(0, np.nan), np.nan),
                      index=candles.index)
    raw = (ratio < float(threshold)).fillna(False).astype(int)
    return _finalize(raw, "doji")


def detect_hammer(candles: pd.DataFrame) -> pd.Series:
    """``+1`` where a hammer completes, ``0`` elsewhere.

    Rule (all three must hold):

    * lower shadow ``> 2 x`` body,
    * upper shadow ``< 0.5 x`` body,
    * the body sits in the **upper 30%** of the candle range, i.e.
      ``min(open, close) >= low + 0.7 * (high - low)``.

    Zero-range bars and zero-body bars are excluded (the shadow-to-body ratios
    are undefined/degenerate for a zero body).
    """
    _require_columns(candles, ("open", "high", "low", "close"))
    if candles.empty:
        return pd.Series(dtype=int, index=candles.index, name="hammer")

    o, h, low, c, body, rng, upper, lower = _parts(candles)
    body_low = pd.concat([o, c], axis=1).min(axis=1)
    valid = (rng > 0) & (body > 0)
    cond = (
        valid
        & (lower > 2.0 * body)
        & (upper < 0.5 * body)
        & (body_low >= low + 0.7 * rng)
    )
    raw = cond.fillna(False).astype(int)
    return _finalize(raw, "hammer")
