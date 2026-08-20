"""Immutable raw-candle preprocessing.

This module deliberately never fills a gap, never fabricates a candle, and
never mutates the raw data. It provides two representations:

* the legacy ``normalize_ohlcv`` relative-to-first-close representation used by
  the offline mock predictor; and
* the ``to_kronos_frame`` representation (raw OHLCV + UTC timestamps) that the
  upstream Kronos predictor consumes. Kronos performs its own z-score
  normalisation internally, so the real path must receive *raw* values, not the
  mock path's relative units.
"""
import math
from datetime import datetime, timezone
from typing import List

from .types import Candle

# Timeframe -> milliseconds. Unsupported timeframes are rejected explicitly.
TF = {'1h': 3_600_000, '4h': 14_400_000, '1d': 86_400_000}


def closed(candles: List[Candle], timeframe: str, now_ms: int) -> List[Candle]:
    """Return only candles whose close time is <= ``now_ms``.

    A candle is closed once ``timestamp_ms + TF[timeframe] <= now_ms``. The
    currently forming candle (open time <= now < close time) is excluded.
    """
    step = TF[timeframe]
    return [c for c in candles if c.timestamp_ms + step <= now_ms]


def validate_context(candles: List[Candle], timeframe: str, length: int) -> List[Candle]:
    """Validate the closed-candle context and return the last ``length`` rows.

    Enforced invariants (all must hold or a ``ValueError`` is raised):

    * supported timeframe
    * at least ``length`` closed candles
    * timestamps strictly increasing with exactly ``TF[timeframe]`` spacing
      (no gaps, no duplicates, no reordering)
    * every OHLCV value finite (no NaN / +/-inf)
    * OHLC relationships (high >= low/open/close, low <= open/close)
    * open/high/low/close > 0 and volume >= 0

    Gaps are reported, never filled.
    """
    if timeframe not in TF:
        raise ValueError('unsupported timeframe: %r' % timeframe)
    if length < 1:
        raise ValueError('context length must be >= 1')
    if len(candles) < length:
        raise ValueError(
            'insufficient closed candles: need %d, have %d' % (length, len(candles)))
    xs = list(candles[-length:])
    step = TF[timeframe]
    for a, b in zip(xs, xs[1:]):
        delta = b.timestamp_ms - a.timestamp_ms
        if delta != step:
            raise ValueError(
                'missing or misordered candle in context: gap of %d ms at %d (expected %d ms step)'
                % (delta, a.timestamp_ms, step))
    for c in xs:
        vals = (c.open, c.high, c.low, c.close, c.volume)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError('invalid numeric candle (NaN or infinity) at %d' % c.timestamp_ms)
        if min(c.open, c.high, c.low, c.close) <= 0:
            raise ValueError('non-positive OHLC value in candle at %d' % c.timestamp_ms)
        if c.volume < 0:
            raise ValueError('negative volume in candle at %d' % c.timestamp_ms)
        if not (c.high >= c.low and c.high >= c.open and c.high >= c.close
                and c.low <= c.open and c.low <= c.close):
            raise ValueError('invalid OHLC relationship in candle at %d' % c.timestamp_ms)
    return xs


def normalize_ohlcv(candles: List[Candle]) -> List[List[float]]:
    """Relative-to-first-close representation for the offline mock predictor.

    Kept only for the mock path; the real Kronos model consumes raw OHLCV via
    ``to_kronos_frame`` instead.
    """
    base = candles[0].close
    return [[c.open / base, c.high / base, c.low / base, c.close / base,
             math.log1p(c.volume)] for c in candles]


def _ms_to_datetime(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)


def to_kronos_frame(candles: List[Candle]):
    """Convert validated closed candles to the upstream Kronos input contract.

    Returns ``(df, x_timestamp)`` where ``df`` is a pandas DataFrame with the
    columns Kronos requires (``open, high, low, close, volume`` - ``amount`` is
    derived by Kronos as volume * mean price) and ``x_timestamp`` is a pandas
    Series of timezone-aware UTC datetimes, one per candle, in the same order.
    """
    import pandas as pd
    df = pd.DataFrame([{
        'open': c.open, 'high': c.high, 'low': c.low,
        'close': c.close, 'volume': c.volume,
    } for c in candles])
    x_timestamp = pd.Series([_ms_to_datetime(c.timestamp_ms) for c in candles])
    return df, x_timestamp


def future_timestamps(last_closed_ms: int, timeframe: str, horizon: int):
    """Datetime index of the ``horizon`` candles following the last closed one."""
    import pandas as pd
    step = TF[timeframe]
    return pd.Series([_ms_to_datetime(last_closed_ms + step * (i + 1))
                      for i in range(horizon)])
