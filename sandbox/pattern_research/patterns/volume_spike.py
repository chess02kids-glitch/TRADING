"""Pattern 4 — Volume spike (Phase 9E).

A spike bar is one whose volume is ``threshold`` times its own recent average;
its direction is taken from the candle's body (bullish → ``+1``, bearish →
``-1``).

No look-ahead: the rolling mean ends at the *current* bar (all closed data),
and the emitted signal is ``.shift(1)``-ed like every other detector here, so
``signal[t]`` reflects a spike that completed at bar ``t-1``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .momentum import _finalize, _require_columns, compute_forward_return  # noqa: F401

DEFAULT_WINDOW = 20
DEFAULT_THRESHOLD = 2.0


def compute_volume_ratio(candles: pd.DataFrame, window: int = DEFAULT_WINDOW) -> pd.Series:
    """``volume[t] / mean(volume[t-window+1 : t])`` — the relative volume.

    The rolling mean includes the current bar (all values are known at bar
    close, so this is not look-ahead). Bars before the window is full, and bars
    whose rolling mean is ``0``, yield ``NaN``.
    """
    _require_columns(candles, ("volume",))
    if int(window) < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if candles.empty:
        return pd.Series(dtype=float, index=candles.index, name="volume_ratio")

    vol = candles["volume"].astype(float)
    mean = vol.rolling(int(window), min_periods=int(window)).mean()
    ratio = vol / mean.replace(0.0, np.nan)
    return ratio.rename("volume_ratio")


def detect_volume_spike(candles: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD,
                        window: int = DEFAULT_WINDOW) -> pd.Series:
    """``+1`` bullish spike bar, ``-1`` bearish spike bar, ``0`` elsewhere.

    Rule: ``volume_ratio > threshold`` (default 2.0) **and** the candle has a
    non-zero body; the sign is the body's sign. ``.shift(1)``-ed on return.
    """
    _require_columns(candles, ("open", "close", "volume"))
    if candles.empty:
        return pd.Series(dtype=int, index=candles.index, name="volume_spike")
    if float(threshold) <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")

    ratio = compute_volume_ratio(candles, window=window)
    spike = (ratio > float(threshold)).fillna(False)
    body_sign = np.sign(candles["close"].astype(float) - candles["open"].astype(float))
    raw = (spike.astype(int) * body_sign).fillna(0).astype(int)
    return _finalize(raw, "volume_spike")
