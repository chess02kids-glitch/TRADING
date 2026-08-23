"""Pattern 1 — Momentum structure (Phase 9B).

Two classical swing-structure patterns on 1h candles:

* **HH/HL** (higher highs + higher lows) → bullish structure, signal ``+1``
* **LL/LH** (lower lows + lower highs)  → bearish structure, signal ``-1``

No look-ahead, guaranteed two ways
----------------------------------
1. The raw condition at bar ``t`` only compares bars ``t, t-1, ... t-lookback``
   — all closed, all in the past.
2. The returned series is then ``.shift(1)``-ed, so ``signal[t]`` means "the
   pattern *completed at bar t-1*". The signal is therefore known **before**
   bar ``t`` even opens, and :func:`compute_forward_return` measures the trade
   from ``close[t]`` onwards. Nothing about bar ``t`` or later is used to
   produce ``signal[t]``.

Signal timing convention (used by every pattern module in this sandbox):

    pattern completes at bar t-1  →  signal[t] = ±1  →  entry at close[t]
    →  exit at close[t + horizon]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LOOKBACK = 3


def _require_columns(candles: pd.DataFrame, cols) -> None:
    missing = [c for c in cols if c not in candles.columns]
    if missing:
        raise ValueError(f"candles missing required columns: {missing}")


def _finalize(raw: pd.Series, name: str) -> pd.Series:
    """Shift by 1 bar (no look-ahead), fill NaNs with 0, return int series."""
    return raw.shift(1).fillna(0).astype(int).rename(name)


def detect_higher_high_higher_low(
    candles: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK
) -> pd.Series:
    """``+1`` where a bullish HH/HL structure is in place, ``0`` elsewhere.

    Rule (``lookback=3``): each of the last 3 highs is higher than the one
    before it **and** each of the last 3 lows is higher than the one before it,
    i.e. ``high[t] > high[t-1] > high[t-2] > high[t-3]`` and the same for lows.

    Returns a ``.shift(1)``-ed ``pd.Series`` of ints indexed like ``candles``.
    """
    _require_columns(candles, ("high", "low"))
    if int(lookback) < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if candles.empty:
        return pd.Series(dtype=int, index=candles.index, name="momentum_hh_hl")

    high, low = candles["high"], candles["low"]
    cond = pd.Series(True, index=candles.index)
    for k in range(int(lookback)):
        cond &= high.shift(k) > high.shift(k + 1)
        cond &= low.shift(k) > low.shift(k + 1)
    raw = cond.fillna(False).astype(int)  # +1 bullish / 0 none
    return _finalize(raw, "momentum_hh_hl")


def detect_lower_low_lower_high(
    candles: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK
) -> pd.Series:
    """``-1`` where a bearish LL/LH structure is in place, ``0`` elsewhere.

    Rule (``lookback=3``): each of the last 3 lows is lower than the one before
    it **and** each of the last 3 highs is lower than the one before it.
    ``.shift(1)``-ed like every other detector here.
    """
    _require_columns(candles, ("high", "low"))
    if int(lookback) < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if candles.empty:
        return pd.Series(dtype=int, index=candles.index, name="momentum_ll_lh")

    high, low = candles["high"], candles["low"]
    cond = pd.Series(True, index=candles.index)
    for k in range(int(lookback)):
        cond &= low.shift(k) < low.shift(k + 1)
        cond &= high.shift(k) < high.shift(k + 1)
    raw = -cond.fillna(False).astype(int)  # -1 bearish / 0 none
    return _finalize(raw, "momentum_ll_lh")


def detect_momentum_combined(
    candles: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK
) -> pd.Series:
    """Convenience: ``+1`` HH/HL, ``-1`` LL/LH, ``0`` otherwise (mutually
    exclusive by construction — a bar cannot be both)."""
    return (detect_higher_high_higher_low(candles, lookback)
            + detect_lower_low_lower_high(candles, lookback)).rename("momentum")


def compute_forward_return(
    candles: pd.DataFrame,
    signal_series: pd.Series,
    horizon: int = 1,
    include_flat: bool = False,
) -> pd.DataFrame:
    """Join signals to their realised forward return and score them.

    ``forward_return[t] = close[t + horizon] / close[t] - 1`` — the trade is
    entered at the close of the *signal* bar ``t`` (the pattern itself
    completed at ``t-1``) and exited ``horizon`` bars later. Bars whose exit
    falls outside the sample are dropped.

    Args:
        candles: canonical OHLCV frame.
        signal_series: ``±1 / 0`` series aligned to ``candles`` (already
            shifted by the detector).
        horizon: 1, 2 or 3 bars.
        include_flat: keep ``signal == 0`` rows (default ``False``: the frame
            contains one row per pattern occurrence, i.e. per "event").

    Returns:
        DataFrame indexed by timestamp with columns
        ``signal, forward_return, correct`` (``correct`` is ``1`` when
        ``sign(forward_return) == sign(signal)``, else ``0``).
    """
    _require_columns(candles, ("close",))
    if int(horizon) < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    signal = pd.Series(signal_series).reindex(candles.index).fillna(0)
    close = candles["close"].astype(float)
    fwd = close.shift(-int(horizon)) / close - 1.0

    out = pd.DataFrame({"signal": signal.astype(int), "forward_return": fwd})
    out = out.dropna(subset=["forward_return"])
    if not include_flat:
        out = out[out["signal"] != 0]
    out["correct"] = (
        np.sign(out["forward_return"]) == np.sign(out["signal"])
    ).astype(int)
    # A flat forward return (exactly 0.0) can never match a ±1 signal — the
    # comparison above already scores it 0, which is the honest treatment.
    out.index.name = candles.index.name or "timestamp"
    return out
