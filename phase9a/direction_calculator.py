"""Past-only breakout direction and forward-return computation (Step 1 of 9A).

Two pure functions turn raw candle history + completed breakout rows from the
``har_predictions`` table into the two tables the rest of Phase 9A consumes:

* :func:`compute_breakout_direction` — for every breakout bar, the candle's
  direction (``close >= open`` → +1 UP, else -1 DOWN) computed *only* from the
  breakout bar itself.
* :func:`compute_forward_returns` — for every breakout bar, the realised return
  over the next ``N`` bars (``close_{t+N} / close_t - 1``) and its sign.

Leakage discipline (enforced here, by construction):

* Direction uses **only** the breakout bar's ``open`` and ``close`` — never a
  future bar. The breakout bar is located by exact timestamp match against the
  candle series, so its identity is verified from past data.
* Forward returns use bars **strictly after** the breakout bar
  (positional offset ``+N`` within the same asset's ascending candle series).
* When the ``t+N`` bar does not exist (breakout near the end of the series),
  that horizon is **skipped** — no row is emitted, no NaN forward-filled.
* If the breakout bar itself cannot be matched to a candle, the event is
  dropped with a warning (direction cannot be verified without the candle).

The breakout bar's timestamp in ``har_predictions`` is the ISO8601 UTC *open
time* of the predicted bar (see ``kronos_trading.alerts.prediction_logger``),
which is exactly the CCXT candle open time — so the two match directly.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Required candle columns (timestamp may be ISO8601 string, epoch-ms int/float,
# or a datetime — all normalized internally to UTC epoch-millis).
REQUIRED_CANDLE_COLS = ("timestamp", "open", "high", "low", "close", "volume")

# Required columns we read from the breakout rows (DB rows where
# breakout_flag == 1). ``regime`` is optional but carried through so the
# continuation tester can split hit rates by regime.
REQUIRED_BREAKOUT_COLS = ("timestamp", "asset", "har_predicted_range")


def _to_epoch_ms(value) -> Optional[int]:
    """Coerce one timestamp value to integer UTC epoch-millis.

    Accepts ISO8601 strings (``"2024-01-15T14:00:00Z"``), ``datetime`` /
    ``pandas.Timestamp`` objects, and numeric epoch values (millis when the
    magnitude looks like ms, otherwise treated as seconds). Returns ``None``
    for missing/unparseable values so one bad cell cannot crash a run.
    """
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    # Numeric epoch: ms if it looks like ms, else seconds.
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        v = int(value)
        return v if abs(v) >= 1_000_000_000_000 else v * 1000
    if isinstance(value, (float, np.floating)):
        v = float(value)
        v = v if abs(v) >= 1_000_000_000_000 else v * 1000.0
        return int(round(v))

    # Strings / datetime / Timestamp.
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        logger.warning("Unparseable timestamp %r — skipping", value)
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if pd.isna(ts):
        return None
    return int(ts.value // 1_000_000)


def _epoch_ms_series(series: pd.Series) -> pd.Series:
    """Vectorized wrapper around :func:`_to_epoch_ms` returning int64 epoch-ms."""
    return series.map(_to_epoch_ms).astype("Int64")


def _build_candle_index(
    candles: pd.DataFrame,
) -> Tuple[Dict[Optional[str], pd.DataFrame], Dict[Optional[str], Dict[int, int]]]:
    """Return ``(sorted_frames, position_lookups)`` keyed by asset.

    * ``sorted_frames[asset]`` — that asset's candles sorted ascending by
      epoch-ms with a fresh positional index.
    * ``position_lookups[asset]`` — ``{epoch_ms: positional_index}`` so the
      breakout bar and its ``t+N`` neighbours can be located in O(1).

    When the candle frame has no ``asset`` column it is treated as a single
    asset series keyed under ``None`` (the caller then matches every breakout
    row against it).
    """
    df = candles.copy()
    if "asset" in df.columns:
        df["_epoch_ms"] = _epoch_ms_series(df["timestamp"])
        df = df.dropna(subset=["_epoch_ms"])
        df["_epoch_ms"] = df["_epoch_ms"].astype("int64")
        frames: Dict[Optional[str], pd.DataFrame] = {}
        lookups: Dict[Optional[str], Dict[int, int]] = {}
        for asset, sub in df.groupby("asset", sort=True):
            sub = sub.sort_values("_epoch_ms").reset_index(drop=True)
            frames[str(asset)] = sub
            lookups[str(asset)] = dict(zip(sub["_epoch_ms"].tolist(), sub.index.tolist()))
        return frames, lookups

    df["_epoch_ms"] = _epoch_ms_series(df["timestamp"])
    df = df.dropna(subset=["_epoch_ms"])
    df["_epoch_ms"] = df["_epoch_ms"].astype("int64")
    df = df.sort_values("_epoch_ms").reset_index(drop=True)
    return {None: df}, {None: dict(zip(df["_epoch_ms"].tolist(), df.index.tolist()))}


def _validate_candles(candles: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in candles.columns]
    if missing:
        raise ValueError(f"candles missing required columns: {missing}")


def _validate_breakout_rows(breakout_rows: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_BREAKOUT_COLS if c not in breakout_rows.columns]
    if missing:
        raise ValueError(f"breakout_rows missing required columns: {missing}")


def compute_breakout_direction(
    candles: pd.DataFrame,
    breakout_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Breakout-bar candle direction for each completed breakout event.

    Args:
        candles: OHLCV frame with columns
            ``[timestamp, open, high, low, close, volume]`` (optionally
            ``asset``). Sorted ascending internally; the in-progress bar is
            harmless because breakout rows are completed (past) bars.
        breakout_rows: completed rows from ``har_predictions`` where
            ``breakout_flag == 1``. Must contain ``timestamp``, ``asset``,
            ``har_predicted_range``; ``regime`` / ``actual_range`` are carried
            through when present.

    Returns:
        Frame with columns ``timestamp, asset, breakout_direction,
        close_at_breakout, open_at_breakout, actual_range_at_breakout,
        har_predicted_at_breakout, regime``. ``breakout_direction`` is ``+1``
        when ``close >= open`` else ``-1``. Unmatched breakouts are dropped.

    ``regime`` is an extra column beyond the strict output contract: it is
    required downstream by ``compute_hit_rate(by_regime=...)`` and is simply
    forwarded from the breakout row (``None`` when absent).
    """
    if breakout_rows is None or len(breakout_rows) == 0:
        return pd.DataFrame(columns=[
            "timestamp", "asset", "breakout_direction", "close_at_breakout",
            "open_at_breakout", "actual_range_at_breakout",
            "har_predicted_at_breakout", "regime",
        ])
    _validate_candles(candles)
    _validate_breakout_rows(breakout_rows)

    frames, lookups = _build_candle_index(candles)
    multi_asset = None not in frames  # candle frame carries an asset column

    records: List[dict] = []
    n_dropped = 0
    for _, br in breakout_rows.iterrows():
        asset = str(br["asset"])
        ts_ms = _to_epoch_ms(br["timestamp"])
        key = asset if multi_asset else None
        lookup = lookups.get(key)
        if ts_ms is None or lookup is None or ts_ms not in lookup:
            n_dropped += 1
            continue
        row = frames[key].iloc[lookup[ts_ms]]
        o = float(row["open"])
        c = float(row["close"])
        actual_range = br["actual_range"] if "actual_range" in br.index else None
        if actual_range is None or (isinstance(actual_range, float) and pd.isna(actual_range)):
            actual_range = float(row["high"]) - float(row["low"])
        records.append({
            "timestamp": br["timestamp"],
            "asset": asset,
            "breakout_direction": 1 if c >= o else -1,
            "close_at_breakout": c,
            "open_at_breakout": o,
            "actual_range_at_breakout": float(actual_range),
            "har_predicted_at_breakout": float(br["har_predicted_range"]),
            "regime": br["regime"] if "regime" in br.index else None,
        })

    if n_dropped:
        logger.warning(
            "compute_breakout_direction: dropped %d/%d breakout events with no "
            "matching candle (cannot verify direction)", n_dropped,
            len(breakout_rows),
        )
    return pd.DataFrame(records, columns=[
        "timestamp", "asset", "breakout_direction", "close_at_breakout",
        "open_at_breakout", "actual_range_at_breakout",
        "har_predicted_at_breakout", "regime",
    ])


def compute_forward_returns(
    candles: pd.DataFrame,
    breakout_rows: pd.DataFrame,
    horizons: Iterable[int] = (1, 2, 3),
) -> pd.DataFrame:
    """Realised returns over the next ``N`` bars for each breakout event.

    For a breakout bar at positional index ``i`` within its asset's ascending
    candle series, the ``t+N`` return is ``close_{i+N} / close_i - 1``.
    ``forward_direction`` is ``+1`` for a positive return, ``-1`` for a
    negative return (a precisely-zero return yields ``0`` and naturally counts
    as a miss against any ``±1`` breakout direction).

    Args:
        candles: OHLCV frame (same contract as
            :func:`compute_breakout_direction`).
        breakout_rows: completed breakout rows (``breakout_flag == 1``).
        horizons: forward offsets in bars (default ``[1, 2, 3]``).

    Returns:
        Frame with columns ``timestamp, asset, horizon, forward_return,
        forward_direction``. One row per ``(breakout event, horizon)`` that
        actually has a ``t+N`` bar; horizons past the end of the series are
        skipped (no row emitted).
    """
    horizons = sorted({int(h) for h in horizons})
    if breakout_rows is None or len(breakout_rows) == 0 or not horizons:
        return pd.DataFrame(columns=[
            "timestamp", "asset", "horizon", "forward_return", "forward_direction",
        ])
    _validate_candles(candles)
    _validate_breakout_rows(breakout_rows)

    frames, lookups = _build_candle_index(candles)
    multi_asset = None not in frames

    records: List[dict] = []
    n_missing = 0
    for _, br in breakout_rows.iterrows():
        asset = str(br["asset"])
        ts_ms = _to_epoch_ms(br["timestamp"])
        key = asset if multi_asset else None
        lookup = lookups.get(key)
        if ts_ms is None or lookup is None or ts_ms not in lookup:
            n_missing += 1
            continue
        idx = lookup[ts_ms]
        frame = frames[key]
        base_close = float(frame.iloc[idx]["close"])
        n_bars = len(frame)
        for h in horizons:
            j = idx + h
            if j >= n_bars:
                continue  # t+N bar does not exist -> skip this horizon
            fut_close = float(frame.iloc[j]["close"])
            fr = fut_close / base_close - 1.0
            fd = 1 if fr > 0.0 else (-1 if fr < 0.0 else 0)
            records.append({
                "timestamp": br["timestamp"],
                "asset": asset,
                "horizon": h,
                "forward_return": fr,
                "forward_direction": fd,
            })

    if n_missing:
        logger.warning(
            "compute_forward_returns: skipped %d/%d breakout events with no "
            "matching candle", n_missing, len(breakout_rows),
        )
    return pd.DataFrame(records, columns=[
        "timestamp", "asset", "horizon", "forward_return", "forward_direction",
    ])
