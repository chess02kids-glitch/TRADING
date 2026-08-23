"""OHLCV loading for the pattern-research sandbox.

Data source: **CCXT KuCoin public API** — no API keys, no secrets, no DB.
The fetch logic mirrors ``kronos_trading/alerts/har_forecaster.py``:

* public ``ccxt.kucoin`` client with ``enableRateLimit`` and spot markets,
* rows sanitised (malformed / unparseable rows dropped),
* the still-forming candle is dropped (``ts + bar_ms > now_ms``) so no
  in-progress bar can leak into a "past" feature,
* duplicate timestamps deduplicated (last occurrence wins),
* candles returned sorted ascending by timestamp.

**Timeframes:** :func:`fetch_ohlcv` supports ``1h``, ``4h`` and ``1d`` bars
(:data:`SUPPORTED_TIMEFRAMES`). The bar length of the chosen timeframe drives
BOTH the pagination step and the "is this bar closed yet?" rule
(``open_time + bar_len <= now``), exactly like ``har_forecaster``. Each
timeframe gets its own cache file (``BTCUSDT_1h_730d.csv`` vs
``BTCUSDT_4h_730d.csv`` vs ``BTCUSDT_1d_730d.csv``), and
:func:`infer_timeframe` can recover the timeframe from a DataFrame's bar
spacing — used to warn when a ``--csv`` file contradicts ``--timeframe``.

The only addition here versus har_forecaster is **pagination**: har_forecaster
needs 800 bars (one request), the sandbox needs 730 days x 24 = 17,520 hourly
bars, so the loader walks forward with ``since`` until it reaches now.

Everything is returned as a ``pandas.DataFrame`` (the "candles" object used by
every pattern module) with:

    index    : ``timestamp`` — tz-aware UTC DatetimeIndex, ascending, unique
    columns  : ``open, high, low, close, volume`` (float64)

A CSV disk cache is used so repeated research runs do not re-hit the exchange,
and ``load_candles(csv=...)`` allows fully offline runs from a saved file.
"""
from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# --- Constants (fixed by the sandbox spec) ----------------------------------
DEFAULT_TIMEFRAME = "1h"
DEFAULT_DAYS = 730
DEFAULT_ASSETS = ("BTC/USDT", "ETH/USDT")
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Timeframes this sandbox supports for research runs. The bar length drives
# BOTH the pagination step and the "is this bar closed yet?" rule.
SUPPORTED_TIMEFRAMES = ("1h", "4h", "1d")

# Timeframe -> bar length in ms (mirrors har_forecaster.TIMEFRAME_MS).
TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

KUCOIN_MAX_LIMIT = 1500  # KuCoin klines hard cap per request
MIN_CANDLES = 50         # same floor as har_forecaster.MIN_CANDLES

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


class InsufficientCandlesError(ValueError):
    """Fewer closed candles than the sandbox requires (``MIN_CANDLES``)."""


def _normalize_symbol(asset: str) -> str:
    """Accept ``'BTC/USDT'`` or ``'BTC'`` and return a CCXT spot symbol."""
    symbol = str(asset).strip().upper()
    if "/" not in symbol:
        symbol = f"{symbol}/USDT"
    return symbol


def _default_exchange():
    """CCXT KuCoin public client — no API keys (same as har_forecaster)."""
    import ccxt  # lazy import so the module imports without ccxt installed

    return ccxt.kucoin({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })


def _rows_to_frame(rows: List[list]) -> pd.DataFrame:
    """Sanitise raw CCXT OHLCV rows into the canonical candles DataFrame."""
    records = []
    for row in rows:
        if row is None or len(row) < 6:
            logger.warning("Skipping malformed OHLCV row: %r", row)
            continue
        try:
            records.append((
                int(row[0]), float(row[1]), float(row[2]),
                float(row[3]), float(row[4]), float(row[5]),
            ))
        except (TypeError, ValueError):
            logger.warning("Skipping unparseable OHLCV row: %r", row)
            continue

    if not records:
        return empty_candles()

    df = pd.DataFrame(records, columns=["ts"] + OHLCV_COLUMNS)
    # Duplicate timestamps: last occurrence wins (same rule as har_forecaster).
    df = df.drop_duplicates(subset="ts", keep="last").sort_values("ts")
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns="ts").set_index("timestamp")
    df = df[OHLCV_COLUMNS].astype(float)
    return _normalize_index(df)


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Force a nanosecond-resolution UTC DatetimeIndex so frames built from an
    API fetch and from a CSV round-trip compare equal."""
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    try:
        idx = idx.as_unit("ns")
    except (AttributeError, ValueError):  # pragma: no cover - older pandas
        pass
    df = df.copy()
    df.index = idx.rename("timestamp")
    return df


def empty_candles() -> pd.DataFrame:
    """An empty candles frame with the canonical schema (for edge cases)."""
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return pd.DataFrame({c: pd.Series(dtype=float) for c in OHLCV_COLUMNS}, index=idx)


def fetch_ohlcv(
    asset: str,
    timeframe: str = DEFAULT_TIMEFRAME,
    days: int = DEFAULT_DAYS,
    exchange=None,
    now_ms: Optional[int] = None,
    limit: int = KUCOIN_MAX_LIMIT,
    max_requests: int = 500,
) -> pd.DataFrame:
    """Fetch the last ``days`` days of *closed* candles from KuCoin (public).

    Args:
        asset: ``'BTC/USDT'`` or ``'BTC'`` (``/USDT`` appended when missing).
        timeframe: e.g. ``'1h'`` (default).
        days: lookback window in days (default 730).
        exchange: injectable CCXT-compatible object exposing
            ``fetch_ohlcv(symbol, timeframe, since=..., limit=...)`` (test hook).
            When ``None`` a public KuCoin client is created — no API keys.
        now_ms: current time in ms (test hook); defaults to the real clock.
        limit: rows per request (KuCoin caps at 1500).
        max_requests: safety stop for the pagination loop.

    Returns:
        Canonical candles DataFrame, ascending, deduplicated, in-progress bar
        removed.

    Raises:
        InsufficientCandlesError: fewer than :data:`MIN_CANDLES` closed candles.
    """
    symbol = _normalize_symbol(asset)
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe {timeframe!r}; "
                         f"allowed: {list(SUPPORTED_TIMEFRAMES)}")
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}")
    if exchange is None:
        exchange = _default_exchange()
    if now_ms is None:
        now_ms = int(time.time() * 1000)

    bar_ms = TIMEFRAME_MS[timeframe]
    since = int(now_ms - days * 86_400_000)
    collected: List[list] = []
    requests_made = 0

    while since < now_ms and requests_made < max_requests:
        raw = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        requests_made += 1
        if not raw:
            break
        collected.extend(raw)
        last_ts = max(int(r[0]) for r in raw if r and len(r) >= 1)
        next_since = last_ts + bar_ms
        if next_since <= since:  # exchange not advancing — stop rather than spin
            break
        since = next_since
        if len(raw) < limit:
            break

    df = _rows_to_frame(collected)

    # Drop the in-progress bar: a candle is closed only when
    # open_time + bar_length <= now (identical rule to har_forecaster).
    if not df.empty:
        open_ms = df.index.astype("int64") // 1_000_000
        closed = open_ms + bar_ms <= now_ms
        df = df[closed]
        cutoff = pd.to_datetime(now_ms - days * 86_400_000, unit="ms", utc=True)
        df = df[df.index >= cutoff]

    if len(df) < MIN_CANDLES:
        raise InsufficientCandlesError(
            f"Only {len(df)} closed {timeframe} candles for {symbol} "
            f"(need >= {MIN_CANDLES}); exchange returned {len(collected)} raw rows"
        )
    return df


def fetch_candles(
    asset: str,
    timeframe: str = DEFAULT_TIMEFRAME,
    days: int = DEFAULT_DAYS,
    **kwargs,
) -> pd.DataFrame:
    """Alias for :func:`fetch_ohlcv` (the original brief's function name).

    Accepts the same keyword arguments (``exchange``, ``now_ms``, ``limit``,
    ``max_requests``) and forwards them unchanged.
    """
    return fetch_ohlcv(asset, timeframe=timeframe, days=days, **kwargs)


def cache_path(asset: str, timeframe: str = DEFAULT_TIMEFRAME,
               days: int = DEFAULT_DAYS, cache_dir: str = CACHE_DIR) -> str:
    """Deterministic cache filename for one (asset, timeframe, days) dataset.

    The timeframe is part of the filename (``BTCUSDT_1h_730d.csv``,
    ``BTCUSDT_4h_730d.csv``, ``BTCUSDT_1d_730d.csv``) so two timeframes of the
    same asset can never share a cache file.
    """
    slug = _normalize_symbol(asset).replace("/", "")
    return os.path.join(cache_dir, f"{slug}_{timeframe}_{days}d.csv")


def infer_timeframe(candles: pd.DataFrame) -> Optional[str]:
    """Infer the bar timeframe from the median spacing between bars.

    Returns ``"1h"``, ``"4h"`` or ``"1d"`` when the median spacing matches one
    of :data:`SUPPORTED_TIMEFRAMES` within a 1% tolerance, and ``None`` when
    there are fewer than 3 bars or the spacing matches nothing (e.g. 15m bars
    or irregular data). Used to sanity-check that ``--csv`` data really has the
    spacing the ``--timeframe`` flag claims.
    """
    if candles is None or len(candles) < 3:
        return None
    idx = pd.DatetimeIndex(candles.index)
    if len(idx) < 3:
        return None
    deltas = idx.to_series().diff().dropna()
    if deltas.empty:
        return None
    median_ms = float(deltas.dt.total_seconds().median() * 1000.0)
    for tf in SUPPORTED_TIMEFRAMES:
        target = float(TIMEFRAME_MS[tf])
        if abs(median_ms - target) / target <= 0.01:
            return tf
    return None


def save_csv(candles: pd.DataFrame, path: str) -> str:
    """Write candles to ``path`` (creating parent dirs). Returns the path."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    candles.to_csv(path, index_label="timestamp")
    return path


def load_csv(path: str) -> pd.DataFrame:
    """Read a candles CSV written by :func:`save_csv` back into the schema."""
    df = pd.read_csv(path)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.rename(columns={ts_col: "timestamp"}).set_index("timestamp")
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {path} missing OHLCV columns: {missing}")
    df = df[OHLCV_COLUMNS].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return _normalize_index(df)


def load_candles(
    asset: str,
    timeframe: str = DEFAULT_TIMEFRAME,
    days: int = DEFAULT_DAYS,
    csv: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: str = CACHE_DIR,
    exchange=None,
    now_ms: Optional[int] = None,
) -> pd.DataFrame:
    """Load candles: explicit CSV → disk cache → KuCoin public API.

    ``csv`` makes the sandbox fully offline (useful on machines with no
    exchange egress). Otherwise a fetched dataset is cached under
    ``cache_dir`` so repeated research runs cost zero API calls.
    """
    if csv:
        logger.info("Loading candles from CSV %s", csv)
        return load_csv(csv)

    path = cache_path(asset, timeframe, days, cache_dir)
    if use_cache and os.path.exists(path):
        logger.info("Loading candles from cache %s", path)
        return load_csv(path)

    candles = fetch_ohlcv(asset, timeframe=timeframe, days=days,
                          exchange=exchange, now_ms=now_ms)
    if use_cache:
        save_csv(candles, path)
        logger.info("Cached %d candles to %s", len(candles), path)
    return candles
