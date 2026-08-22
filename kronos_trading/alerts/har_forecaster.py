"""HAR volatility forecaster for the alert bot (Step 1).

This module wraps the *validated* HAR range model from the 8-phase research
study. The model specification is fixed and matches
``kronos_trading.volatility_baselines.har_forecast`` exactly:

    R_{t+1} = B0 + B1 * R_t + B2 * mean(R_{t-4:t}) + B3 * mean(R_{t-21:t})

where ``R_t = high_t - low_t`` (candle range) and the mean windows are the
5-bar and 22-bar means ending at the last *closed* bar ``t``. Coefficients are
estimated by expanding-window OLS on past data only - the target bar is always
strictly after the features that predict it, so there is no future leakage.

Two deliberate deviations from the alert spec draft, both documented here:

1. ``fetch_candles`` defaults to ``n=800`` instead of ``n=100``. The regime
   classifier requires a rolling 30-day percentile (720 x 1h bars); 100 bars
   cannot support it. 800 = 720 + 80 warm-up buffer. HAR itself only needs
   ~50. Pass ``n=100`` explicitly if a smaller fetch is ever wanted.
2. ``predict_next_range`` returns a small dataclass (``HarForecast``) rather
   than a bare float, because the logger (Step 2) must persist the fitted
   coefficients B0..B3 alongside the prediction. ``har_forecast.predicted_range``
   is the value used everywhere downstream.

Negative OLS forecasts are returned as-is (honest OLS output), matching the
validated research implementation - never clamped, never floored at zero.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from kronos_trading.types import Candle
from kronos_trading.volatility_baselines import (
    HAR_MIN_TRAIN,      # 24 - fixed a priori in the validated research model
    ROLLING_WINDOWS,    # (5, 22) - fixed a priori, do not tune
)

logger = logging.getLogger(__name__)

# --- Alert-bot constants (fixed, per the Phase 1 spec) -----------------------
MIN_CANDLES = 50              # spec: minimum 50 candles for HAR estimation
DEFAULT_FETCH_CANDLES = 800   # 720 (30d x 24h) + 80 buffer - see module docstring
REGIME_WINDOW_BARS = 720      # rolling 30-day percentile window (1h bars)
REGIME_LOW_Q = 1.0 / 3.0      # 33rd percentile boundary (fixed)
REGIME_HIGH_Q = 2.0 / 3.0     # 66th percentile boundary (fixed)
REGIME_MIN_BARS = 100         # below this, regime percentiles are not meaningful

# Timeframe -> bar length in ms (mirrors scripts/data/fetcher.py).
TIMEFRAME_MS = {
    "1h": 3600000,
    "4h": 14400000,
    "1d": 86400000,
}


class InsufficientCandlesError(ValueError):
    """Fewer closed candles than the pipeline requires (MIN_CANDLES)."""


class DegenerateFitError(RuntimeError):
    """OLS fit produced non-finite coefficients (degenerate design matrix)."""


@dataclass(frozen=True)
class HarForecast:
    """One HAR prediction for the next (not yet closed) bar.

    ``predicted_range`` is the raw OLS output for the next bar's range
    (``high - low``); it may be negative when the fit is badly extrapolating -
    this is the honest, validated-model output and is logged as-is.
    ``coefficients`` is ``(B0, B1, B2, B3)`` for persistence in the logger.
    ``regime`` is optional metadata (``'low'``/``'medium'``/``'high'``) - it is
    NOT computed by ``predict_next_range``; callers attach it afterwards from
    ``classify_regime`` (e.g. ``dataclasses.replace(forecast, regime=...)``).

    Phase 9A bias correction: ``bias_correction`` is the mean signed error of
    HAR over the last 7 days of completed bars (``mean(actual - predicted)``;
    negative ⇒ HAR over-predicts, so the correction reduces the forecast).
    ``corrected_predicted_range = predicted_range + bias_correction``. Both
    default to ``0.0`` (no correction) so uncorrected forecasts are unchanged.
    The original HAR OLS formula is never modified.
    """
    predicted_range: float
    coefficients: Tuple[float, float, float, float]
    n_obs: int
    regime: Optional[str] = None
    bias_correction: float = 0.0
    corrected_predicted_range: float = 0.0


def _normalize_symbol(asset: str) -> str:
    """Accept ``'BTC/USDT'`` or ``'BTC'`` and return a CCXT spot symbol."""
    symbol = asset.strip().upper()
    if "/" not in symbol:
        symbol = f"{symbol}/USDT"
    return symbol


def _default_exchange():
    """CCXT KuCoin public client - no API keys (Binance blocks GitHub Action US IPs)."""
    import ccxt  # lazy import so the module stays importable without ccxt

    return ccxt.kucoin({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })


def candle_ranges(candles: List[Candle]) -> List[float]:
    """Ranges ``high - low`` of all candles, oldest to newest.

    Defensive cleaning: negative ranges (corrupt rows) are clamped to 0.0 and
    non-finite ranges are dropped with a warning - one bad candle must not
    poison the OLS fit.
    """
    out: List[float] = []
    for c in candles:
        r = float(c.high) - float(c.low)
        if not np.isfinite(r):
            logger.warning("Dropping candle with non-finite range: ts=%s h=%s l=%s",
                           c.timestamp_ms, c.high, c.low)
            continue
        out.append(max(r, 0.0))
    return out


def fetch_candles(
    asset: str,
    timeframe: str = "1h",
    n: int = DEFAULT_FETCH_CANDLES,
    exchange=None,
    now_ms: Optional[int] = None,
) -> List[Candle]:
    """Fetch the latest ``n`` *closed* candles for ``asset`` via CCXT.

    Args:
        asset: ``'BTC/USDT'`` or ``'BTC'`` (``/USDT`` appended when missing).
        timeframe: ``'1h'`` (default), ``'4h'`` or ``'1d'``.
        n: number of candles to request. Default 800 so the 30-day regime
            percentile (720 bars) is computable from the first run.
        exchange: injectable CCXT-compatible object with
            ``fetch_ohlcv(symbol, timeframe, limit)`` (test hook). When None,
            a public Binance client is created - no API keys.
        now_ms: current time in ms (test hook). Defaults to the real clock.

    Returns:
        Closed candles sorted ascending by timestamp, deduplicated (last
        occurrence wins).

    Raises:
        InsufficientCandlesError: fewer than MIN_CANDLES closed candles
            available after dropping the in-progress bar.
    """
    symbol = _normalize_symbol(asset)
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"Unsupported timeframe {timeframe!r}; "
                         f"allowed: {sorted(TIMEFRAME_MS)}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if exchange is None:
        exchange = _default_exchange()
    if now_ms is None:
        now_ms = int(time.time() * 1000)

    raw = exchange.fetch_ohlcv(symbol, timeframe, limit=n)
    if raw is None:
        raw = []

    # Sanitize rows, drop the in-progress (unclosed) bar: a candle is closed
    # only when its open time + bar length <= now. CCXT includes the current
    # forming candle as the last row - including it would leak the current
    # bar's range into the "past" features.
    bar_ms = TIMEFRAME_MS[timeframe]
    seen: Dict[int, Candle] = {}
    for row in raw:
        if len(row) < 6:
            logger.warning("Skipping malformed OHLCV row: %r", row)
            continue
        ts = int(row[0])
        if ts + bar_ms > now_ms:
            continue  # still forming
        try:
            candle = Candle(
                timestamp_ms=ts,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
        except (TypeError, ValueError):
            logger.warning("Skipping unparseable OHLCV row: %r", row)
            continue
        seen[ts] = candle  # duplicate timestamp: last occurrence wins

    candles = [seen[ts] for ts in sorted(seen)]
    if len(candles) < MIN_CANDLES:
        raise InsufficientCandlesError(
            f"Only {len(candles)} closed {timeframe} candles for {symbol} "
            f"(need >= {MIN_CANDLES}); exchange returned {len(raw)} raw rows"
        )
    return candles


def compute_har_features(candles: List[Candle]) -> Tuple[np.ndarray, np.ndarray]:
    """Design matrix + targets for expanding-window HAR OLS (past-only).

    For each training row ``j`` (``j`` from ``w22`` to the last candle), the
    target is ``R_j`` and the features are computed strictly from bars before
    ``j``:

        X_j = [1, R_{j-1}, mean(R_{j-5:j-1}), mean(R_{j-22:j-1})]

    Returns ``(X, y)`` as float arrays with shape ``(n - 22, 4)`` and
    ``(n - 22,)``.
    """
    ranges = candle_ranges(candles)
    w5, w22 = ROLLING_WINDOWS
    n = len(ranges)
    if n < w22 + 1:
        raise InsufficientCandlesError(
            f"Need at least {w22 + 1} candles to build HAR features, got {n}"
        )
    X: List[List[float]] = []
    y: List[float] = []
    for j in range(w22, n):
        X.append([
            1.0,
            ranges[j - 1],
            float(np.mean(ranges[j - w5:j])),
            float(np.mean(ranges[j - w22:j])),
        ])
        y.append(ranges[j])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def fit_har_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS coefficients ``(B0, B1, B2, B3)`` via ``np.linalg.lstsq``.

    Uses every available training row (expanding window: nothing is held back
    or rolled off). ``HAR_MIN_TRAIN`` (24) is the minimum row count, matching
    the validated research model.

    Raises:
        ValueError: fewer rows than required or shape mismatch.
        DegenerateFitError: the fit produced non-finite coefficients.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(f"X must have shape (rows, 4), got {X.shape}")
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError(f"y must have length {X.shape[0]}, got {len(y)}")
    if len(y) < HAR_MIN_TRAIN:
        raise ValueError(
            f"Need >= {HAR_MIN_TRAIN} training rows for HAR OLS, got {len(y)}"
        )
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
        raise DegenerateFitError(f"lstsq failed: {exc}") from exc
    if not np.all(np.isfinite(beta)):
        raise DegenerateFitError(f"Non-finite OLS coefficients: {beta}")
    return beta


def predict_next_range(candles: List[Candle]) -> HarForecast:
    """HAR prediction of the next bar's range from closed candles only.

    Features are evaluated at the last closed bar ``t`` and the forecast is
    for bar ``t + 1`` (the bar currently forming when the bot runs on the
    hour). The OLS fit uses all rows strictly before the target of each row -
    never future data.

    Raises:
        InsufficientCandlesError: fewer than MIN_CANDLES (50) candles.
        DegenerateFitError: OLS fit is degenerate.
    """
    ranges = candle_ranges(candles)
    if len(ranges) < MIN_CANDLES:
        raise InsufficientCandlesError(
            f"Need >= {MIN_CANDLES} closed candles for HAR, got {len(ranges)}"
        )
    X, y = compute_har_features(candles)
    beta = fit_har_ols(X, y)
    w5, w22 = ROLLING_WINDOWS
    xf = np.asarray([
        1.0,
        ranges[-1],
        float(np.mean(ranges[-w5:])),
        float(np.mean(ranges[-w22:])),
    ], dtype=float)
    forecast = float(xf @ beta)
    if not np.isfinite(forecast):
        raise DegenerateFitError(f"Non-finite forecast: {forecast}")
    return HarForecast(
        predicted_range=forecast,
        coefficients=(float(beta[0]), float(beta[1]), float(beta[2]), float(beta[3])),
        n_obs=int(len(y)),
    )


def last_completed_open_close(candles: List[Candle]) -> Tuple[Optional[float], Optional[float]]:
    """Open/close of the most recent *completed* candle (Phase 9A).

    The candle series passed to the forecaster is ascending and contains only
    closed bars (``fetch_candles`` drops the forming bar), so the last element
    is the most recent completed bar. Its ``open`` / ``close`` are exactly what
    Phase 9A needs to derive the breakout-bar direction, and the scheduler
    forwards them into ``check_breakout(..., candle_open=, candle_close=)``.

    Returns ``(None, None)`` for an empty series so callers can treat a missing
    bar defensively without a special-case exception.
    """
    if not candles:
        return None, None
    last = candles[-1]
    return float(last.open), float(last.close)


# ---------------------------------------------------------------------------
# Phase 9A bias correction (additive, past-only; never modifies HAR OLS)
# ---------------------------------------------------------------------------

def _parse_ts(value) -> Optional[datetime]:
    """Parse an ISO8601 string / datetime / Timestamp to a UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        import pandas as pd  # local import to avoid a hard pandas dependency
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts is None or not np.isfinite(ts.value):
        return None
    return ts.to_pydatetime().astimezone(timezone.utc) if ts.tzinfo else ts.to_pydatetime().replace(tzinfo=timezone.utc)


def compute_bias_correction(
    history: List[dict],
    window_days: int = 7,
    min_bars: int = 24,
) -> float:
    """Mean signed HAR error over the last ``window_days`` of completed bars.

    Uses **only past completed predictions** (rows with a non-null
    ``actual_range``) - never look-ahead. The reference "now" is the newest
    timestamp in ``history`` (deterministic, clock-independent). Rows older
    than ``window_days`` from that reference are excluded.

    Formula::

        bias = mean(actual_range - har_predicted_range)

    over the in-window completed bars. Returns ``0.0`` when fewer than
    ``min_bars`` (default 24) in-window completed bars are available, so a
    too-short window applies no correction rather than a noisy one.
    """
    if not history:
        return 0.0

    parsed = []
    for row in history:
        actual = row.get("actual_range")
        predicted = row.get("har_predicted_range")
        ts = _parse_ts(row.get("timestamp"))
        if ts is None:
            continue
        if actual is None or predicted is None:
            continue
        try:
            a = float(actual)
            p = float(predicted)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(a) and np.isfinite(p)):
            continue
        parsed.append((ts, a - p))

    if not parsed:
        return 0.0

    ref = max(ts for ts, _ in parsed)
    cutoff = ref - timedelta(days=int(window_days))
    in_window = [err for ts, err in parsed if ts >= cutoff]
    if len(in_window) < int(min_bars):
        return 0.0
    return float(sum(in_window) / len(in_window))


def apply_bias_correction(forecast: "HarForecast", bias: float) -> "HarForecast":
    """Return a copy of ``forecast`` with the additive bias correction applied.

    ``corrected_predicted_range = predicted_range + bias`` (``bias`` is
    ``mean(actual - predicted)``: negative when HAR over-predicts, so the
    corrected range is smaller). The original HAR OLS prediction
    (``predicted_range``) is preserved unchanged.
    """
    return replace(forecast,
                   bias_correction=float(bias),
                   corrected_predicted_range=float(forecast.predicted_range) + float(bias))


def compute_regime_thresholds(
    historical_ranges: List[float],
    window_bars: int = REGIME_WINDOW_BARS,
) -> Optional[Tuple[float, float]]:
    """33rd/66th percentiles of the last ``window_bars`` *actual* ranges.

    Percentiles are computed from historical data only (never from the
    prediction being classified). Returns ``None`` when fewer than
    ``REGIME_MIN_BARS`` (100) ranges are available.
    """
    if len(historical_ranges) < REGIME_MIN_BARS:
        return None
    window = historical_ranges[-window_bars:]
    q_low = float(np.percentile(window, 100.0 * REGIME_LOW_Q))
    q_high = float(np.percentile(window, 100.0 * REGIME_HIGH_Q))
    return q_low, q_high


def classify_regime(
    predicted: float,
    historical_ranges: List[float],
    window_bars: int = REGIME_WINDOW_BARS,
) -> Optional[str]:
    """Classify a HAR prediction against rolling 30-day terciles.

    LOW     : predicted < 33rd percentile of actual ranges
    MEDIUM  : 33rd <= predicted <= 66th percentile
    HIGH    : predicted > 66th percentile

    Returns ``None`` when there is not enough history for meaningful
    percentiles (callers should omit the regime line from the message).
    """
    thresholds = compute_regime_thresholds(historical_ranges, window_bars)
    if thresholds is None:
        return None
    q_low, q_high = thresholds
    if predicted < q_low:
        return "low"
    if predicted > q_high:
        return "high"
    return "medium"
