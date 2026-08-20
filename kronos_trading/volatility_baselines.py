"""Phase 5b - serious volatility/range baselines (all past-only, fixed a priori).

These baselines replace/augment the weak "previous range" persistence used in
the Phase 5 range target. Every constant here is FIXED before any results are
looked at; nothing is tuned against out-of-sample windows.

Baselines (all operate on the sequence of closed-candle ranges
``range_t = high_t - low_t`` strictly before the prediction timestamp):

* **A. previous range**      ``range_{t-1}`` (the last closed bar's range).
* **B. rolling mean range**  mean of the last 5 and last 22 closed ranges
  (windows fixed a priori).
* **C. EWMA range**          exponentially weighted average with a FIXED decay
  ``alpha = 2/(span+1)``, ``span = 22``, seeded on the first closed range and
  recursed over the full context (past-only).
* **D. HAR-style range**     ``forecast = beta0 + beta1*range_{t-1} +
  beta2*mean_range_last_5 + beta3*mean_range_last_22`` fitted by OLS on the
  context's own lag structure using an EXPANDING past-only window
  (refit at each step, minimum 24 training rows). Negative forecasts are left
  as-is (honest OLS output).

Regimes are past-only terciles of a rolling 22-bar mean range, using expanding
quantiles computed from the context itself (never from the target or future).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

# --- Fixed a priori constants (do NOT tune against results) -----------------
EWMA_SPAN = 22                 # alpha = 2/(span+1) = 2/23
ROLLING_WINDOWS = (5, 22)      # rolling mean windows
HAR_MIN_TRAIN = 24             # minimum OLS training rows for HAR
REGIME_MEASURE_WINDOW = 22     # rolling window for the regime measure
REGIME_LOW_Q = 1.0 / 3.0       # tercile boundary (fixed)
REGIME_HIGH_Q = 2.0 / 3.0


def _ewma_alpha(span: int) -> float:
    return 2.0 / (span + 1.0)


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def rolling_mean_range(ranges: List[float], window: int) -> Optional[float]:
    """Mean of the last ``window`` closed ranges (past-only)."""
    if len(ranges) < window:
        return None
    return _mean(ranges[-window:])


def ewma_range(ranges: List[float], span: int = EWMA_SPAN) -> Optional[float]:
    """Exponentially weighted average of ranges, seeded on the first value.

    Recurses over all provided (closed) ranges; the returned value is the EWMA
    at the last closed bar. ``alpha = 2/(span+1)`` (fixed convention).
    """
    if not ranges:
        return None
    a = _ewma_alpha(span)
    e = ranges[0]
    for r in ranges[1:]:
        e = a * r + (1.0 - a) * e
    return e


def har_forecast(ranges: List[float], min_train: int = HAR_MIN_TRAIN,
                 windows=(5, 22)) -> Optional[float]:
    """HAR-style range forecast with expanding past-only OLS.

    Training rows (all strictly before the prediction target): for each closed
    bar ``j`` in the context, target ``range_j`` and features
    ``[1, range_{j-1}, mean(range_{j-5..j}), mean(range_{j-22..j})]``. The
    forecast features are computed at the last closed bar. Returns ``None`` when
    there is insufficient history (or the fit is degenerate).
    """
    w5, w22 = windows
    n = len(ranges)
    if n < w22 + min_train:
        return None
    X: List[List[float]] = []
    y: List[float] = []
    for j in range(w22, n):  # j in [w22, n-1]; features strictly before bar j
        X.append([1.0, ranges[j - 1],
                  _mean(ranges[j - w5:j]),
                  _mean(ranges[j - w22:j])])
        y.append(ranges[j])
    if len(y) < min_train:
        return None
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float)
    try:
        beta = np.linalg.lstsq(Xa, ya, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    xf = np.asarray([1.0, ranges[n - 1],
                     _mean(ranges[n - w5:n]),
                     _mean(ranges[n - w22:n])], dtype=float)
    forecast = float(xf @ beta)
    if not math.isfinite(forecast):
        return None
    return forecast  # negative values are left as-is (honest OLS output)


def volatility_forecasts(ranges: List[float]) -> Dict[str, Optional[float]]:
    """All fixed volatility baselines for one closed-candle context."""
    return {
        'prev': ranges[-1] if ranges else None,
        'rolling5': rolling_mean_range(ranges, ROLLING_WINDOWS[0]),
        'rolling22': rolling_mean_range(ranges, ROLLING_WINDOWS[1]),
        'ewma': ewma_range(ranges, EWMA_SPAN),
        'har': har_forecast(ranges, HAR_MIN_TRAIN, ROLLING_WINDOWS),
    }


def assign_regime(ranges: List[float],
                  measure_window: int = REGIME_MEASURE_WINDOW) -> str:
    """Assign a past-only volatility regime from the context's own history.

    The regime measure is a rolling ``measure_window``-bar mean range. Its value
    at the last closed bar is compared against the 1/3 and 2/3 quantiles of the
    measure's expanding past-only series. Returns one of
    ``'low'`` / ``'medium'`` / ``'high'`` (or ``'undefined'`` when there is too
    little history).
    """
    n = len(ranges)
    if n < measure_window + 1:
        return 'undefined'
    series = [_mean(ranges[j - measure_window + 1:j + 1])
              for j in range(measure_window - 1, n)]
    current = series[-1]
    q_low = float(np.percentile(series, 100.0 * REGIME_LOW_Q))
    q_high = float(np.percentile(series, 100.0 * REGIME_HIGH_Q))
    if current < q_low:
        return 'low'
    if current > q_high:
        return 'high'
    return 'medium'
