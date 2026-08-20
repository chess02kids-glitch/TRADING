"""Step 1 tests - HAR forecaster for the alert bot.

Covers: closed-candle-only fetching (in-progress bar dropped), dedupe/sort/
sanitize, symbol normalization, insufficient-history errors, past-only feature
construction (no future leakage), OLS coefficient recovery, exact consistency
with the validated research HAR (``kronos_trading.volatility_baselines.har_forecast``),
honest (unclamped) negative forecasts, and 30-day rolling tercile regimes.
All tests are deterministic - no network, no ccxt.
"""
import math

import numpy as np
import pytest

from kronos_trading.alerts.har_forecaster import (
    DEFAULT_FETCH_CANDLES,
    HAR_MIN_TRAIN,
    MIN_CANDLES,
    REGIME_MIN_BARS,
    REGIME_WINDOW_BARS,
    ROLLING_WINDOWS,
    InsufficientCandlesError,
    HarForecast,
    candle_ranges,
    classify_regime,
    compute_har_features,
    compute_regime_thresholds,
    fetch_candles,
    fit_har_ols,
    predict_next_range,
)
from kronos_trading.types import Candle
from kronos_trading.volatility_baselines import har_forecast

H = 3_600_000
BASE = 1_700_000_000_000
W5, W22 = ROLLING_WINDOWS


def mk(ranges, base=BASE, step=H):
    """Deterministic candles whose ranges are exactly ``ranges``."""
    out = []
    for i, r in enumerate(ranges):
        close = 100.0 + i * 0.05
        out.append(Candle(base + i * step, close - r / 2, close + r / 2,
                          close - r / 2, close, 10.0))
    return out


def raw_rows(n, start_ts=BASE, step=H, ranges=None, volume=10.0):
    """CCXT-style raw OHLCV rows ``[ts, o, h, l, c, v]`` (no in-progress bar)."""
    rows = []
    for i in range(n):
        r = ranges[i] if ranges is not None else 1.0 + (i % 7)
        close = 100.0 + i * 0.05
        rows.append([start_ts + i * step, close - r / 2, close + r / 2,
                     close - r / 2, close, volume])
    return rows


class FakeExchange:
    """Minimal CCXT-compatible double recording fetch calls."""

    def __init__(self, rows=None, error=None):
        self.rows = rows if rows is not None else []
        self.error = error
        self.calls = []

    def fetch_ohlcv(self, symbol, timeframe, limit=None):
        self.calls.append((symbol, timeframe, limit))
        if self.error is not None:
            raise self.error
        return self.rows


# ---------------------------------------------------------------------------
# fetch_candles
# ---------------------------------------------------------------------------

class TestFetchCandles:
    def test_drops_in_progress_bar(self):
        n = 60
        rows = raw_rows(n + 1)  # last row opens at the current hour (in progress)
        now_ms = BASE + n * H + 30_000  # 30 s after the hour
        fx = FakeExchange(rows)
        candles = fetch_candles("BTC/USDT", "1h", n=n + 1, exchange=fx,
                                now_ms=now_ms)
        assert len(candles) == n
        assert candles[-1].timestamp_ms == BASE + (n - 1) * H
        assert all(c.timestamp_ms + H <= now_ms for c in candles)

    def test_insufficient_closed_candles_raises(self):
        rows = raw_rows(MIN_CANDLES - 5)
        fx = FakeExchange(rows)
        with pytest.raises(InsufficientCandlesError) as exc:
            fetch_candles("BTC/USDT", "1h", exchange=fx,
                          now_ms=BASE + 10_000 * H)
        assert "45" in str(exc.value) and "50" in str(exc.value)

    def test_dedupes_sorts_and_skips_malformed(self):
        rows = raw_rows(60)
        rows.append(rows[10])          # duplicate timestamp
        rows.insert(0, [1, "x"])       # malformed (too short)
        fx = FakeExchange(rows)
        candles = fetch_candles("BTC/USDT", "1h", n=60, exchange=fx,
                                now_ms=BASE + 10_000 * H)
        ts = [c.timestamp_ms for c in candles]
        assert ts == sorted(ts)
        assert len(ts) == len(set(ts)) == 60

    def test_symbol_normalization_and_limit(self):
        fx = FakeExchange(raw_rows(60))
        fetch_candles("BTC", "1h", n=60, exchange=fx, now_ms=BASE + 10_000 * H)
        assert fx.calls[0][0] == "BTC/USDT"
        assert fx.calls[0][2] == 60

    def test_default_fetch_size_supports_30day_regime(self):
        # Default fetch must cover REGIME_WINDOW_BARS (720) + HAR warm-up.
        assert DEFAULT_FETCH_CANDLES >= REGIME_WINDOW_BARS + MIN_CANDLES

    def test_rejects_unknown_timeframe(self):
        fx = FakeExchange(raw_rows(60))
        with pytest.raises(ValueError):
            fetch_candles("BTC/USDT", "7h", exchange=fx)

    def test_propagates_exchange_errors(self):
        # The scheduler (Step 5) owns catch-and-continue; here the error must
        # surface so callers can handle it explicitly.
        boom = RuntimeError("rate limit")
        fx = FakeExchange(error=boom)
        with pytest.raises(RuntimeError, match="rate limit"):
            fetch_candles("BTC/USDT", "1h", exchange=fx)


# ---------------------------------------------------------------------------
# candle_ranges
# ---------------------------------------------------------------------------

class TestCandleRanges:
    def test_ranges_and_defensive_cleaning(self):
        candles = [
            Candle(1, 100, 110, 90, 105, 1.0),    # range 20
            Candle(2, 100, 90, 100, 95, 1.0),     # corrupt: high < low -> 0
            Candle(3, 100, float("nan"), 90, 95, 1.0),  # non-finite -> dropped
            Candle(4, 100, 101, 99, 100, 1.0),    # range 2
        ]
        assert candle_ranges(candles) == [20.0, 0.0, 2.0]


# ---------------------------------------------------------------------------
# compute_har_features / fit_har_ols
# ---------------------------------------------------------------------------

class TestHarFeatures:
    def test_past_only_structure_and_shapes(self):
        ranges = [float(i + 1) for i in range(60)]
        X, y = compute_har_features(mk(ranges))
        assert X.shape == (60 - W22, 4)
        assert y.shape == (60 - W22,)
        for row, j in zip(range(len(X)), range(W22, 60)):
            # Features must be strictly before the target bar j.
            assert X[row, 0] == 1.0
            assert X[row, 1] == pytest.approx(ranges[j - 1])
            assert X[row, 2] == pytest.approx(np.mean(ranges[j - W5:j]))
            assert X[row, 3] == pytest.approx(np.mean(ranges[j - W22:j]))
            assert y[row] == pytest.approx(ranges[j])

    def test_too_few_candles_raises(self):
        with pytest.raises(InsufficientCandlesError):
            compute_har_features(mk([1.0] * W22))

    def test_fit_recovers_known_coefficients(self):
        rng = np.random.default_rng(11)
        n = 200
        x1 = rng.uniform(1, 10, n)
        x2 = rng.uniform(5, 50, n)
        x3 = rng.uniform(5, 50, n)
        true = np.array([0.5, 0.3, -0.2, 0.1])
        y = true[0] + true[1] * x1 + true[2] * x2 + true[3] * x3 \
            + rng.normal(0, 0.01, n)
        X = np.column_stack([np.ones(n), x1, x2, x3])
        beta = fit_har_ols(X, y)
        assert beta == pytest.approx(true, abs=0.05)

    def test_fit_requires_min_rows_and_matching_shapes(self):
        X = np.ones((HAR_MIN_TRAIN - 1, 4))
        with pytest.raises(ValueError, match="24"):
            fit_har_ols(X, np.ones(HAR_MIN_TRAIN - 1))
        with pytest.raises(ValueError, match="shape"):
            fit_har_ols(np.ones((30, 3)), np.ones(30))
        with pytest.raises(ValueError, match="length"):
            fit_har_ols(np.ones((30, 4)), np.ones(29))


# ---------------------------------------------------------------------------
# predict_next_range
# ---------------------------------------------------------------------------

class TestPredictNextRange:
    def test_exact_consistency_with_validated_har_model(self):
        """The alert bot must run the exact validated research model."""
        rng = np.random.default_rng(7)
        ranges = [0.5 + (i % 5) * 0.3 + rng.normal(0, 0.05) for i in range(200)]
        candles = mk(ranges)
        result = predict_next_range(candles)
        assert isinstance(result, HarForecast)
        # Bit-for-bit agreement with kronos_trading.volatility_baselines.har_forecast
        assert result.predicted_range == pytest.approx(
            har_forecast(ranges), abs=1e-9)
        assert len(result.coefficients) == 4
        # Coefficients equal a direct manual expanding OLS on the same data.
        X, y = compute_har_features(candles)
        beta = fit_har_ols(X, y)
        assert result.coefficients == pytest.approx(tuple(beta))
        assert result.n_obs == len(ranges) - W22

    def test_requires_min_candles(self):
        with pytest.raises(InsufficientCandlesError, match="50"):
            predict_next_range(mk([1.0] * (MIN_CANDLES - 1)))
        # Exactly 50 works with well-conditioned data.
        result = predict_next_range(mk([1.0 + (i % 3) for i in range(50)]))
        assert math.isfinite(result.predicted_range)

    def test_returns_honest_negative_forecast_unclamped(self):
        # OLS extrapolation below zero must be returned as-is (validated-model
        # behavior); clamping would corrupt calibration logging.
        ranges = [20.0] * 48 + [0.0] * 2
        candles = mk(ranges)
        result = predict_next_range(candles)
        assert result.predicted_range < 0.0
        X, y = compute_har_features(candles)
        beta = fit_har_ols(X, y)
        xf = np.asarray([1.0, ranges[-1],
                         np.mean(ranges[-W5:]), np.mean(ranges[-W22:])])
        assert result.predicted_range == pytest.approx(float(xf @ beta), abs=1e-9)

    def test_in_progress_bar_never_influences_prediction(self):
        """Fetch drops the forming bar; the pipeline prediction must equal the
        prediction computed from closed candles only - even when the forming
        bar has an absurd range."""
        n = 55
        closed = raw_rows(n)
        huge_open = BASE + n * H
        closed.append([huge_open, 100.0, 1e9, 0.0, 100.0, 10.0])  # in progress
        now_ms = huge_open + 30_000
        fetched = fetch_candles("BTC/USDT", "1h", n=n + 1, exchange=FakeExchange(closed),
                                now_ms=now_ms)
        p_fetched = predict_next_range(fetched)
        p_closed = predict_next_range(mk([1.0 + (i % 7) for i in range(n)]))
        assert p_fetched.predicted_range == pytest.approx(p_closed.predicted_range)


# ---------------------------------------------------------------------------
# classify_regime / compute_regime_thresholds
# ---------------------------------------------------------------------------

class TestRegime:
    def test_terciles(self):
        # 480 bars of 2.0 + 240 bars of 5.0 -> q33 = 2.0, q66 ~= 3.0.
        ranges = [2.0] * 480 + [5.0] * 240
        assert classify_regime(1.0, ranges) == "low"
        assert classify_regime(2.5, ranges) == "medium"
        assert classify_regime(4.0, ranges) == "high"
        # Boundary equality falls in the middle bucket (strict comparisons).
        assert classify_regime(2.0, ranges) == "medium"

    def test_rolling_30day_window_excludes_old_data(self):
        old = [0.0001] * (REGIME_WINDOW_BARS - 720)  # stale ultra-quiet regime
        recent = [2.0] * 480 + [5.0] * 240
        combined = old + recent
        assert compute_regime_thresholds(combined) == pytest.approx(
            compute_regime_thresholds(recent))
        # If the old data were included, 1.0 would look high; with the window
        # it is low.
        assert classify_regime(1.0, combined) == "low"

    def test_insufficient_history_returns_none(self):
        assert classify_regime(1.0, [1.0] * (REGIME_MIN_BARS - 1)) is None
        assert compute_regime_thresholds([]) is None

    def test_thresholds_use_actual_ranges_only(self):
        # Percentiles must come from historical actual ranges, never from the
        # prediction being classified.
        ranges = [2.0] * 480 + [5.0] * 240
        q_low, q_high = compute_regime_thresholds(ranges)
        assert q_low == pytest.approx(np.percentile(ranges[-REGIME_WINDOW_BARS:],
                                                    100.0 / 3.0))
        assert q_high == pytest.approx(np.percentile(ranges[-REGIME_WINDOW_BARS:],
                                                     200.0 / 3.0))
