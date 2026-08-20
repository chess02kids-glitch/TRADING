"""Step 3 tests - breakout detection and live calibration (pure functions).

Covers: breakout classification incl. severity bands and the division-by-zero
guards (zero/negative predictions), prediction-error arithmetic, live
calibration aggregates (HAR MAE vs lag-1 persistence MAE, worst/best ratios,
breakout stats, degradation flag), and both message formatters.

Deterministic, no network, no DB - the ``make_history`` helper builds
synthetic ``get_prediction_history`` output (DESC, newest first) with all
columns filled.
"""
from datetime import datetime, timedelta, timezone

import pytest

from kronos_trading.alerts.breakout_detector import (
    DEGRADATION_FACTOR,
    MIN_CALIBRATION_OBS,
    RECENT_30D_ROWS,
    RECENT_7D_ROWS,
    BreakoutResult,
    LiveCalibration,
    PredictionError,
    _is_degrading,
    check_breakout,
    compute_prediction_error,
    format_breakout_message,
    format_calibration_message,
    get_live_calibration,
)

UTC = timezone.utc
T0 = datetime(2024, 1, 15, tzinfo=UTC)


def make_history(n: int,
                 har_errors: list = None,
                 breakout_flags: list = None,
                 regimes: list = None,
                 predicted_ranges: list = None) -> list:
    """Synthetic get_prediction_history() output - all rows completed.

    Rows are returned DESC (newest first). ``har_errors[i]`` is the error of
    chronological row ``i`` (oldest first in the argument), so
    ``actual_range = har_predicted_range + har_errors[i]``.
    """
    errors = har_errors if har_errors is not None else [0.0] * n
    predicted = predicted_ranges if predicted_ranges is not None else [100.0] * n
    flags = breakout_flags if breakout_flags is not None else [0] * n
    regs = regimes if regimes is not None else [None] * n
    assert len(errors) == len(predicted) == len(flags) == len(regs) == n

    rows = []
    for i in range(n):  # i = chronological index, 0 = oldest
        pred = float(predicted[i])
        err = float(errors[i])
        ts_str = (T0 + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append({
            "id": i + 1,
            "timestamp": ts_str,
            "asset": "BTC/USDT",
            "timeframe": "1h",
            "har_predicted_range": pred,
            "coef_b0": 1.0, "coef_b1": 0.5, "coef_b2": 0.3, "coef_b3": 0.1,
            "n_obs": 178,
            "regime": regs[i],
            "actual_range": pred + err,
            "prediction_error": err,
            "abs_prediction_error": abs(err),
            "breakout_flag": flags[i],
            "created_at": ts_str,
        })
    return list(reversed(rows))  # DESC: newest first


# ---------------------------------------------------------------------------
# check_breakout
# ---------------------------------------------------------------------------

class TestCheckBreakout:
    def test_no_breakout_below_threshold(self):
        r = check_breakout(actual_range=150.0, predicted_range=100.0)
        assert r.is_breakout is False
        assert r.ratio == pytest.approx(1.5)
        assert r.severity == "none"

    def test_breakout_at_exact_threshold(self):
        r = check_breakout(200.0, 100.0)
        assert r.is_breakout is True
        assert r.ratio == pytest.approx(2.0)

    def test_breakout_above_threshold(self):
        r = check_breakout(300.0, 100.0)
        assert r.is_breakout is True
        assert r.ratio == pytest.approx(3.0)

    def test_severity_none(self):
        assert check_breakout(150.0, 100.0).severity == "none"

    def test_severity_moderate(self):
        assert check_breakout(250.0, 100.0).severity == "moderate"

    def test_severity_severe(self):
        assert check_breakout(400.0, 100.0).severity == "severe"

    def test_severity_extreme(self):
        assert check_breakout(600.0, 100.0).severity == "extreme"

    def test_zero_predicted_range_safe(self):
        r = check_breakout(100.0, 0.0)  # must not raise ZeroDivisionError
        assert r.is_breakout is False
        assert r.ratio is None
        assert r.severity == "none"

    def test_negative_predicted_range_safe(self):
        r = check_breakout(100.0, -5.0)  # honest unclamped HAR output
        assert r.is_breakout is False
        assert r.ratio is None
        assert r.severity == "none"

    def test_custom_threshold(self):
        r = check_breakout(150.0, 100.0, threshold=1.5)
        assert r.is_breakout is True
        # Severity bands stay fixed at 2.0/3.0/5.0 regardless of threshold.
        assert r.severity == "none"


# ---------------------------------------------------------------------------
# compute_prediction_error
# ---------------------------------------------------------------------------

class TestPredictionError:
    def test_over_prediction_error(self):
        e = compute_prediction_error(120.0, 100.0)
        assert e.error == pytest.approx(20.0)
        assert e.abs_error == pytest.approx(20.0)
        assert e.direction == "over"
        assert e.pct_error == pytest.approx(20.0)

    def test_under_prediction_error(self):
        e = compute_prediction_error(80.0, 100.0)
        assert e.error == pytest.approx(-20.0)
        assert e.abs_error == pytest.approx(20.0)
        assert e.direction == "under"
        assert e.pct_error == pytest.approx(-20.0)

    def test_exact_prediction_error(self):
        e = compute_prediction_error(100.0, 100.0)
        assert e.error == pytest.approx(0.0)
        assert e.direction == "exact"
        assert e.pct_error == pytest.approx(0.0)

    def test_pct_error_none_when_predicted_zero(self):
        e = compute_prediction_error(100.0, 0.0)
        assert e.pct_error is None
        assert e.direction == "over"

    def test_negative_predicted_pct_error_none(self):
        e = compute_prediction_error(100.0, -5.0)
        assert e.pct_error is None
        assert e.error == pytest.approx(105.0)


# ---------------------------------------------------------------------------
# get_live_calibration
# ---------------------------------------------------------------------------

class TestLiveCalibration:
    def test_returns_none_below_minimum(self):
        assert get_live_calibration(make_history(MIN_CALIBRATION_OBS - 1)) is None
        assert get_live_calibration([]) is None

    def test_har_mae_correct(self):
        cal = get_live_calibration(
            make_history(24, har_errors=[1.0] * 12 + [-2.0] * 12))
        assert cal is not None
        assert cal.n_obs == 24
        assert cal.har_mae == pytest.approx(1.5)      # (12*1 + 12*2) / 24
        assert cal.mean_bias == pytest.approx(-0.5)   # (12 - 24) / 24

    def test_har_beats_persistence_true(self):
        # HAR predicts a trending series perfectly; lag-1 persistence lags by
        # exactly one step, so persistence MAE = 1.0 > HAR MAE = 0.0.
        n = 24
        cal = get_live_calibration(
            make_history(n, har_errors=[0.0] * n,
                         predicted_ranges=[100.0 + i for i in range(n)]))
        assert cal is not None
        assert cal.har_mae == pytest.approx(0.0)
        assert cal.persistence_mae == pytest.approx(1.0)
        assert cal.har_beats_persistence is True

    def test_har_beats_persistence_false(self):
        # Constant -5 error: HAR MAE = 5.0; lag-1 persistence error is -4.0
        # (MAE 4.0), so HAR loses to the naive baseline.
        n = 24
        cal = get_live_calibration(
            make_history(n, har_errors=[-5.0] * n,
                         predicted_ranges=[100.0 + i for i in range(n)]))
        assert cal is not None
        assert cal.har_mae == pytest.approx(5.0)
        assert cal.persistence_mae == pytest.approx(4.0)
        assert cal.har_beats_persistence is False

    def test_worst_and_best_ratio(self):
        n = 24
        cal = get_live_calibration(
            make_history(n, har_errors=[0.0] * 20 + [50.0, 100.0, 200.0, 400.0]))
        assert cal is not None
        # Newest four rows have ratios 5.0, 3.0, 2.0, 1.5; the rest are 1.0.
        assert cal.worst_ratio == pytest.approx(5.0)
        assert cal.best_ratio == pytest.approx(1.0)

    def test_worst_ratio_none_all_negative_predictions(self):
        cal = get_live_calibration(
            make_history(24, predicted_ranges=[-5.0] * 24))
        assert cal is not None
        assert cal.worst_ratio is None
        assert cal.best_ratio is None
        assert cal.n_obs == 24  # ratios absent, other stats still computed

    def test_is_degrading_true(self):
        # Newest 168 rows (7d) have error 10.0; older 552 rows have 1.0.
        # har_mae = 3.1, recent_mae_7d = 10.0 > 3.1 * 1.5.
        n = RECENT_7D_ROWS + 552
        cal = get_live_calibration(
            make_history(n, har_errors=[1.0] * 552 + [10.0] * RECENT_7D_ROWS))
        assert cal is not None
        assert cal.har_mae == pytest.approx(3.1)
        assert cal.recent_mae_7d == pytest.approx(10.0)
        assert cal.recent_mae_30d == pytest.approx(3.1)
        assert cal.recent_mae_7d > cal.har_mae * DEGRADATION_FACTOR
        assert cal.is_degrading is True

    def test_is_degrading_false_when_recent_none(self):
        # With the 24-row calibration minimum, get_live_calibration only
        # returns a result when the 7-day window has >= 24 rows, so the
        # None branch of the degradation guard is defensive. Exercise the
        # exact guard the function uses, plus the minimal-history case
        # (thin recent data never flags degradation).
        assert _is_degrading(None, 1.0) is False
        cal = get_live_calibration(
            make_history(MIN_CALIBRATION_OBS, har_errors=[0.0] * MIN_CALIBRATION_OBS))
        assert cal is not None
        assert cal.is_degrading is False

    def test_breakout_count_and_rate(self):
        cal = get_live_calibration(
            make_history(100, breakout_flags=[1] * 5 + [0] * 95))
        assert cal is not None
        assert cal.breakout_count == 5
        assert cal.breakout_rate == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# format_breakout_message
# ---------------------------------------------------------------------------

class TestFormatBreakoutMessage:
    def test_no_message_when_no_breakout(self):
        r = check_breakout(150.0, 100.0)
        assert format_breakout_message("BTC/USDT", "1h", r,
                                       "2024-01-15T14:00:00Z") == ""

    def test_moderate_message_contains_key_fields(self):
        r = check_breakout(250.0, 100.0)
        msg = format_breakout_message("BTC/USDT", "1h", r,
                                      "2024-01-15T14:00:00Z")
        assert msg != ""
        assert "⚠️ VOLATILITY SPIKE — BTC/USDT 1h" in msg
        assert "HAR predicted: $100.00" in msg
        assert "Actual range:  $250.00" in msg
        assert "Ratio: 2.50× expected" in msg
        assert "Severity: MODERATE" in msg
        assert "Time: 2024-01-15T14:00:00Z" in msg

    def test_severe_message_emoji(self):
        r = check_breakout(400.0, 100.0)
        msg = format_breakout_message("ETH/USDT", "1h", r, "2024-01-15T14:00:00Z")
        assert "🚨 VOLATILITY BREAKOUT — ETH/USDT 1h" in msg
        assert "Severity: SEVERE" in msg

    def test_extreme_message_emoji(self):
        r = check_breakout(600.0, 100.0)
        msg = format_breakout_message("BTC/USDT", "1h", r, "2024-01-15T14:00:00Z")
        assert "🔴 EXTREME VOLATILITY — BTC/USDT 1h" in msg
        assert "Severity: EXTREME ⚠️⚠️⚠️" in msg

    def test_custom_threshold_breakout_still_renders(self):
        # threshold=1.5 with ratio 1.5: is_breakout=True, severity "none" -
        # the message still renders (moderate template fallback).
        r = check_breakout(150.0, 100.0, threshold=1.5)
        msg = format_breakout_message("BTC/USDT", "1h", r, "2024-01-15T14:00:00Z")
        assert msg != ""
        assert "⚠️ VOLATILITY SPIKE" in msg
        assert "Severity: NONE" in msg


# ---------------------------------------------------------------------------
# format_calibration_message
# ---------------------------------------------------------------------------

class TestFormatCalibrationMessage:
    def test_calibration_message_contains_mae(self):
        cal = get_live_calibration(
            make_history(24, har_errors=[1.0] * 12 + [-2.0] * 12))
        assert cal is not None
        msg = format_calibration_message("BTC/USDT", "1h", cal)
        assert "📊 HAR CALIBRATION REPORT" in msg
        assert "HAR MAE:         1.5000" in msg
        assert "Observations: 24" in msg
        assert "HAR beats naive: ✅ YES" in msg

    def test_calibration_message_degrading_warning(self):
        n = RECENT_7D_ROWS + 552
        cal = get_live_calibration(
            make_history(n, har_errors=[1.0] * 552 + [10.0] * RECENT_7D_ROWS))
        assert cal is not None and cal.is_degrading is True
        msg = format_calibration_message("BTC/USDT", "1h", cal)
        assert "Degrading:  ⚠️ YES" in msg
        assert "7-day MAE:  10.0000" in msg

    def test_calibration_message_na_placeholders(self):
        cal = get_live_calibration(
            make_history(24, predicted_ranges=[-5.0] * 24))
        assert cal is not None
        msg = format_calibration_message("BTC/USDT", "1h", cal)
        assert "Worst ratio: N/A" in msg
        assert "Best ratio:  N/A" in msg
        assert "Degrading:  ✅ NO" in msg
