"""Step 2 tests - SQLite prediction logger for the alert bot.

Covers: schema creation + idempotency, duplicate-safe logging, actual-range
filling with error/breakout computation (incl. negative predictions), pending
vs completed queries, ordering and limits, per-asset/per-timeframe isolation,
and the calibration summary (HAR MAE vs lag-1 persistence MAE, bias, breakout
stats, regime counts). Every test uses ``tmp_path`` - the real database is
never touched, nothing needs the network, everything is deterministic.
"""
import re
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from kronos_trading.alerts.har_forecaster import HarForecast
from kronos_trading.alerts.prediction_logger import (
    BREAKOUT_THRESHOLD,
    DEFAULT_HISTORY_LIMIT,
    MIN_CALIBRATION_OBS,
    get_calibration_summary,
    get_pending_predictions,
    get_prediction_history,
    initialize_db,
    log_prediction,
    update_actual,
)

UTC = timezone.utc
T0 = datetime(2024, 1, 15, tzinfo=UTC)

EXPECTED_COLUMNS = {
    "id", "timestamp", "asset", "timeframe", "har_predicted_range",
    "coef_b0", "coef_b1", "coef_b2", "coef_b3", "n_obs", "regime",
    "actual_range", "prediction_error", "abs_prediction_error",
    "breakout_flag", "created_at",
}


def ts(i: int) -> str:
    """ISO8601 UTC timestamp: T0 + i hours, e.g. '2024-01-15T00:00:00Z'."""
    return (T0 + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")


def forecast(predicted: float, coefs=(1.0, 0.5, 0.3, 0.1), n_obs=178) -> HarForecast:
    return HarForecast(predicted_range=predicted, coefficients=coefs, n_obs=n_obs)


def query(db, sql, params=()):
    """Direct sqlite3 read (row factory on) - independent of the module code."""
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def count_rows(db, table="har_predictions"):
    return query(db, f"SELECT COUNT(*) AS c FROM {table}")[0]["c"]


# ---------------------------------------------------------------------------
# initialize_db
# ---------------------------------------------------------------------------

class TestInitializeDb:
    def test_initialize_db_creates_table(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        cols = {r["name"] for r in query(db, "PRAGMA table_info(har_predictions)")}
        assert cols == EXPECTED_COLUMNS
        assert len(cols) == 16  # exactly the spec'd schema, nothing extra

    def test_initialize_db_is_idempotent(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        initialize_db(db)  # no error, no duplicate table
        tables = query(
            db,
            "SELECT COUNT(*) AS c FROM sqlite_master "
            "WHERE type = 'table' AND name = 'har_predictions'",
        )
        assert tables[0]["c"] == 1
        # Data survives a re-init.
        log_prediction(db, ts(0), "BTC/USDT", "1h", forecast(100.0))
        initialize_db(db)
        assert count_rows(db) == 1


# ---------------------------------------------------------------------------
# log_prediction
# ---------------------------------------------------------------------------

class TestLogPrediction:
    def test_log_prediction_inserts_row(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        f = HarForecast(predicted_range=123.45,
                        coefficients=(1.1, 0.4, 0.2, 0.05), n_obs=178)
        rid = log_prediction(db, ts(0), "BTC/USDT", "1h", f, regime="high")
        row = query(db, "SELECT * FROM har_predictions WHERE id = ?", (rid,))[0]
        assert row["timestamp"] == ts(0)
        assert row["asset"] == "BTC/USDT"
        assert row["timeframe"] == "1h"
        assert row["har_predicted_range"] == pytest.approx(123.45)
        assert row["coef_b0"] == pytest.approx(1.1)
        assert row["coef_b1"] == pytest.approx(0.4)
        assert row["coef_b2"] == pytest.approx(0.2)
        assert row["coef_b3"] == pytest.approx(0.05)
        assert row["n_obs"] == 178
        assert row["regime"] == "high"
        # Unfilled outcome fields - prediction logged before the bar closes.
        assert row["actual_range"] is None
        assert row["prediction_error"] is None
        assert row["abs_prediction_error"] is None
        assert row["breakout_flag"] == 0
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", row["created_at"])

    def test_log_prediction_returns_row_id(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        rid1 = log_prediction(db, ts(0), "BTC/USDT", "1h", forecast(100.0))
        rid2 = log_prediction(db, ts(1), "BTC/USDT", "1h", forecast(101.0))
        assert isinstance(rid1, int) and rid1 > 0
        assert isinstance(rid2, int) and rid2 > rid1  # autoincrement

    def test_log_prediction_duplicate_is_idempotent(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        rid1 = log_prediction(db, ts(0), "BTC/USDT", "1h", forecast(100.0))
        # Same key, different values: must be skipped, original kept.
        rid2 = log_prediction(db, ts(0), "BTC/USDT", "1h", forecast(999.0))
        assert rid2 == rid1
        assert count_rows(db) == 1
        row = query(db, "SELECT har_predicted_range FROM har_predictions")[0]
        assert row["har_predicted_range"] == pytest.approx(100.0)

    def test_asset_and_timeframe_normalization(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        log_prediction(db, ts(0), "btc/usdt", "1H", forecast(100.0))
        # Normalized keys match across insert and lookup.
        assert update_actual(db, ts(0), "BTC/USDT", "1h", 101.0) is True
        hist = get_prediction_history(db, "BTC/USDT", "1h")
        assert len(hist) == 1
        assert hist[0]["asset"] == "BTC/USDT" and hist[0]["timeframe"] == "1h"


# ---------------------------------------------------------------------------
# update_actual
# ---------------------------------------------------------------------------

class TestUpdateActual:
    def test_update_actual_fills_error_fields(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        log_prediction(db, ts(0), "BTC/USDT", "1h", forecast(100.0))
        assert update_actual(db, ts(0), "BTC/USDT", "1h", 150.0) is True
        row = query(db, "SELECT * FROM har_predictions WHERE id = 1")[0]
        assert row["actual_range"] == pytest.approx(150.0)
        assert row["prediction_error"] == pytest.approx(50.0)
        assert row["abs_prediction_error"] == pytest.approx(50.0)
        assert row["breakout_flag"] == 0

    def test_update_actual_sets_breakout_flag_true(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        log_prediction(db, ts(0), "BTC/USDT", "1h", forecast(100.0))
        assert update_actual(db, ts(0), "BTC/USDT", "1h", 250.0) is True
        row = query(db, "SELECT breakout_flag, prediction_error "
                        "FROM har_predictions WHERE id = 1")[0]
        assert row["breakout_flag"] == 1
        assert row["prediction_error"] == pytest.approx(150.0)

    def test_update_actual_sets_breakout_flag_false(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        log_prediction(db, ts(0), "BTC/USDT", "1h", forecast(100.0))
        assert update_actual(db, ts(0), "BTC/USDT", "1h", 150.0) is True
        row = query(db, "SELECT breakout_flag FROM har_predictions WHERE id = 1")[0]
        assert row["breakout_flag"] == 0

    def test_update_actual_returns_false_if_not_found(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        assert update_actual(db, ts(0), "BTC/USDT", "1h", 150.0) is False
        assert count_rows(db) == 0  # nothing created

    def test_update_actual_handles_negative_prediction(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        # Negative predicted range is honest OLS output - see Step 1.
        log_prediction(db, ts(0), "BTC/USDT", "1h", forecast(-5.0))
        assert update_actual(db, ts(0), "BTC/USDT", "1h", 100.0) is True
        row = query(db, "SELECT * FROM har_predictions WHERE id = 1")[0]
        assert row["breakout_flag"] == 0      # never a breakout vs <=0 prediction
        assert row["prediction_error"] == pytest.approx(105.0)
        assert row["abs_prediction_error"] == pytest.approx(105.0)

    def test_update_actual_does_not_overwrite_completed_row(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        log_prediction(db, ts(0), "BTC/USDT", "1h", forecast(100.0))
        assert update_actual(db, ts(0), "BTC/USDT", "1h", 150.0) is True
        # Second call with a different value: first value must win.
        assert update_actual(db, ts(0), "BTC/USDT", "1h", 999.0) is True
        row = query(db, "SELECT actual_range, prediction_error "
                        "FROM har_predictions WHERE id = 1")[0]
        assert row["actual_range"] == pytest.approx(150.0)
        assert row["prediction_error"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# get_prediction_history / get_pending_predictions
# ---------------------------------------------------------------------------

class TestQueries:
    def test_get_prediction_history_returns_completed_only(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        for i in range(3):
            log_prediction(db, ts(i), "BTC/USDT", "1h", forecast(100.0 + i))
        update_actual(db, ts(0), "BTC/USDT", "1h", 101.0)
        update_actual(db, ts(2), "BTC/USDT", "1h", 103.0)
        hist = get_prediction_history(db, "BTC/USDT", "1h")
        assert len(hist) == 2
        assert [r["timestamp"] for r in hist] == [ts(2), ts(0)]  # DESC

    def test_get_prediction_history_respects_n_limit(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        for i in range(10):
            log_prediction(db, ts(i), "BTC/USDT", "1h", forecast(100.0 + i))
            update_actual(db, ts(i), "BTC/USDT", "1h", 101.0 + i)
        hist = get_prediction_history(db, "BTC/USDT", "1h", n=5)
        assert len(hist) == 5
        assert [r["timestamp"] for r in hist] == [ts(9), ts(8), ts(7), ts(6), ts(5)]

    def test_get_pending_predictions_returns_nulls(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        for i in range(3):
            log_prediction(db, ts(i), "BTC/USDT", "1h", forecast(100.0 + i))
        update_actual(db, ts(0), "BTC/USDT", "1h", 101.0)
        pending = get_pending_predictions(db, "BTC/USDT", "1h")
        assert len(pending) == 2
        assert [r["timestamp"] for r in pending] == [ts(1), ts(2)]  # ASC


# ---------------------------------------------------------------------------
# get_calibration_summary
# ---------------------------------------------------------------------------

class TestCalibrationSummary:
    def test_get_calibration_summary_returns_none_if_few(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        for i in range(MIN_CALIBRATION_OBS - 1):
            log_prediction(db, ts(i), "BTC/USDT", "1h", forecast(100.0 + i))
            update_actual(db, ts(i), "BTC/USDT", "1h", 101.0 + i)
        assert get_calibration_summary(db, "BTC/USDT", "1h") is None
        # Uncompleted rows must not count towards the minimum either.
        log_prediction(db, ts(99), "BTC/USDT", "1h", forecast(50.0))
        assert get_calibration_summary(db, "BTC/USDT", "1h") is None

    def test_get_calibration_summary_correct_mae(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        n = 25
        for i in range(n):
            log_prediction(db, ts(i), "BTC/USDT", "1h",
                           forecast(100.0 + i), regime="medium")
            update_actual(db, ts(i), "BTC/USDT", "1h", 101.0 + i)
        s = get_calibration_summary(db, "BTC/USDT", "1h")
        assert s is not None
        assert s["n_obs"] == n
        # Every error is exactly +1.0.
        assert s["har_mae"] == pytest.approx(1.0)
        assert s["mean_prediction_error"] == pytest.approx(1.0)  # bias
        # Persistence = previous row's HAR prediction: error is exactly +2.0,
        # and the first chronological row is skipped.
        assert s["persistence_mae"] == pytest.approx(2.0)
        assert s["har_beats_persistence"] is True
        assert s["breakout_count"] == 0
        assert s["breakout_rate"] == pytest.approx(0.0)
        assert s["regime_counts"] == {"low": 0, "medium": n, "high": 0}

    def test_get_calibration_summary_har_beats_persistence(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        n = MIN_CALIBRATION_OBS
        # HAR predicts perfectly (error 0); lag-1 persistence lags the trend
        # by exactly one step (error 5.0).
        for i in range(n):
            log_prediction(db, ts(i), "BTC/USDT", "1h", forecast(100.0 + 5 * i))
            update_actual(db, ts(i), "BTC/USDT", "1h", 100.0 + 5 * i)
        s = get_calibration_summary(db, "BTC/USDT", "1h")
        assert s is not None
        assert s["har_mae"] == pytest.approx(0.0)
        assert s["persistence_mae"] == pytest.approx(5.0)
        assert s["har_beats_persistence"] is True

    def test_get_calibration_summary_regime_counts(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        regimes = ["low"] * 10 + ["medium"] * 8 + ["high"] * 4 + [None] * 2
        for i, rg in enumerate(regimes):
            log_prediction(db, ts(i), "BTC/USDT", "1h", forecast(100.0 + i),
                           regime=rg)
            update_actual(db, ts(i), "BTC/USDT", "1h", 101.0 + i)
        s = get_calibration_summary(db, "BTC/USDT", "1h")
        assert s is not None
        assert s["n_obs"] == len(regimes) == 24
        # NULL regimes are excluded from the counts.
        assert s["regime_counts"] == {"low": 10, "medium": 8, "high": 4}


# ---------------------------------------------------------------------------
# isolation + full round trip
# ---------------------------------------------------------------------------

class TestIsolationAndRoundTrip:
    def test_multiple_assets_are_isolated(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        for i in range(3):
            log_prediction(db, ts(i), "BTC/USDT", "1h", forecast(100.0 + i))
            log_prediction(db, ts(i), "ETH/USDT", "1h", forecast(10.0 + i))
        update_actual(db, ts(0), "BTC/USDT", "1h", 101.0)
        update_actual(db, ts(1), "BTC/USDT", "1h", 102.0)
        update_actual(db, ts(0), "ETH/USDT", "1h", 11.0)
        btc_hist = get_prediction_history(db, "BTC/USDT", "1h")
        eth_hist = get_prediction_history(db, "ETH/USDT", "1h")
        assert len(btc_hist) == 2
        assert all(r["asset"] == "BTC/USDT" for r in btc_hist)
        assert len(eth_hist) == 1
        assert all(r["asset"] == "ETH/USDT" for r in eth_hist)
        btc_pending = get_pending_predictions(db, "BTC/USDT", "1h")
        assert [r["timestamp"] for r in btc_pending] == [ts(2)]

    def test_multiple_timeframes_are_isolated(self, tmp_path):
        db = str(tmp_path / "alerts.db")
        initialize_db(db)
        for i in range(3):
            log_prediction(db, ts(i), "BTC/USDT", "1h", forecast(100.0 + i))
            log_prediction(db, ts(i), "BTC/USDT", "4h", forecast(400.0 + i))
            update_actual(db, ts(i), "BTC/USDT", "1h", 101.0 + i)
            update_actual(db, ts(i), "BTC/USDT", "4h", 401.0 + i)
        h1 = get_prediction_history(db, "BTC/USDT", "1h")
        h4 = get_prediction_history(db, "BTC/USDT", "4h")
        assert len(h1) == 3 and all(r["timeframe"] == "1h" for r in h1)
        assert len(h4) == 3 and all(r["timeframe"] == "4h" for r in h4)
        pending_1h = get_pending_predictions(db, "BTC/USDT", "1h")
        pending_4h = get_pending_predictions(db, "BTC/USDT", "4h")
        assert pending_1h == [] and pending_4h == []

    def test_full_round_trip(self, tmp_path):
        db = str(tmp_path / "roundtrip.db")
        initialize_db(db)
        n = 25
        predicted = [100.0 + i for i in range(n)]
        actual = [p + 1.0 for p in predicted]
        actual[10] = 300.0  # vs predicted[10] = 110 -> 300 > 2x110: breakout
        regimes = ["low", "medium", "high"]

        # 1) Log predictions BEFORE bars close (no actuals yet).
        for i in range(n):
            log_prediction(db, ts(i), "BTC/USDT", "1h", forecast(predicted[i]),
                           regime=regimes[i % 3])
        log_prediction(db, ts(0), "ETH/USDT", "1h", forecast(50.0))
        assert len(get_pending_predictions(db, "BTC/USDT", "1h")) == n
        assert len(get_pending_predictions(db, "ETH/USDT", "1h")) == 1
        assert get_prediction_history(db, "BTC/USDT", "1h") == []

        # 2) Fill actuals AFTER bars close.
        for i in range(n):
            assert update_actual(db, ts(i), "BTC/USDT", "1h", actual[i]) is True
        assert update_actual(db, ts(0), "ETH/USDT", "1h", 60.0) is True

        # 3) History: completed only, newest first.
        hist = get_prediction_history(db, "BTC/USDT", "1h")
        assert len(hist) == n
        assert hist[0]["timestamp"] == ts(n - 1)
        assert len(get_pending_predictions(db, "BTC/USDT", "1h")) == 0

        # 4) Calibration summary consistent with manual computation.
        s = get_calibration_summary(db, "BTC/USDT", "1h")
        assert s is not None
        assert s["n_obs"] == n
        assert s["har_mae"] == pytest.approx(
            sum(abs(a - p) for a, p in zip(actual, predicted)) / n)
        assert s["mean_prediction_error"] == pytest.approx(
            sum(a - p for a, p in zip(actual, predicted)) / n)
        pers_errs = [actual[i] - predicted[i - 1] for i in range(1, n)]
        assert s["persistence_mae"] == pytest.approx(
            sum(abs(e) for e in pers_errs) / len(pers_errs))
        assert s["breakout_count"] == 1
        assert s["breakout_rate"] == pytest.approx(1.0 / n)
        expected_counts = {"low": 0, "medium": 0, "high": 0}
        for i in range(n):
            expected_counts[regimes[i % 3]] += 1
        assert s["regime_counts"] == expected_counts

        # 5) Assets stay isolated end-to-end.
        eth_hist = get_prediction_history(db, "ETH/USDT", "1h")
        assert len(eth_hist) == 1
        assert eth_hist[0]["actual_range"] == pytest.approx(60.0)
        assert eth_hist[0]["prediction_error"] == pytest.approx(10.0)
