"""Step 5 tests - scheduler (all external calls mocked, tmp_path DBs only).

Covers: next-bar timestamp alignment (1h/4h/1d), sleep-until-next-cycle
arithmetic (incl. the 1.0 s floor), the full single-cycle pipeline (both
assets, DB logging, per-asset failure isolation, pending-prediction fill,
breakout alerts, no-look-ahead timestamps), calibration-cycle gating, and the
run_forever loop (startup, clean shutdown on KeyboardInterrupt, survival after
cycle errors, DB init ordering).

No network, no real Telegram, no real CCXT: ``fetch_candles`` and every
``send_*``/``time.sleep`` call is patched; the DB is a tmp_path file.
"""
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from kronos_trading.alerts.har_forecaster import HarForecast, predict_next_range
from kronos_trading.alerts.prediction_logger import (
    DEFAULT_DB_PATH,
    get_pending_predictions,
    get_prediction_history,
    log_prediction,
    update_actual,
)
from kronos_trading.alerts.scheduler import (
    CycleResult,
    SchedulerConfig,
    next_bar_timestamp,
    run_calibration_cycle,
    run_forever,
    run_single_cycle,
    seconds_until_next_cycle,
)
from kronos_trading.alerts.telegram_sender import SendResult, TelegramConfig
from kronos_trading.types import Candle

UTC = timezone.utc
BASE = datetime(2024, 1, 15, tzinfo=UTC)
MOD = "kronos_trading.alerts.scheduler"


def ts_str(hour):
    """ISO8601 UTC: 2024-01-15T{hour}:00:00Z."""
    return (BASE + timedelta(hours=hour)).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_fake_candles(n=100, base_close=50000.0, base_range=500.0,
                      start_ts=None, step_ms=3_600_000):
    """Deterministic Candle list - same inputs always give the same output."""
    if start_ts is None:
        start_ts = int(BASE.timestamp() * 1000)
    candles = []
    for i in range(n):
        close = base_close + i * 10.0
        r = base_range + (i % 5) * 10.0
        candles.append(Candle(start_ts + i * step_ms, close - r / 2,
                              close + r / 2, close - r / 2, close, 10.0))
    return candles


def make_scheduler_config(tmp_path):
    """SchedulerConfig pointing at a tmp_path DB (test isolation)."""
    return SchedulerConfig(
        assets=["BTC/USDT", "ETH/USDT"],
        timeframe="1h",
        candle_delay_seconds=0.0,
        n_candles=100,
        breakout_threshold=2.0,
        calibration_interval_hours=24,
        db_path=str(tmp_path / "test.db"),
    )


def make_telegram_config():
    return TelegramConfig(bot_token="test_token", chat_id="test_chat_id")


def make_forecast(predicted=100.0):
    return HarForecast(predicted, (1.0, 0.5, 0.3, 0.1), 178)


def ok_send():
    return SendResult(success=True, message_id=1, error=None, attempts=1)


def ok_cycle():
    return CycleResult(timestamp=ts_str(15), success=True, assets_processed=[],
                       errors=[], forecasts={}, breakouts={}, send_results={},
                       duration_seconds=0.01)


# ---------------------------------------------------------------------------
# next_bar_timestamp
# ---------------------------------------------------------------------------

class TestNextBarTimestamp:
    def test_next_bar_1h_on_the_hour(self):
        assert next_bar_timestamp(datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC),
                                  "1h") == "2024-01-15T15:00:00Z"

    def test_next_bar_1h_mid_hour(self):
        assert next_bar_timestamp(datetime(2024, 1, 15, 14, 7, 22, tzinfo=UTC),
                                  "1h") == "2024-01-15T15:00:00Z"

    def test_next_bar_1h_one_second_before(self):
        assert next_bar_timestamp(datetime(2024, 1, 15, 14, 59, 59, tzinfo=UTC),
                                  "1h") == "2024-01-15T15:00:00Z"

    def test_next_bar_4h_alignment(self):
        assert next_bar_timestamp(datetime(2024, 1, 15, 14, 7, tzinfo=UTC),
                                  "4h") == "2024-01-15T16:00:00Z"
        assert next_bar_timestamp(datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
                                  "4h") == "2024-01-15T16:00:00Z"

    def test_next_bar_1d_alignment(self):
        assert next_bar_timestamp(datetime(2024, 1, 15, 14, 7, tzinfo=UTC),
                                  "1d") == "2024-01-16T00:00:00Z"

    def test_next_bar_naive_input_treated_as_utc(self):
        assert next_bar_timestamp(datetime(2024, 1, 15, 14, 7, 22),
                                  "1h") == "2024-01-15T15:00:00Z"


# ---------------------------------------------------------------------------
# seconds_until_next_cycle
# ---------------------------------------------------------------------------

class TestSecondsUntilNextCycle:
    def test_sleep_is_positive(self):
        assert seconds_until_next_cycle(
            datetime(2024, 1, 15, 14, 7, 22, tzinfo=UTC), "1h") > 0

    def test_sleep_minimum_one_second(self):
        # Even with zero delay right at a boundary, never return 0/negative.
        assert seconds_until_next_cycle(
            datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC), "1h",
            delay_seconds=0.0) >= 1.0

    def test_sleep_with_delay(self):
        # 14:00:30 + 30s delay -> next hour is 15:00:00 -> 3570 + 30 = 3600.
        secs = seconds_until_next_cycle(
            datetime(2024, 1, 15, 14, 0, 30, tzinfo=UTC), "1h", delay_seconds=30.0)
        assert secs == pytest.approx(3600.0, abs=1.0)

    def test_sleep_mid_hour(self):
        # 14:07:22 -> 15:00:00 is 52m38s = 3158s; +30 = 3188.
        secs = seconds_until_next_cycle(
            datetime(2024, 1, 15, 14, 7, 22, tzinfo=UTC), "1h", delay_seconds=30.0)
        assert secs == pytest.approx(3188.0, abs=1.0)


# ---------------------------------------------------------------------------
# run_single_cycle
# ---------------------------------------------------------------------------

class TestRunSingleCycle:
    def _cycle_now(self):
        return BASE + timedelta(hours=15, seconds=30)  # 2024-01-15T15:00:30Z

    def test_cycle_returns_cycle_result(self, tmp_path):
        cfg = make_scheduler_config(tmp_path)
        with patch(f"{MOD}.fetch_candles", return_value=make_fake_candles()), \
             patch(f"{MOD}.send_forecast", return_value=ok_send()):
            result = run_single_cycle(make_telegram_config(), cfg,
                                      now=self._cycle_now())
        assert isinstance(result, CycleResult)
        assert result.timestamp == ts_str(15)

    def test_cycle_processes_both_assets(self, tmp_path):
        cfg = make_scheduler_config(tmp_path)
        with patch(f"{MOD}.fetch_candles", return_value=make_fake_candles()), \
             patch(f"{MOD}.send_forecast", return_value=ok_send()):
            result = run_single_cycle(make_telegram_config(), cfg,
                                      now=self._cycle_now())
        assert result.assets_processed == ["BTC/USDT", "ETH/USDT"]
        assert set(result.forecasts) == {"BTC/USDT", "ETH/USDT"}
        assert result.success is True

    def test_cycle_logs_prediction_to_db(self, tmp_path):
        cfg = make_scheduler_config(tmp_path)
        candles = make_fake_candles()
        expected = predict_next_range(candles)  # deterministic
        with patch(f"{MOD}.fetch_candles", return_value=candles), \
             patch(f"{MOD}.send_forecast", return_value=ok_send()):
            run_single_cycle(make_telegram_config(), cfg, now=self._cycle_now())
        pending = get_pending_predictions(cfg.db_path, "BTC/USDT", "1h")
        assert len(pending) == 1
        row = pending[0]
        assert row["timestamp"] == ts_str(15)
        assert row["actual_range"] is None          # logged BEFORE close
        assert row["har_predicted_range"] == pytest.approx(
            expected.predicted_range)
        assert row["coef_b0"] == pytest.approx(expected.coefficients[0])
        assert row["coef_b1"] == pytest.approx(expected.coefficients[1])
        assert row["coef_b2"] == pytest.approx(expected.coefficients[2])
        assert row["coef_b3"] == pytest.approx(expected.coefficients[3])
        assert row["regime"] in ("low", "medium", "high")  # 100 candles -> defined
        assert len(get_pending_predictions(cfg.db_path, "ETH/USDT", "1h")) == 1

    def test_cycle_fetch_failure_skips_asset(self, tmp_path):
        cfg = make_scheduler_config(tmp_path)

        def flaky(symbol, timeframe="1h", **kwargs):
            if symbol == "BTC/USDT":
                raise RuntimeError("network down")
            return make_fake_candles()

        with patch(f"{MOD}.fetch_candles", side_effect=flaky), \
             patch(f"{MOD}.send_forecast", return_value=ok_send()) as sf:
            result = run_single_cycle(make_telegram_config(), cfg,
                                      now=self._cycle_now())
        assert result.assets_processed == ["ETH/USDT"]
        assert any("BTC/USDT" in e and "fetch failed" in e
                   for e in result.errors)
        assert result.success is False
        assert "ETH/USDT" in result.forecasts and "BTC/USDT" not in result.forecasts
        sf.assert_not_called()  # combined message needs both assets

    def test_cycle_predict_failure_skips_asset(self, tmp_path):
        cfg = make_scheduler_config(tmp_path)
        candles = make_fake_candles()
        with patch(f"{MOD}.fetch_candles", return_value=candles), \
             patch(f"{MOD}.predict_next_range",
                   side_effect=[RuntimeError("fit failed"),
                                HarForecast(120.0, (1.0, 0.5, 0.3, 0.1), 178)]), \
             patch(f"{MOD}.send_forecast", return_value=ok_send()) as sf:
            result = run_single_cycle(make_telegram_config(), cfg,
                                      now=self._cycle_now())
        assert any("BTC/USDT" in e and "prediction failed" in e
                   for e in result.errors)
        assert "BTC/USDT" not in result.forecasts
        assert "ETH/USDT" in result.forecasts  # RULE 1: ETH still processed
        sf.assert_not_called()

    def test_cycle_sends_forecast_message(self, tmp_path):
        cfg = make_scheduler_config(tmp_path)
        with patch(f"{MOD}.fetch_candles", return_value=make_fake_candles()), \
             patch(f"{MOD}.send_forecast", return_value=ok_send()) as sf:
            result = run_single_cycle(make_telegram_config(), cfg,
                                      now=self._cycle_now())
        sf.assert_called_once()
        assert result.send_results["forecast"].success is True

    def test_cycle_duration_recorded(self, tmp_path):
        cfg = make_scheduler_config(tmp_path)
        with patch(f"{MOD}.fetch_candles", return_value=make_fake_candles()), \
             patch(f"{MOD}.send_forecast", return_value=ok_send()):
            result = run_single_cycle(make_telegram_config(), cfg,
                                      now=self._cycle_now())
        assert result.duration_seconds > 0.0

    def test_cycle_fills_pending_predictions(self, tmp_path):
        cfg = make_scheduler_config(tmp_path)
        # Pending prediction for the 14:00 bar (logged in a previous run).
        log_prediction(cfg.db_path, ts_str(14), "BTC/USDT", "1h",
                       make_forecast(540.0))
        # Fresh fetch covers the 14:00 candle (index 4 -> range 540.0).
        candles = make_fake_candles(
            start_ts=int((BASE + timedelta(hours=10)).timestamp() * 1000))
        with patch(f"{MOD}.fetch_candles", return_value=candles), \
             patch(f"{MOD}.send_forecast", return_value=ok_send()), \
             patch(f"{MOD}.send_breakout") as sb:
            result = run_single_cycle(make_telegram_config(), cfg,
                                      now=self._cycle_now())
        hist = get_prediction_history(cfg.db_path, "BTC/USDT", "1h")
        row = next(r for r in hist if r["timestamp"] == ts_str(14))
        assert row["actual_range"] == pytest.approx(540.0)
        assert row["prediction_error"] == pytest.approx(0.0)
        assert row["breakout_flag"] == 0          # 540 not > 2*540
        sb.assert_not_called()                     # no breakout -> no alert
        assert result.success is True

    def test_cycle_sends_breakout_when_detected(self, tmp_path):
        cfg = make_scheduler_config(tmp_path)
        log_prediction(cfg.db_path, ts_str(14), "BTC/USDT", "1h",
                       make_forecast(100.0))
        candles = make_fake_candles(
            start_ts=int((BASE + timedelta(hours=10)).timestamp() * 1000))
        # Replace the 14:00 candle with one whose range is exactly 300 (3x).
        target_ts = int((BASE + timedelta(hours=14)).timestamp() * 1000)
        candles = [
            Candle(target_ts, 100000.0, 100150.0, 99850.0, 100000.0, 10.0)
            if c.timestamp_ms == target_ts else c for c in candles
        ]
        with patch(f"{MOD}.fetch_candles", return_value=candles), \
             patch(f"{MOD}.send_forecast", return_value=ok_send()), \
             patch(f"{MOD}.send_breakout", return_value=ok_send()) as sb:
            result = run_single_cycle(make_telegram_config(), cfg,
                                      now=self._cycle_now())
        sb.assert_called_once()
        breakout = sb.call_args.args[3]  # (config, asset, tf, result, ts)
        assert breakout.is_breakout is True
        assert breakout.ratio == pytest.approx(3.0)
        assert result.breakouts["BTC/USDT"] is breakout
        hist = get_prediction_history(cfg.db_path, "BTC/USDT", "1h")
        row = next(r for r in hist if r["timestamp"] == ts_str(14))
        assert row["breakout_flag"] == 1

    def test_no_lookahead_prediction_timestamp(self, tmp_path):
        cfg = make_scheduler_config(tmp_path)
        with patch(f"{MOD}.fetch_candles", return_value=make_fake_candles()), \
             patch(f"{MOD}.send_forecast", return_value=ok_send()):
            run_single_cycle(make_telegram_config(), cfg, now=self._cycle_now())
        rows = get_pending_predictions(cfg.db_path, "BTC/USDT", "1h")
        # Prediction is for the NEXT bar (15:00, currently forming), never for
        # the just-completed bar (14:00) and never evaluated in the same cycle.
        assert [r["timestamp"] for r in rows] == [ts_str(15)]
        assert ts_str(15) > ts_str(14)  # strictly after the last closed bar
        assert all(r["actual_range"] is None for r in rows)


# ---------------------------------------------------------------------------
# run_calibration_cycle
# ---------------------------------------------------------------------------

class TestRunCalibrationCycle:
    def test_calibration_skipped_if_no_history(self, tmp_path):
        cfg = make_scheduler_config(tmp_path)
        with patch(f"{MOD}.send_calibration_report") as sc:
            run_calibration_cycle(make_telegram_config(), cfg)
        sc.assert_not_called()

    def test_calibration_sends_if_history_exists(self, tmp_path):
        cfg = make_scheduler_config(tmp_path)
        for asset in ("BTC/USDT", "ETH/USDT"):
            for i in range(30):
                log_prediction(cfg.db_path, ts_str(i), asset, "1h",
                               make_forecast(100.0 + i))
                update_actual(cfg.db_path, ts_str(i), asset, "1h", 101.0 + i)
        with patch(f"{MOD}.send_calibration_report") as sc:
            run_calibration_cycle(make_telegram_config(), cfg)
        assert sc.call_count == 2
        assert [c.args[1] for c in sc.call_args_list] == ["BTC/USDT", "ETH/USDT"]


# ---------------------------------------------------------------------------
# run_forever
# ---------------------------------------------------------------------------

class TestRunForever:
    def test_run_forever_sends_startup(self, tmp_path):
        cfg = make_scheduler_config(tmp_path)
        with patch(f"{MOD}.run_single_cycle", return_value=ok_cycle()) as rc, \
             patch(f"{MOD}.time.sleep", side_effect=KeyboardInterrupt), \
             patch(f"{MOD}.send_startup_message", return_value=ok_send()) as su, \
             patch(f"{MOD}.send_shutdown_message", return_value=ok_send()) as sd, \
             patch(f"{MOD}.run_calibration_cycle"):
            run_forever(make_telegram_config(), cfg)
        su.assert_called_once()
        rc.assert_called_once()
        sd.assert_called_once()

    def test_run_forever_sends_shutdown_on_interrupt(self, tmp_path):
        cfg = make_scheduler_config(tmp_path)
        with patch(f"{MOD}.run_single_cycle", return_value=ok_cycle()), \
             patch(f"{MOD}.time.sleep", side_effect=KeyboardInterrupt), \
             patch(f"{MOD}.send_startup_message", return_value=ok_send()), \
             patch(f"{MOD}.send_shutdown_message", return_value=ok_send()) as sd, \
             patch(f"{MOD}.run_calibration_cycle"):
            run_forever(make_telegram_config(), cfg)
        sd.assert_called_once()

    def test_run_forever_continues_after_cycle_error(self, tmp_path):
        cfg = make_scheduler_config(tmp_path)
        with patch(f"{MOD}.run_single_cycle",
                   side_effect=[RuntimeError("boom"), KeyboardInterrupt]) as rc, \
             patch(f"{MOD}.time.sleep"), \
             patch(f"{MOD}.send_startup_message", return_value=ok_send()), \
             patch(f"{MOD}.send_shutdown_message", return_value=ok_send()) as sd, \
             patch(f"{MOD}.run_calibration_cycle"):
            run_forever(make_telegram_config(), cfg)  # must not raise
        assert rc.call_count == 2  # first cycle errored, loop continued
        sd.assert_called_once()

    def test_run_forever_initializes_db(self, tmp_path):
        cfg = make_scheduler_config(tmp_path)

        def first_cycle(*args, **kwargs):
            assert initialize_db.called, "initialize_db must run before first cycle"
            raise KeyboardInterrupt

        with patch(f"{MOD}.initialize_db") as initialize_db, \
             patch(f"{MOD}.run_single_cycle", side_effect=first_cycle), \
             patch(f"{MOD}.time.sleep"), \
             patch(f"{MOD}.send_startup_message", return_value=ok_send()), \
             patch(f"{MOD}.send_shutdown_message", return_value=ok_send()), \
             patch(f"{MOD}.run_calibration_cycle"):
            run_forever(make_telegram_config(), cfg)
        initialize_db.assert_called_once()


# ---------------------------------------------------------------------------
# SchedulerConfig
# ---------------------------------------------------------------------------

class TestSchedulerConfig:
    def test_scheduler_config_default(self):
        cfg = SchedulerConfig.default()
        assert cfg.assets == ["BTC/USDT", "ETH/USDT"]
        assert cfg.timeframe == "1h"
        assert cfg.candle_delay_seconds == 30.0
        assert cfg.n_candles == 800
        assert cfg.breakout_threshold == 2.0
        assert cfg.calibration_interval_hours == 24
        assert cfg.db_path == DEFAULT_DB_PATH
