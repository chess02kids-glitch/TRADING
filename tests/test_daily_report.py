"""Tests for the HAR daily status report (no network or database access)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from kronos_trading.alerts.breakout_detector import LiveCalibration
from kronos_trading.alerts.telegram_sender import (
    AssetDailyStats,
    SendResult,
    TelegramConfig,
    send_daily_report,
)
from scripts import run_daily_report

SENDER_MODULE = "kronos_trading.alerts.telegram_sender"
SCRIPT_MODULE = "scripts.run_daily_report"
CONFIG = TelegramConfig("test-token", "test-chat")
SUCCESS = SendResult(True, 42, None, 1)


def make_live_calibration(
    n_obs=100,
    har_mae=423.45,
    persistence_mae=487.23,
    har_beats=True,
    mean_bias=12.3,
    breakout_count=4,
    breakout_rate=0.04,
    worst_ratio=3.45,
    best_ratio=0.8,
    recent_mae_7d=398.12,
    recent_mae_30d=423.45,
    is_degrading=False,
) -> LiveCalibration:
    """Create a complete calibration object with overridable fields."""
    return LiveCalibration(
        n_obs=n_obs,
        har_mae=har_mae,
        persistence_mae=persistence_mae,
        har_beats_persistence=har_beats,
        mean_bias=mean_bias,
        breakout_count=breakout_count,
        breakout_rate=breakout_rate,
        worst_ratio=worst_ratio,
        best_ratio=best_ratio,
        recent_mae_7d=recent_mae_7d,
        recent_mae_30d=recent_mae_30d,
        is_degrading=is_degrading,
    )


def make_stats(asset="BTC/USDT", calibration=None, **kwargs):
    """Create one market's report statistics."""
    values = {
        "asset": asset,
        "timeframe": "1h",
        "total_predictions": 101,
        "completed": 100,
        "pending": 1,
        "calibration": calibration,
        "days_running": 7,
    }
    values.update(kwargs)
    return AssetDailyStats(**values)


def sent_text(mock_send):
    return mock_send.call_args.args[1]


# AssetDailyStats

def test_asset_daily_stats_creation():
    cal = make_live_calibration()
    stats = make_stats(calibration=cal)
    assert stats.asset == "BTC/USDT"
    assert stats.timeframe == "1h"
    assert stats.total_predictions == 101
    assert stats.completed == 100
    assert stats.pending == 1
    assert stats.calibration is cal
    assert stats.days_running == 7


def test_asset_daily_stats_no_calibration():
    assert make_stats(calibration=None).calibration is None


# send_daily_report

def test_send_daily_report_empty_stats():
    with patch(f"{SENDER_MODULE}.send_message", return_value=SUCCESS) as send:
        result = send_daily_report(CONFIG, [], 1)
    assert "No prediction data yet" in sent_text(send)
    assert isinstance(result, SendResult)


def test_send_daily_report_no_calibration_data():
    stats = make_stats(
        calibration=None, total_predictions=5, completed=3, pending=2
    )
    with patch(f"{SENDER_MODULE}.send_message", return_value=SUCCESS) as send:
        send_daily_report(CONFIG, [stats], 1)
    text = sent_text(send)
    assert "BTC/USDT" in text
    assert "Predictions logged: 5" in text
    assert "insufficient data" in text


def test_send_daily_report_with_calibration():
    with patch(f"{SENDER_MODULE}.send_message", return_value=SUCCESS) as send:
        send_daily_report(CONFIG, [make_stats(calibration=make_live_calibration())], 7)
    text = sent_text(send)
    assert "HAR MAE" in text
    assert "✅ YES" in text
    assert "BTC/USDT" in text


def test_send_daily_report_two_assets():
    stats = [
        make_stats("BTC/USDT", make_live_calibration()),
        make_stats("ETH/USDT", make_live_calibration()),
    ]
    with patch(f"{SENDER_MODULE}.send_message", return_value=SUCCESS) as send:
        send_daily_report(CONFIG, stats, 7)
    assert "BTC/USDT" in sent_text(send)
    assert "ETH/USDT" in sent_text(send)
    send.assert_called_once()


def test_send_daily_report_degrading_model():
    cal = make_live_calibration(is_degrading=True)
    with patch(f"{SENDER_MODULE}.send_message", return_value=SUCCESS) as send:
        send_daily_report(CONFIG, [make_stats(calibration=cal)], 7)
    assert "⚠️ YES" in sent_text(send)
    assert "DEGRADING" in sent_text(send)


def test_send_daily_report_not_beating_persistence():
    cal = make_live_calibration(har_beats=False)
    with patch(f"{SENDER_MODULE}.send_message", return_value=SUCCESS) as send:
        send_daily_report(CONFIG, [make_stats(calibration=cal)], 7)
    assert "❌ NO" in sent_text(send)
    assert "HAR not beating persistence" in sent_text(send)


def test_send_daily_report_all_beating():
    stats = [
        make_stats("BTC/USDT", make_live_calibration()),
        make_stats("ETH/USDT", make_live_calibration()),
    ]
    with patch(f"{SENDER_MODULE}.send_message", return_value=SUCCESS) as send:
        send_daily_report(CONFIG, stats, 7)
    assert "Overall Status: ✅ ON TRACK" in sent_text(send)


def test_send_daily_report_calibration_day():
    with patch(f"{SENDER_MODULE}.send_message", return_value=SUCCESS) as send:
        send_daily_report(CONFIG, [make_stats()], 7, 30)
    assert "Day 7 of 30" in sent_text(send)
    assert "Days remaining: 23" in sent_text(send)


def test_send_daily_report_returns_sendresult():
    with patch(f"{SENDER_MODULE}.send_message", return_value=SUCCESS):
        result = send_daily_report(CONFIG, [make_stats()], 1)
    assert isinstance(result, SendResult)


def test_send_daily_report_never_raises():
    with patch(f"{SENDER_MODULE}.send_message", side_effect=RuntimeError("boom")):
        result = send_daily_report(CONFIG, [make_stats()], 1)
    assert result == SendResult(False, None, "boom", 0)


def test_send_daily_report_breakout_stats():
    cal = make_live_calibration(breakout_count=4, breakout_rate=0.024)
    with patch(f"{SENDER_MODULE}.send_message", return_value=SUCCESS) as send:
        send_daily_report(CONFIG, [make_stats(calibration=cal)], 7)
    assert "Breakouts: 4 (2.4%)" in sent_text(send)


def test_send_daily_report_7d_mae_shown():
    cal = make_live_calibration(recent_mae_7d=398.12)
    with patch(f"{SENDER_MODULE}.send_message", return_value=SUCCESS) as send:
        send_daily_report(CONFIG, [make_stats(calibration=cal)], 7)
    assert "398.12" in sent_text(send)


def test_send_daily_report_7d_mae_none():
    cal = make_live_calibration(recent_mae_7d=None)
    with patch(f"{SENDER_MODULE}.send_message", return_value=SUCCESS) as send:
        send_daily_report(CONFIG, [make_stats(calibration=cal)], 7)
    assert "N/A" in sent_text(send)


def test_send_daily_report_worst_ratio_shown():
    cal = make_live_calibration(worst_ratio=3.45)
    with patch(f"{SENDER_MODULE}.send_message", return_value=SUCCESS) as send:
        send_daily_report(CONFIG, [make_stats(calibration=cal)], 7)
    assert "3.45×" in sent_text(send)


def test_send_daily_report_positive_bias():
    cal = make_live_calibration(mean_bias=12.3)
    with patch(f"{SENDER_MODULE}.send_message", return_value=SUCCESS) as send:
        send_daily_report(CONFIG, [make_stats(calibration=cal)], 7)
    assert "+12.3" in sent_text(send)
    assert "overestimate" in sent_text(send)


def test_send_daily_report_negative_bias():
    cal = make_live_calibration(mean_bias=-0.8)
    with patch(f"{SENDER_MODULE}.send_message", return_value=SUCCESS) as send:
        send_daily_report(CONFIG, [make_stats(calibration=cal)], 7)
    assert "-0.8" in sent_text(send)
    assert "underestimate" in sent_text(send)


# compute_calibration_day

def test_calibration_day_no_history():
    with patch(f"{SCRIPT_MODULE}.get_prediction_history", return_value=[]):
        assert run_daily_report.compute_calibration_day("unused", "BTC/USDT", "1h") == 1


def test_calibration_day_one_day_ago():
    timestamp = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    with patch(
        f"{SCRIPT_MODULE}.get_prediction_history",
        return_value=[{"timestamp": timestamp}],
    ):
        assert run_daily_report.compute_calibration_day("unused", "BTC/USDT", "1h") == 2


def test_calibration_day_seven_days_ago():
    timestamp = (datetime.now(timezone.utc) - timedelta(hours=168)).isoformat()
    with patch(
        f"{SCRIPT_MODULE}.get_prediction_history",
        return_value=[{"timestamp": timestamp}],
    ):
        day = run_daily_report.compute_calibration_day("unused", "BTC/USDT", "1h")
    assert day in (7, 8)


# build_asset_stats

def test_build_asset_stats_empty_db():
    with patch(f"{SCRIPT_MODULE}.get_prediction_history", return_value=[]), patch(
        f"{SCRIPT_MODULE}.get_pending_predictions", return_value=[]
    ):
        stats = run_daily_report.build_asset_stats("unused", "BTC/USDT", "1h", 1)
    assert stats.total_predictions == stats.completed == stats.pending == 0
    assert stats.calibration is None


def test_build_asset_stats_with_data():
    history = [
        {
            "actual_range": float(i + 2),
            "har_predicted_range": float(i + 1),
            "breakout_flag": 0,
        }
        for i in range(30)
    ]
    with patch(f"{SCRIPT_MODULE}.get_prediction_history", return_value=history), patch(
        f"{SCRIPT_MODULE}.get_pending_predictions", return_value=[]
    ):
        stats = run_daily_report.build_asset_stats("unused", "BTC/USDT", "1h", 2)
    assert stats.total_predictions == stats.completed == 30
    assert stats.pending == 0
    assert stats.calibration is not None


def test_build_asset_stats_never_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    stats = run_daily_report.build_asset_stats(
        str(tmp_path / "missing" / "db.sqlite"), "BTC/USDT", "1h", 1
    )
    assert stats == AssetDailyStats("BTC/USDT", "1h", 0, 0, 0, None, 1)


# main

def test_main_no_telegram_env(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with patch(f"{SCRIPT_MODULE}.load_dotenv"), patch(
        f"{SCRIPT_MODULE}.TelegramConfig.from_env",
        side_effect=EnvironmentError("missing"),
    ):
        assert run_daily_report.main() == 1


def test_main_success():
    empty = make_stats(calibration=None, total_predictions=0, completed=0, pending=0)
    with patch(f"{SCRIPT_MODULE}.load_dotenv"), patch(
        f"{SCRIPT_MODULE}.TelegramConfig.from_env", return_value=CONFIG
    ), patch(f"{SCRIPT_MODULE}.initialize_db"), patch(
        f"{SCRIPT_MODULE}.compute_calibration_day", return_value=1
    ), patch(f"{SCRIPT_MODULE}.build_asset_stats", return_value=empty), patch(
        f"{SCRIPT_MODULE}.send_daily_report", return_value=SUCCESS
    ) as send:
        assert run_daily_report.main() == 0
    send.assert_called_once()


def test_main_send_failure_returns_nonzero():
    failure = SendResult(False, None, "Telegram unavailable", 1)
    empty = make_stats(calibration=None, total_predictions=0, completed=0, pending=0)
    with patch(f"{SCRIPT_MODULE}.load_dotenv"), patch(
        f"{SCRIPT_MODULE}.TelegramConfig.from_env", return_value=CONFIG
    ), patch(f"{SCRIPT_MODULE}.initialize_db"), patch(
        f"{SCRIPT_MODULE}.compute_calibration_day", return_value=1
    ), patch(f"{SCRIPT_MODULE}.build_asset_stats", return_value=empty), patch(
        f"{SCRIPT_MODULE}.send_daily_report", return_value=failure
    ):
        assert run_daily_report.main() == 1
