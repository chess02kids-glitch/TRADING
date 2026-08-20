"""Step 4 tests - Telegram sender (all API calls mocked, no network, no .env).

Covers: successful sends and message_id extraction, empty-text skipping, no
retry on API errors, retry-on-network-error semantics (exhaustion, recovery,
timeouts), never-raises behavior, token/chat-id never logged, correct URL /
payload / timeout, forecast message content (incl. regime N/A), breakout
skip/send, calibration report, ``TelegramConfig.from_env`` (missing-credential
errors via monkeypatch, real .env never read), and startup/shutdown messages.

The ``make_mock_response`` helper builds a fake ``requests.Response``; every
HTTP call is patched with ``unittest.mock``. Tests are deterministic and run
with ``retry_delay=0`` so they stay fast.
"""
import logging
from unittest.mock import MagicMock, patch

import pytest
import requests

from kronos_trading.alerts.breakout_detector import (
    LiveCalibration,
    check_breakout,
)
from kronos_trading.alerts.har_forecaster import HarForecast
from kronos_trading.alerts.telegram_sender import (
    API_BASE_URL,
    SEND_MESSAGE_METHOD,
    TIMEOUT_SECONDS,
    SendResult,
    TelegramConfig,
    send_breakout,
    send_calibration_report,
    send_forecast,
    send_message,
    send_shutdown_message,
    send_startup_message,
)

TOKEN = "test_token_123"
CHAT_ID = "test_chat_456"
MODULE = "kronos_trading.alerts.telegram_sender"


def make_mock_response(ok=True, message_id=12345, description=None):
    """Build a mock requests.Response for Telegram API responses."""
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    if ok:
        mock.json.return_value = {
            "ok": True,
            "result": {"message_id": message_id}
        }
    else:
        mock.json.return_value = {
            "ok": False,
            "description": description or "Bad Request"
        }
    return mock


def make_config():
    return TelegramConfig(bot_token=TOKEN, chat_id=CHAT_ID)


def make_forecast(predicted=100.0, regime=None):
    return HarForecast(predicted_range=predicted,
                       coefficients=(1.0, 0.5, 0.3, 0.1), n_obs=178,
                       regime=regime)


def make_calibration():
    return LiveCalibration(
        n_obs=24, har_mae=1.5, persistence_mae=2.0,
        har_beats_persistence=True, mean_bias=0.5,
        breakout_count=1, breakout_rate=1 / 24,
        worst_ratio=3.0, best_ratio=1.0,
        recent_mae_7d=1.2, recent_mae_30d=1.4, is_degrading=False,
    )


# ---------------------------------------------------------------------------
# send_message
# ---------------------------------------------------------------------------

class TestSendMessage:
    def test_send_message_success(self):
        with patch(f"{MODULE}.requests.post",
                   return_value=make_mock_response(ok=True, message_id=12345)) as m:
            result = send_message(make_config(), "hello", retry_delay=0)
        assert result.success is True
        assert result.message_id == 12345
        assert result.error is None
        assert result.attempts == 1
        m.assert_called_once()

    def test_send_message_empty_string_skipped(self):
        with patch(f"{MODULE}.requests.post") as m:
            result = send_message(make_config(), "", retry_delay=0)
        assert result == SendResult(success=True, message_id=None,
                                    error=None, attempts=0)
        m.assert_not_called()  # no HTTP call for an empty message

    def test_send_message_api_error_no_retry(self):
        resp = make_mock_response(ok=False, description="Bad Request: chat not found")
        with patch(f"{MODULE}.requests.post", return_value=resp) as m:
            result = send_message(make_config(), "hello", retry_delay=0)
        assert result.success is False
        assert result.message_id is None
        assert result.attempts == 1  # API errors are permanent - no retry
        assert "chat not found" in result.error
        m.assert_called_once()

    def test_send_message_network_error_retries(self):
        with patch(f"{MODULE}.requests.post",
                   side_effect=requests.ConnectionError("boom")) as m:
            result = send_message(make_config(), "hello", retries=3, retry_delay=0)
        assert result.success is False
        assert result.message_id is None
        assert result.error == "boom"
        assert result.attempts == 3
        assert m.call_count == 4  # initial + 3 retries

    def test_send_message_network_error_succeeds_on_retry(self):
        side_effects = [
            requests.ConnectionError("boom"),
            make_mock_response(ok=True, message_id=77),
        ]
        with patch(f"{MODULE}.requests.post", side_effect=side_effects) as m:
            result = send_message(make_config(), "hello", retry_delay=0)
        assert result.success is True
        assert result.message_id == 77
        assert result.attempts == 2
        assert m.call_count == 2

    def test_send_message_timeout_retries(self):
        with patch(f"{MODULE}.requests.post",
                   side_effect=requests.Timeout("slow")) as m:
            result = send_message(make_config(), "hello", retries=3, retry_delay=0)
        assert result.success is False
        assert m.call_count == 4  # timeouts retry like other network errors
        assert result.attempts == 3

    def test_send_message_never_raises(self):
        with patch(f"{MODULE}.requests.post",
                   side_effect=RuntimeError("unexpected")):
            result = send_message(make_config(), "hello", retry_delay=0)
        assert isinstance(result, SendResult)
        assert result.success is False
        assert "unexpected" in result.error

    def test_send_message_logs_token_never(self, caplog):
        with patch(f"{MODULE}.requests.post",
                   return_value=make_mock_response(ok=True)):
            with caplog.at_level(logging.DEBUG):
                send_message(make_config(), "hello", retry_delay=0)
        assert TOKEN not in caplog.text
        assert CHAT_ID not in caplog.text
        assert "message_id=12345" in caplog.text  # success logged (INFO)

    def test_send_message_correct_url_format(self):
        with patch(f"{MODULE}.requests.post",
                   return_value=make_mock_response(ok=True)) as m:
            send_message(make_config(), "hello", retry_delay=0)
        url = m.call_args.args[0]
        assert url == f"{API_BASE_URL}/bot{TOKEN}/{SEND_MESSAGE_METHOD}"
        body = m.call_args.kwargs["json"]
        assert body["chat_id"] == CHAT_ID
        assert body["text"] == "hello"

    def test_send_message_timeout_parameter(self):
        with patch(f"{MODULE}.requests.post",
                   return_value=make_mock_response(ok=True)) as m:
            send_message(make_config(), "hello", retry_delay=0)
        assert m.call_args.kwargs["timeout"] == TIMEOUT_SECONDS
        assert TIMEOUT_SECONDS == 10


# ---------------------------------------------------------------------------
# send_forecast
# ---------------------------------------------------------------------------

class TestSendForecast:
    def test_send_forecast_builds_correct_message(self):
        btc = make_forecast(predicted=100.5, regime="high")
        eth = make_forecast(predicted=25.25)
        with patch(f"{MODULE}.send_message",
                   return_value=SendResult(True, 1, None, 1)) as m:
            result = send_forecast(make_config(), btc, eth,
                                   "2024-01-15T14:00:00Z")
        assert result.success is True
        text = m.call_args.args[1]  # second positional arg of send_message
        assert "🔮 HAR Volatility Forecast" in text
        assert "BTC/USDT 1h" in text
        assert "$100.50" in text
        assert "ETH/USDT 1h" in text
        assert "$25.25" in text
        assert "⏰ 2024-01-15T14:00:00Z UTC" in text
        assert "p<1e-26" in text

    def test_send_forecast_regime_shown(self):
        btc = make_forecast(predicted=100.0, regime="high")
        eth = make_forecast(predicted=25.0, regime="low")
        with patch(f"{MODULE}.send_message",
                   return_value=SendResult(True, 1, None, 1)) as m:
            send_forecast(make_config(), btc, eth, "2024-01-15T14:00:00Z")
        text = m.call_args.args[1]
        assert "Regime: high" in text
        assert "Regime: low" in text

    def test_send_forecast_regime_none_shows_na(self):
        btc = make_forecast(predicted=100.0)
        eth = make_forecast(predicted=25.0)
        with patch(f"{MODULE}.send_message",
                   return_value=SendResult(True, 1, None, 1)) as m:
            send_forecast(make_config(), btc, eth, "2024-01-15T14:00:00Z")
        text = m.call_args.args[1]
        assert "Regime: N/A" in text  # both regimes None -> N/A
        assert "Regime: high" not in text

    def test_send_forecast_returns_sendresult(self):
        btc = make_forecast(predicted=100.0)
        eth = make_forecast(predicted=25.0)
        with patch(f"{MODULE}.send_message",
                   return_value=SendResult(True, 1, None, 1)):
            result = send_forecast(make_config(), btc, eth,
                                   "2024-01-15T14:00:00Z")
        assert isinstance(result, SendResult)


# ---------------------------------------------------------------------------
# send_breakout
# ---------------------------------------------------------------------------

class TestSendBreakout:
    def test_send_breakout_no_breakout_skipped(self):
        result = check_breakout(150.0, 100.0)  # ratio 1.5, not a breakout
        with patch(f"{MODULE}.requests.post") as m:
            out = send_breakout(make_config(), "BTC/USDT", "1h", result,
                                "2024-01-15T14:00:00Z")
        assert out == SendResult(success=True, message_id=None,
                                 error=None, attempts=0)
        m.assert_not_called()

    def test_send_breakout_moderate_sends(self):
        result = check_breakout(250.0, 100.0)  # moderate, is_breakout=True
        assert result.severity == "moderate"
        with patch(f"{MODULE}.requests.post",
                   return_value=make_mock_response(ok=True)) as m:
            out = send_breakout(make_config(), "BTC/USDT", "1h", result,
                                "2024-01-15T14:00:00Z")
        assert out.success is True
        m.assert_called_once()

    def test_send_breakout_extreme_sends(self):
        result = check_breakout(600.0, 100.0)  # extreme
        assert result.severity == "extreme"
        with patch(f"{MODULE}.requests.post",
                   return_value=make_mock_response(ok=True)) as m:
            out = send_breakout(make_config(), "BTC/USDT", "1h", result,
                                "2024-01-15T14:00:00Z")
        assert out.success is True
        m.assert_called_once()

    def test_send_breakout_returns_sendresult(self):
        result = check_breakout(250.0, 100.0)
        with patch(f"{MODULE}.requests.post",
                   return_value=make_mock_response(ok=True)):
            out = send_breakout(make_config(), "BTC/USDT", "1h", result,
                                "2024-01-15T14:00:00Z")
        assert isinstance(out, SendResult)


# ---------------------------------------------------------------------------
# send_calibration_report
# ---------------------------------------------------------------------------

class TestSendCalibrationReport:
    def test_send_calibration_sends_message(self):
        cal = make_calibration()
        with patch(f"{MODULE}.requests.post",
                   return_value=make_mock_response(ok=True)) as m:
            result = send_calibration_report(make_config(), "BTC/USDT", "1h", cal)
        assert result.success is True
        m.assert_called_once()
        assert "HAR MAE" in m.call_args.kwargs["json"]["text"]

    def test_send_calibration_returns_sendresult(self):
        cal = make_calibration()
        with patch(f"{MODULE}.requests.post",
                   return_value=make_mock_response(ok=True)):
            result = send_calibration_report(make_config(), "BTC/USDT", "1h", cal)
        assert isinstance(result, SendResult)


# ---------------------------------------------------------------------------
# TelegramConfig
# ---------------------------------------------------------------------------

class TestTelegramConfig:
    def test_config_from_env_success(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env_token_1")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "env_chat_1")
        with patch(f"{MODULE}.load_dotenv") as ld:
            cfg = TelegramConfig.from_env()
        ld.assert_called_once()
        assert cfg.bot_token == "env_token_1"
        assert cfg.chat_id == "env_chat_1"

    def test_config_from_env_missing_token(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "env_chat_1")
        with patch(f"{MODULE}.load_dotenv"):
            with pytest.raises(EnvironmentError, match="TELEGRAM_BOT_TOKEN"):
                TelegramConfig.from_env()

    def test_config_from_env_missing_chat_id(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env_token_1")
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        with patch(f"{MODULE}.load_dotenv"):
            with pytest.raises(EnvironmentError, match="TELEGRAM_CHAT_ID"):
                TelegramConfig.from_env()

    def test_config_from_env_both_missing(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        with patch(f"{MODULE}.load_dotenv"):
            with pytest.raises(EnvironmentError, match="TELEGRAM_BOT_TOKEN"):
                TelegramConfig.from_env()


# ---------------------------------------------------------------------------
# send_startup / send_shutdown
# ---------------------------------------------------------------------------

class TestLifecycleMessages:
    def test_send_startup_message_content(self):
        with patch(f"{MODULE}.requests.post",
                   return_value=make_mock_response(ok=True)) as m:
            result = send_startup_message(make_config())
        assert result.success is True
        text = m.call_args.kwargs["json"]["text"]
        assert "HAR Alert Bot started" in text
        assert "PAPER RESEARCH ONLY" in text
        assert "No trades are placed." in text

    def test_send_shutdown_message_content(self):
        with patch(f"{MODULE}.requests.post",
                   return_value=make_mock_response(ok=True)) as m:
            result = send_shutdown_message(make_config())
        assert result.success is True
        text = m.call_args.kwargs["json"]["text"]
        assert "HAR Alert Bot stopped" in text
        assert "No trades were placed." in text

    def test_send_startup_returns_sendresult(self):
        with patch(f"{MODULE}.requests.post",
                   return_value=make_mock_response(ok=True)):
            result = send_startup_message(make_config())
        assert isinstance(result, SendResult)
