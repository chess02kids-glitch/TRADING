"""Telegram Bot API wrapper for the alert bot (Step 4).

This module is the ONLY place in the alerts package that touches Telegram.
``har_forecaster``, ``prediction_logger`` and ``breakout_detector`` are fully
isolated from it.

Responsibilities:

* ``TelegramConfig``            - credentials from environment variables only
                                 (never function arguments, never hardcoded).
* ``send_message``              - single POST to the Bot API with retries on
                                 transient network errors, no retry on API
                                 errors, never raises, always returns
                                 ``SendResult``.
* ``send_forecast`` /
  ``send_breakout`` /
  ``send_calibration_report``   - pre-formatted messages (Step 3 formatting
                                 functions) sent through ``send_message``.
* ``send_startup_message`` /
  ``send_shutdown_message``     - lifecycle notifications.

Safety rules:

* Credentials are read from ``.env`` / environment only; nothing is logged -
  not the token (even partially), not the chat id, not message content.
* Empty text is skipped (``breakout_detector`` returns ``""`` when there is
  no alert) and reported as a silent success with ``attempts=0``.
* API-level errors (``"ok": false``) are permanent (wrong token, wrong chat
  id) and are NOT retried; only ``requests.RequestException`` network
  failures are retried, up to ``retries`` times with ``retry_delay`` between
  attempts.
* No public function raises; every path returns ``SendResult`` so a send
  failure can never crash the scheduler.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests
from dotenv import load_dotenv

from kronos_trading.alerts.breakout_detector import (
    BreakoutResult,
    LiveCalibration,
    format_breakout_message,
    format_calibration_message,
)
from kronos_trading.alerts.har_forecaster import HarForecast

logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.telegram.org"
SEND_MESSAGE_METHOD = "sendMessage"
TIMEOUT_SECONDS = 10.0          # per-request timeout (spec)


@dataclass
class TelegramConfig:
    """Telegram bot credentials. Never constructed with hardcoded values."""

    bot_token: str
    chat_id: str

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        """Load credentials from ``.env`` / environment variables.

        Raises:
            EnvironmentError: when TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is
                missing or empty.
        """
        load_dotenv()
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            raise EnvironmentError(
                "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set"
            )
        return cls(bot_token=token, chat_id=chat_id)


@dataclass
class SendResult:
    """Outcome of one Telegram send attempt."""

    success: bool
    message_id: Optional[int]
    error: Optional[str]
    attempts: int


def _send_url(config: TelegramConfig) -> str:
    return f"{API_BASE_URL}/bot{config.bot_token}/{SEND_MESSAGE_METHOD}"


def send_message(
    config: TelegramConfig,
    text: str,
    parse_mode: str = "HTML",
    retries: int = 3,
    retry_delay: float = 2.0,
) -> SendResult:
    """POST ``text`` to the Telegram Bot API and report the outcome.

    Args:
        config: credentials.
        text: message body. Empty/whitespace-only text is skipped and
            reported as a silent success (``attempts=0``, no HTTP call) -
            this is how "no breakout" is represented.
        parse_mode: ``"HTML"`` (default) or ``"Markdown"``; ``None`` sends
            plain text.
        retries: how many times a network error is retried (the initial
            attempt is not counted; total calls = retries + 1).
        retry_delay: seconds between retry attempts.

    Returns:
        ``SendResult`` - never raises. API errors (``ok=false``) return after
        one attempt; network errors are retried up to ``retries`` times.
    """
    if not text or not str(text).strip():
        logger.warning("Telegram send skipped: empty message")
        return SendResult(success=True, message_id=None, error=None, attempts=0)

    url = _send_url(config)
    payload = {"chat_id": config.chat_id, "text": str(text)}
    if parse_mode:
        payload["parse_mode"] = parse_mode

    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.warning("Telegram network error (attempt %d): %s", attempt, exc)
            if attempt <= retries:
                time.sleep(retry_delay)
                continue
            logger.warning("Telegram send failed after %d attempts", attempt)
            return SendResult(success=False, message_id=None,
                              error=str(exc), attempts=retries)
        except Exception as exc:  # noqa: BLE001 - never raise from send_message
            # Unexpected (e.g. malformed response body): fail fast, no retry.
            logger.warning("Telegram unexpected error (attempt %d): %s",
                           attempt, exc)
            return SendResult(success=False, message_id=None,
                              error=str(exc), attempts=attempt)

        if not data.get("ok"):
            description = str(data.get("description") or "unknown API error")
            logger.warning("Telegram API error: %s", description)
            return SendResult(success=False, message_id=None,
                              error=description, attempts=attempt)

        message_id = (data.get("result") or {}).get("message_id")
        logger.info("Telegram send OK: message_id=%s", message_id)
        return SendResult(success=True, message_id=message_id,
                          error=None, attempts=attempt)


def send_forecast(
    config: TelegramConfig,
    btc_forecast: HarForecast,
    eth_forecast: HarForecast,
    timestamp: str,
) -> SendResult:
    """Build and send the hourly HAR volatility forecast (plain text)."""
    btc_regime = getattr(btc_forecast, "regime", None) or "N/A"
    eth_regime = getattr(eth_forecast, "regime", None) or "N/A"
    text = "\n".join([
        "🔮 HAR Volatility Forecast",
        "━" * 20,
        "BTC/USDT 1h",
        f"  Predicted range: ${btc_forecast.predicted_range:.2f}",
        f"  Regime: {btc_regime}",
        "",
        "ETH/USDT 1h",
        f"  Predicted range: ${eth_forecast.predicted_range:.2f}",
        f"  Regime: {eth_regime}",
        "",
        f"⏰ {timestamp} UTC",
        "📊 HAR model (validated p<1e-26)",
    ])
    # Plain text message: no HTML parsing needed.
    return send_message(config, text, parse_mode=None)


def send_breakout(
    config: TelegramConfig,
    asset: str,
    timeframe: str,
    result: BreakoutResult,
    timestamp: str,
) -> SendResult:
    """Send a breakout/spike alert; skipped silently when there is no breakout."""
    if not result.is_breakout:
        return SendResult(success=True, message_id=None, error=None, attempts=0)
    text = format_breakout_message(asset, timeframe, result, timestamp)
    return send_message(config, text)


def send_calibration_report(
    config: TelegramConfig,
    asset: str,
    timeframe: str,
    cal: LiveCalibration,
) -> SendResult:
    """Send the periodic HAR calibration report."""
    text = format_calibration_message(asset, timeframe, cal)
    return send_message(config, text)


def send_startup_message(config: TelegramConfig) -> SendResult:
    """Send the one-time bot-startup notice."""
    text = "\n".join([
        "🟢 HAR Alert Bot started",
        "━" * 20,
        "Assets: BTC/USDT, ETH/USDT",
        "Timeframe: 1h",
        "Model: HAR (validated OOS)",
        "Mode: PAPER RESEARCH ONLY",
        "",
        "⚠️ This is a research monitoring",
        "tool. Not financial advice.",
        "No trades are placed.",
    ])
    result = send_message(config, text)
    if result.success:
        logger.info("Bot startup message sent (message_id=%s)", result.message_id)
    return result


def send_shutdown_message(config: TelegramConfig) -> SendResult:
    """Send the clean-shutdown notice."""
    text = "\n".join([
        "🔴 HAR Alert Bot stopped",
        "━" * 20,
        "Shutting down cleanly.",
        "No trades were placed.",
    ])
    result = send_message(config, text)
    if result.success:
        logger.info("Bot shutdown message sent (message_id=%s)", result.message_id)
    return result
