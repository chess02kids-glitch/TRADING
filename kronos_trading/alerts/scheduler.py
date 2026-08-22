"""Hourly scheduler for the HAR alert bot (Step 5).

This is the brain of the bot. Every hour, 30 seconds after the hour mark, it:

1. Fills actual ranges for pending predictions from previous runs
   (bars that have since closed - never the current bar).
2. Fetches the latest candles for each asset (one fetch serves both the
   pending-fill and the new prediction).
3. Runs HAR to predict the next bar's range.
4. Classifies the volatility regime (rolling 30-day terciles).
5. Logs the prediction to SQLite - always for the NEXT bar (currently
   forming), never for a bar whose actual is already known.
6. Sends the combined forecast message to Telegram.
7. Sends breakout alerts for any pending row whose actual exceeded
   ``breakout_threshold * predicted``.
8. Every ``calibration_interval_hours``, sends the calibration report.
9. Sleeps until the next cycle.

Robustness rules (enforced by design):

* RULE 1: one bad asset never kills another - each asset runs in its own
  try/except and failures are collected in ``CycleResult.errors``.
* RULE 2: one bad cycle never kills the loop - ``run_forever`` catches
  everything, sleeps 60 s and continues.
* RULE 3: no look-ahead - predictions are logged for the next (forming) bar
  with ``actual_range`` NULL; actuals are filled only for closed bars.
* RULE 4: timestamps are always UTC (timezone-aware ``datetime``).
* RULE 5: the exchange is injectable (None = default CCXT Binance client).
* RULE 6: credentials only ever come in via ``TelegramConfig``.
* RULE 7: the DB path only ever comes in via ``SchedulerConfig``.

Timestamp convention (documented deviation from the Step 5 draft):

The draft says to log predictions under ``next_bar_timestamp(current_time)``
which, for a run at 15:00:30, yields 16:00. But at 15:00:30 the last CLOSED
bar is 14:00, so the validated h=1 HAR predicts the bar that opened at 15:00
(it closes at 16:00). Logging under 16:00 would silently shift the model to a
2-bar horizon and every pending row would wait two runs before filling.
Instead, predictions are logged under the *open time of the predicted bar*
(``_bar_open_time_utc`` = the timeframe boundary at/before ``now``, i.e. the
currently forming bar) - this keeps the validated one-step horizon and the
"logged before close, filled after close" guarantee. ``next_bar_timestamp``
itself is implemented exactly as specified and is used for cycle timing
(sleep scheduling) and tests.
"""
from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from kronos_trading.alerts.breakout_detector import (
    BreakoutResult,
    check_breakout,
    get_live_calibration,
)
from kronos_trading.alerts.har_forecaster import (
    HarForecast,
    classify_regime,
    fetch_candles,
    predict_next_range,
)
from kronos_trading.alerts.market_context import (
    get_market_context,
    MarketContext,
)
from kronos_trading.alerts.prediction_logger import (
    DEFAULT_DB_PATH,
    get_pending_predictions,
    get_prediction_history,
    initialize_db,
    log_prediction,
    update_actual,
)
from kronos_trading.alerts.telegram_sender import (
    SendResult,
    TelegramConfig,
    send_breakout,
    send_calibration_report,
    send_forecast,
    send_shutdown_message,
    send_startup_message,
)

logger = logging.getLogger(__name__)

_ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"
_CYCLE_ERROR_SLEEP_SECONDS = 60.0


@dataclass
class CycleResult:
    """Outcome of one full hourly cycle."""

    timestamp: str                       # cycle timestamp (ISO8601 UTC)
    success: bool                        # True iff errors is empty
    assets_processed: List[str]
    errors: List[str]
    forecasts: Dict[str, HarForecast]    # asset -> forecast for the next bar
    breakouts: Dict[str, BreakoutResult]  # asset -> latest breakout this cycle
    send_results: Dict[str, SendResult]  # "forecast", "breakout:{asset}:{ts}"
    duration_seconds: float


@dataclass
class SchedulerConfig:
    """Configuration for the alert-bot scheduler (never hardcoded paths)."""

    assets: List[str]
    timeframe: str
    candle_delay_seconds: float          # wait after the hour mark for close
    n_candles: int                       # candles to fetch per asset
    breakout_threshold: float            # actual > threshold * predicted
    calibration_interval_hours: int
    db_path: str

    @classmethod
    def default(cls) -> "SchedulerConfig":
        return cls(
            assets=["BTC/USDT", "ETH/USDT"],
            timeframe="1h",
            candle_delay_seconds=30.0,
            n_candles=800,
            breakout_threshold=2.0,
            calibration_interval_hours=24,
            db_path=DEFAULT_DB_PATH,
        )


# ---------------------------------------------------------------------------
# Time helpers (pure, UTC only)
# ---------------------------------------------------------------------------

def _as_utc(now: datetime) -> datetime:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now


def _next_bar_utc(now: datetime, timeframe: str) -> datetime:
    """Datetime of the bar boundary strictly after ``now`` (UTC)."""
    now = _as_utc(now)
    if timeframe == "1h":
        floored = now.replace(minute=0, second=0, microsecond=0)
        return floored + timedelta(hours=1)
    if timeframe == "4h":
        bucket = (now.hour // 4) * 4
        floored = now.replace(hour=bucket, minute=0, second=0, microsecond=0)
        if floored <= now:
            floored += timedelta(hours=4)
        return floored
    if timeframe == "1d":
        floored = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return floored + timedelta(days=1)
    raise ValueError(f"Unsupported timeframe {timeframe!r}; allowed: 1h, 4h, 1d")


def _bar_open_time_utc(now: datetime, timeframe: str) -> datetime:
    """Open time (UTC) of the bar that is currently forming at ``now``.

    This is the h=1 prediction target given features through the last closed
    bar: e.g. at 15:00:30 with 1h bars the forming bar opened at 15:00 and
    closes at 16:00. See the module docstring for why predictions are logged
    under this timestamp.
    """
    now = _as_utc(now)
    if timeframe == "1h":
        return now.replace(minute=0, second=0, microsecond=0)
    if timeframe == "4h":
        return now.replace(hour=(now.hour // 4) * 4,
                           minute=0, second=0, microsecond=0)
    if timeframe == "1d":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(f"Unsupported timeframe {timeframe!r}; allowed: 1h, 4h, 1d")


def next_bar_timestamp(current_time: datetime, timeframe: str) -> str:
    """ISO8601 UTC timestamp of the NEXT bar after ``current_time``.

    Examples (1h): 14:00:00 -> 15:00:00Z, 14:07:22 -> 15:00:00Z, 14:59:59 ->
    15:00:00Z. For 4h the next boundary is 00/04/08/12/16/20; for 1d it is the
    next midnight UTC.
    """
    return _next_bar_utc(current_time, timeframe).strftime(_ISO_FMT)


def seconds_until_next_cycle(
    now: datetime,
    timeframe: str,
    delay_seconds: float = 30.0,
) -> float:
    """Seconds to sleep until the next candle close plus ``delay_seconds``.

    Never returns less than 1.0 (a 0/negative sleep would busy-loop).
    """
    next_dt = _next_bar_utc(now, timeframe)
    delta = (next_dt - now).total_seconds() + float(delay_seconds)
    return max(1.0, delta)


# ---------------------------------------------------------------------------
# Pending-prediction fill
# ---------------------------------------------------------------------------

def _iso_to_ms(ts: str) -> Optional[int]:
    """Parse an ISO8601 UTC string (``...Z``, optional fraction) to epoch ms."""
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            dt = datetime.strptime(ts, fmt)
        except ValueError:
            continue
        return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    return None


def _fill_pending(
    telegram_config: TelegramConfig,
    scheduler_config: SchedulerConfig,
    asset: str,
    candles: List[Any],
    result: CycleResult,
) -> None:
    """Complete pending predictions whose bars have now closed.

    A pending row is only completed when the candle for its exact bar-open
    timestamp is present in the freshly fetched candles. Bars still forming
    (not yet in the fetch) are skipped with a warning and stay pending until
    the next cycle - the "filled in next run" guarantee. When a row is
    completed, ``check_breakout`` decides whether a breakout alert is sent.
    """
    tf = scheduler_config.timeframe
    db = scheduler_config.db_path
    pending = get_pending_predictions(db, asset, tf)
    if not pending:
        return
    by_ts = {c.timestamp_ms: c for c in candles}
    for row in pending:
        row_ts = row["timestamp"]
        ts_ms = _iso_to_ms(row_ts)
        if ts_ms is None:
            logger.warning("Pending row %s %s @ %s: unparseable timestamp",
                           asset, tf, row_ts)
            continue
        candle = by_ts.get(ts_ms)
        if candle is None:
            logger.warning("Pending %s %s @ %s: candle not in fetched window "
                           "(still forming or outside window) - will retry",
                           asset, tf, row_ts)
            continue
        actual = float(candle.high) - float(candle.low)
        if not update_actual(db, row_ts, asset, tf, actual):
            continue  # already completed by an earlier run - keep first value
        breakout = check_breakout(
            actual, float(row["har_predicted_range"]),
            threshold=scheduler_config.breakout_threshold,
        )
        if breakout.is_breakout:
            result.breakouts[asset] = breakout
            key = f"breakout:{asset}:{row_ts}"
            result.send_results[key] = send_breakout(
                telegram_config, asset, tf, breakout, row_ts)


# ---------------------------------------------------------------------------
# Single cycle
# ---------------------------------------------------------------------------

def run_single_cycle(
    telegram_config: TelegramConfig,
    scheduler_config: SchedulerConfig,
    exchange: Any = None,
    now: Optional[datetime] = None,
) -> CycleResult:
    """Run one complete hourly cycle (independently testable).

    Per asset (each in its own try/except - RULE 1): fetch candles, complete
    pending predictions whose bars have closed, predict the next bar's range
    with HAR, classify the regime, and log the prediction (before the bar
    closes). After all assets: send the combined forecast message. A failure
    in one asset never affects the others; all failures land in
    ``CycleResult.errors`` and ``success`` is False when any occurred.

    Args:
        telegram_config: credentials (never read from env here).
        scheduler_config: assets/timeframe/db path (never hardcoded here).
        exchange: injectable CCXT-compatible exchange; None creates the
            default public Binance client inside ``fetch_candles``.
        now: current time (UTC) - test hook. Defaults to the real clock.

    Returns:
        CycleResult with per-asset forecasts, breakouts, send outcomes,
        errors and duration.
    """
    started = time.perf_counter()
    tf = scheduler_config.timeframe
    db = scheduler_config.db_path
    now = _as_utc(now if now is not None else datetime.now(timezone.utc))
    cycle_ts = _bar_open_time_utc(now, tf).strftime(_ISO_FMT)
    result = CycleResult(
        timestamp=cycle_ts, success=True, assets_processed=[],
        errors=[], forecasts={}, breakouts={}, send_results={},
        duration_seconds=0.0,
    )

    try:
        initialize_db(db)
    except Exception as exc:  # noqa: BLE001 - one bad cycle must not crash
        result.errors.append(f"db init failed: {exc}")
        result.success = False
        result.duration_seconds = time.perf_counter() - started
        logger.exception("Cycle %s: DB init failed", cycle_ts)
        return result

    # Fetch market context early so it can be logged with predictions
    # Optional enrichment — never blocks the cycle
    context: MarketContext | None = None
    try:
        context = get_market_context(timeout=8.0)
        if context.is_complete:
            logger.info(
                f"Market context fetched: F&G={context.fear_greed_value}, "
                f"BTC dom={context.btc_dominance}%"
            )
        else:
            logger.warning(
                f"Market context incomplete: {context.fetch_errors}"
            )
    except Exception as e:
        logger.warning(
            f"Market context fetch failed unexpectedly: {e}"
        )
        context = None

    for asset in scheduler_config.assets:
        try:
            candles = fetch_candles(asset, tf, n=scheduler_config.n_candles,
                                    exchange=exchange)
        except Exception as exc:  # noqa: BLE001 - RULE 1
            result.errors.append(f"{asset}: fetch failed: {exc}")
            logger.warning("Cycle %s: fetch failed for %s: %s", cycle_ts,
                           asset, exc)
            continue
        result.assets_processed.append(asset)

        try:
            _fill_pending(telegram_config, scheduler_config, asset, candles,
                          result)
        except Exception as exc:  # noqa: BLE001 - pending failure != fatal
            result.errors.append(f"{asset}: pending fill failed: {exc}")
            logger.warning("Cycle %s: pending fill failed for %s: %s",
                           cycle_ts, asset, exc)

        try:
            forecast = predict_next_range(candles)
            historical = [float(c.high) - float(c.low)
                          for c in candles[-720:]]
            regime = classify_regime(forecast.predicted_range, historical)
            forecast = dataclasses.replace(forecast, regime=regime)
            pred_ts = _bar_open_time_utc(now, tf).strftime(_ISO_FMT)
            log_prediction(db, pred_ts, asset, tf, forecast, market_context=context)
            result.forecasts[asset] = forecast
        except Exception as exc:  # noqa: BLE001 - RULE 1
            result.errors.append(f"{asset}: prediction failed: {exc}")
            logger.warning("Cycle %s: prediction failed for %s: %s",
                           cycle_ts, asset, exc)


    # Combined forecast message (BTC + ETH by send_forecast's contract).
    try:
        if set(scheduler_config.assets) <= set(result.forecasts):
            first = result.forecasts[scheduler_config.assets[0]]
            second = result.forecasts[scheduler_config.assets[1]]
            result.send_results["forecast"] = send_forecast(
                telegram_config, first, second, cycle_ts, context=context)
        else:
            missing = set(scheduler_config.assets) - set(result.forecasts)
            logger.warning("Cycle %s: forecast message skipped - missing "
                           "forecasts for %s", cycle_ts, sorted(missing))
    except Exception as exc:  # noqa: BLE001 - sending must not crash the cycle
        result.errors.append(f"forecast send failed: {exc}")
        logger.warning("Cycle %s: forecast send failed: %s", cycle_ts, exc)

    result.success = len(result.errors) == 0
    result.duration_seconds = time.perf_counter() - started
    return result


# ---------------------------------------------------------------------------
# Calibration cycle
# ---------------------------------------------------------------------------

def run_calibration_cycle(
    telegram_config: TelegramConfig,
    scheduler_config: SchedulerConfig,
) -> None:
    """Send the calibration report for every asset (best-effort, never raises).

    Uses only completed predictions (``get_prediction_history`` + ``get_live_calibration``);
    assets with fewer than 24 observations log an info line and are skipped.
    """
    tf = scheduler_config.timeframe
    db = scheduler_config.db_path
    for asset in scheduler_config.assets:
        try:
            history = get_prediction_history(db, asset, tf)
            cal = get_live_calibration(history)
            if cal is None:
                logger.info("Insufficient history for calibration: %s %s",
                            asset, tf)
                continue
            send_calibration_report(telegram_config, asset, tf, cal)
        except Exception as exc:  # noqa: BLE001 - never crash the loop
            logger.warning("Calibration failed for %s: %s", asset, exc)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_forever(
    telegram_config: TelegramConfig,
    scheduler_config: SchedulerConfig,
    exchange: Any = None,
) -> None:
    """Run the hourly alert loop until interrupted (clean shutdown on exit).

    Startup: initialize the DB, send the startup message. Then loop: run one
    cycle, log its summary, run the calibration report when due, sleep until
    the next cycle. ``KeyboardInterrupt``/``SystemExit`` send the shutdown
    message and return (no re-raise). Any other exception is logged with its
    traceback, followed by a 60 s sleep - the loop never dies.
    """
    db = scheduler_config.db_path
    tf = scheduler_config.timeframe
    try:
        initialize_db(db)
    except Exception:  # noqa: BLE001 - startup DB failure must not prevent boot
        logger.exception("initialize_db failed at startup; continuing")

    try:
        send_startup_message(telegram_config)
    except Exception:  # noqa: BLE001 - startup message is best-effort
        logger.exception("Startup message failed")

    last_calibration_time: Optional[datetime] = None

    while True:
        try:
            now = datetime.now(timezone.utc)
            result = run_single_cycle(telegram_config, scheduler_config,
                                      exchange=exchange, now=now)
            logger.info(
                "Cycle %s: success=%s assets=%s errors=%d duration=%.1fs",
                result.timestamp, result.success, result.assets_processed,
                len(result.errors), result.duration_seconds,
            )

            if (last_calibration_time is None
                    or (now - last_calibration_time).total_seconds() / 3600.0
                    > scheduler_config.calibration_interval_hours):
                run_calibration_cycle(telegram_config, scheduler_config)
                last_calibration_time = now

            sleep_seconds = seconds_until_next_cycle(
                datetime.now(timezone.utc), tf,
                scheduler_config.candle_delay_seconds)
            logger.info("Sleeping %.0f seconds", sleep_seconds)
            time.sleep(sleep_seconds)

        except KeyboardInterrupt:
            logger.info("Interrupt received - shutting down")
            try:
                send_shutdown_message(telegram_config)
            except Exception:  # noqa: BLE001 - shutdown send is best-effort
                logger.exception("Shutdown message failed")
            logger.info("Bot stopped cleanly")
            return
        except SystemExit:
            try:
                send_shutdown_message(telegram_config)
            except Exception:  # noqa: BLE001
                logger.exception("Shutdown message failed")
            logger.info("Bot stopped cleanly")
            return
        except Exception:  # noqa: BLE001 - RULE 2: never crash the loop
            logger.exception("Cycle failed; continuing in %.0f seconds",
                             _CYCLE_ERROR_SLEEP_SECONDS)
            try:
                time.sleep(_CYCLE_ERROR_SLEEP_SECONDS)
            except KeyboardInterrupt:
                try:
                    send_shutdown_message(telegram_config)
                except Exception:  # noqa: BLE001
                    logger.exception("Shutdown message failed")
                logger.info("Bot stopped cleanly")
                return
            continue
