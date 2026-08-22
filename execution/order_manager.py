"""Vol-targeted paper position sizing and order construction.

Turns a :class:`SignalInput` into an :class:`OrderParams` using HAR-predicted
volatility to size the position, then (optionally) executes it through the
paper :class:`~execution.exchange_client.ExchangeClient`.

Sizing (pre-registered, fixed)::

    har_vol      = har_predicted_range / current_price   # fractional move
    notional     = (account_size * target_vol) / har_vol # $ at risk-normalized
    max_notional = account_size * 0.10                    # 10% account cap
    notional     = min(notional, max_notional)
    size (base)  = notional / current_price

Guard rails (any violation skips the trade and logs the reason):

* ``direction == 0`` → skip.
* ``har_vol <= 0`` (needs ``har_predicted_range > 0`` and ``price > 0``) → skip.
* notional below the **$10** minimum → skip.

This module places orders only on the paper/sandbox exchange client. It is
**not** wired to the live HAR bot; an optional ``execution_logger`` (duck-typed)
may be supplied to record signals / attempts / outcomes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from execution.exchange_client import ExchangeClient

logger = logging.getLogger(__name__)

DEFAULT_TARGET_VOL = 0.01          # 1% account-volatility target
MAX_POSITION_FRACTION = 0.10       # hard 10%-of-account cap
MIN_NOTIONAL_USD = 10.0            # skip orders below $10 notional


@dataclass
class SignalInput:
    """One directional signal from the analysis layer."""
    timestamp: str
    asset: str
    direction: int                 # +1 BUY, -1 SELL, 0 = no signal
    har_predicted_range: float
    confidence: float              # 0.0 – 1.0
    regime: str


@dataclass
class OrderParams:
    """Fully specified paper order (``size`` is in base currency)."""
    symbol: str
    side: str                      # "buy" or "sell"
    size: float
    target_vol: float
    har_vol_estimate: float
    account_size: float
    # Extra fields forwarded from the signal so the position tracker (which
    # stores direction / har_predicted_range / regime) can be populated from a
    # single OrderParams instance.
    direction: int = 0
    har_predicted_range: float = 0.0
    regime: str = ""


def compute_position_size(
    signal: SignalInput,
    account_size: float,
    current_price: float,
    target_vol: float = DEFAULT_TARGET_VOL,
) -> float:
    """Base-currency size after HAR-vol targeting and the 10% cap.

    Returns ``0.0`` when the signal is unusable (direction 0, non-positive
    price, or non-positive HAR volatility). The $10 minimum is enforced one
    layer up in :func:`build_order_params`.
    """
    if signal.direction == 0:
        return 0.0
    if current_price <= 0 or account_size <= 0:
        return 0.0
    if signal.har_predicted_range <= 0:
        return 0.0
    har_vol = signal.har_predicted_range / current_price
    if har_vol <= 0:
        return 0.0
    notional = (account_size * target_vol) / har_vol
    max_notional = account_size * MAX_POSITION_FRACTION
    capped_notional = min(notional, max_notional)
    return capped_notional / current_price


def build_order_params(
    signal: SignalInput,
    account_size: float,
    current_price: float,
    target_vol: float = DEFAULT_TARGET_VOL,
) -> Optional[OrderParams]:
    """Construct :class:`OrderParams` or return ``None`` with a logged reason."""
    if signal.direction == 0:
        logger.info("Skip %s @ %s: direction is 0", signal.asset, signal.timestamp)
        return None
    if current_price <= 0:
        logger.info("Skip %s @ %s: non-positive price %s", signal.asset,
                    signal.timestamp, current_price)
        return None
    if signal.har_predicted_range <= 0:
        logger.info("Skip %s @ %s: non-positive HAR range %s", signal.asset,
                    signal.timestamp, signal.har_predicted_range)
        return None
    har_vol = signal.har_predicted_range / current_price
    if har_vol <= 0:
        logger.info("Skip %s @ %s: non-positive HAR vol estimate", signal.asset,
                    signal.timestamp)
        return None

    size = compute_position_size(signal, account_size, current_price, target_vol)
    notional = size * current_price
    if notional < MIN_NOTIONAL_USD:
        logger.info("Skip %s @ %s: notional $%.2f below $%.2f minimum",
                    signal.asset, signal.timestamp, notional, MIN_NOTIONAL_USD)
        return None

    side = "buy" if signal.direction > 0 else "sell"
    return OrderParams(
        symbol=signal.asset,
        side=side,
        size=size,
        target_vol=target_vol,
        har_vol_estimate=har_vol,
        account_size=account_size,
        direction=signal.direction,
        har_predicted_range=signal.har_predicted_range,
        regime=signal.regime,
    )


def _ticker_price(ticker: Optional[Dict[str, Any]]) -> Optional[float]:
    """Extract a usable price from a CCXT ticker dict."""
    if not ticker:
        return None
    for key in ("last", "close", "bid", "ask"):
        val = ticker.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
    return None


def execute_signal(
    signal: SignalInput,
    client: ExchangeClient,
    account_size: float,
    execution_logger: Any = None,
    target_vol: float = DEFAULT_TARGET_VOL,
) -> Optional[Dict[str, Any]]:
    """Price → size → place a paper market order; returns the order or ``None``.

    ``execution_logger`` is an optional duck-typed object exposing
    ``log_order_attempt`` / ``log_order_result`` / ``log_skip``; when supplied
    every attempt and outcome is recorded there.
    """
    def _skip(reason: str) -> None:
        logger.info("execute_signal skip: %s", reason)
        if execution_logger is not None:
            try:
                execution_logger.log_skip(reason)
            except Exception:  # pragma: no cover - logger must never break trading flow
                logger.warning("execution_logger.log_skip failed", exc_info=True)

    if signal.direction == 0:
        _skip(f"direction is 0 for {signal.asset}")
        return None

    ticker = client.get_ticker(signal.asset)
    price = _ticker_price(ticker)
    if price is None:
        _skip(f"no usable price for {signal.asset}")
        return None

    params = build_order_params(signal, account_size, price, target_vol)
    if params is None:
        _skip(f"order not buildable for {signal.asset} "
              f"(size/vol/minimum check failed)")
        return None

    if execution_logger is not None:
        try:
            execution_logger.log_order_attempt(signal, params)
        except Exception:  # pragma: no cover
            logger.warning("execution_logger.log_order_attempt failed", exc_info=True)

    order = client.place_market_order(params.symbol, params.side, params.size)
    if order is None:
        if execution_logger is not None:
            try:
                execution_logger.log_order_result({"status": "rejected", "symbol": params.symbol})
            except Exception:  # pragma: no cover
                logger.warning("execution_logger.log_order_result failed", exc_info=True)
        return None

    if execution_logger is not None:
        try:
            execution_logger.log_order_result(order)
        except Exception:  # pragma: no cover
            logger.warning("execution_logger.log_order_result failed", exc_info=True)

    return order
