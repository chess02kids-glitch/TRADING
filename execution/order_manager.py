"""Vol-targeted paper position sizing and order construction.

Turns a :class:`SignalInput` into an :class:`OrderParams` using HAR-predicted
volatility to size the position, then executes it through the paper
:class:`~execution.exchange_client.ExchangeClient`.

Sizing (pre-registered, fixed; returns the USD notional ``size_usd``)::

    har_vol   = har_predicted_range / current_price    # fractional move
    size_usd  = (account_size * target_vol) / har_vol  # vol-targeted USD
    max_size  = account_size * 0.10                     # 10% account cap
    size_usd  = min(size_usd, max_size)
    size_base = size_usd / current_price                # base-currency amount

Guard rails (any violation returns size 0 / ``None``):

* ``har_vol <= 0`` (needs ``har_predicted_range > 0`` and ``price > 0``) → 0.
* ``size_usd < 10`` (below the $10 minimum) → 0.

Paper only; ``sandbox`` is enforced by the exchange client. Not wired to the
live HAR bot.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from execution.exchange_client import ExchangeClient

logger = logging.getLogger(__name__)

DEFAULT_TARGET_VOL = 0.01          # 1% account-volatility target
MAX_POSITION_FRACTION = 0.10       # hard 10%-of-account cap
MIN_ORDER_USD = 10.0               # skip orders below $10 notional


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
    """Fully specified paper order (USD notional + base-currency amount)."""
    symbol: str
    side: str                       # "buy" or "sell"
    size_usd: float
    size_base: float
    target_vol: float
    har_vol_estimate: float
    account_size: float


def compute_position_size(
    signal: SignalInput,
    account_size: float,
    current_price: float,
    target_vol: float = DEFAULT_TARGET_VOL,
) -> float:
    """USD notional size after HAR-vol targeting and the 10% cap.

    Returns ``0.0`` when the signal is unusable (non-positive price, non-positive
    HAR volatility) or the resulting size is below the $10 minimum.
    """
    if current_price <= 0 or account_size <= 0:
        return 0.0
    if signal.har_predicted_range <= 0:
        return 0.0
    har_vol = signal.har_predicted_range / current_price
    if har_vol <= 0:
        return 0.0
    size_usd = (account_size * target_vol) / har_vol
    max_size = account_size * MAX_POSITION_FRACTION
    size_usd = min(size_usd, max_size)
    if size_usd < MIN_ORDER_USD:
        return 0.0
    return float(size_usd)


def build_order_params(
    signal: SignalInput,
    account_size: float,
    current_price: float,
) -> Optional[OrderParams]:
    """Construct :class:`OrderParams` or ``None`` when the size would be 0."""
    if signal.direction == 0:
        logger.info("Skip %s @ %s: direction is 0", signal.asset, signal.timestamp)
        return None
    if current_price <= 0:
        logger.info("Skip %s @ %s: non-positive price", signal.asset, signal.timestamp)
        return None
    size_usd = compute_position_size(signal, account_size, current_price)
    if size_usd == 0.0:
        logger.info("Skip %s @ %s: computed size is 0 (vol/minimum check)",
                    signal.asset, signal.timestamp)
        return None
    har_vol = signal.har_predicted_range / current_price
    side = "buy" if signal.direction > 0 else "sell"
    return OrderParams(
        symbol=signal.asset,
        side=side,
        size_usd=size_usd,
        size_base=size_usd / current_price,
        target_vol=DEFAULT_TARGET_VOL,
        har_vol_estimate=har_vol,
        account_size=account_size,
    )


def execute_signal(
    signal: SignalInput,
    client: ExchangeClient,
    account_size: float,
    current_price: float,
) -> Optional[Dict[str, Any]]:
    """Size → build → place a paper market order; returns the order or ``None``.

    ``current_price`` is supplied by the caller (the orchestrator reads it from
    the exchange ticker). Orders are placed in **base currency** (``size_base``).
    """
    params = build_order_params(signal, account_size, current_price)
    if params is None:
        logger.info("execute_signal: no order built for %s", signal.asset)
        return None
    order = client.place_market_order(params.symbol, params.side, params.size_base)
    if order is None:
        logger.info("execute_signal: order rejected for %s", signal.asset)
        return None
    logger.info("execute_signal: placed %s %s base of %s ($%.2f)",
                params.side, params.size_base, params.symbol, params.size_usd)
    return order
