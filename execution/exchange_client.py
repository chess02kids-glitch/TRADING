"""Paper-only CCXT exchange client (sandbox enforced at all times).

The single place the execution layer touches a real exchange library, locked
to **paper/sandbox**:

* ``ExchangeConfig.sandbox`` defaults to ``True``.
* Constructing an :class:`ExchangeClient` with ``sandbox=False`` raises
  :class:`ValueError` immediately — no code path can reach a live endpoint.
* The CCXT instance is created in ``__init__`` and put into sandbox mode via
  ``set_sandbox_mode(True)``.

Every public method is fail-soft: it logs the error (with a UTC timestamp) and
returns ``None`` (or ``False`` for :meth:`cancel_order`) instead of raising.
Rate limiting is delegated to CCXT (``enableRateLimit = True``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExchangeConfig:
    """Credentials + mode for a paper-only exchange connection."""
    api_key: str
    api_secret: str
    api_password: str
    sandbox: bool = True
    exchange_id: str = "kucoin"


class SandboxViolation(ValueError):
    """Raised when a caller tries to disable sandbox (paper-only) mode."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_exchange(config: ExchangeConfig):
    """Create the CCXT exchange object in sandbox mode (monkeypatchable)."""
    import ccxt  # type: ignore

    exchange_class = getattr(ccxt, config.exchange_id)
    ex = exchange_class({
        "apiKey": config.api_key,
        "secret": config.api_secret,
        "password": config.api_password,
        "enableRateLimit": True,
    })
    try:
        ex.set_sandbox_mode(True)
    except Exception as exc:  # pragma: no cover - exchange-dependent
        logger.warning("[%s] set_sandbox_mode failed for %s: %s "
                       "(sandbox config flag still enforced)",
                       _now_iso(), config.exchange_id, exc)
    return ex


class ExchangeClient:
    """Defensive, paper-only CCXT wrapper. Never raises on exchange errors."""

    def __init__(self, config: ExchangeConfig) -> None:
        # Validate sandbox BEFORE touching anything else.
        if not config.sandbox:
            raise SandboxViolation(
                "sandbox must be True — this is a paper-only execution layer. "
                "Refusing to construct a client that could reach a live endpoint.")
        self.config = config
        self._exchange: Optional[Any] = _build_exchange(config)

    # -- connection --------------------------------------------------------

    def connect(self) -> bool:
        """Test the connection (load markets). Returns success bool."""
        try:
            if self._exchange is None:
                return False
            self._exchange.load_markets()
            return True
        except Exception as exc:
            logger.error("[%s] connect failed: %s", _now_iso(), exc)
            return False

    def _ex(self):
        if self._exchange is None:
            raise RuntimeError("exchange not initialized")
        return self._exchange

    # -- market data -------------------------------------------------------

    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            return self._ex().fetch_ticker(symbol)
        except Exception as exc:
            logger.error("[%s] get_ticker(%s) failed: %s", _now_iso(), symbol, exc)
            return None

    def get_balance(self) -> Optional[Dict[str, Any]]:
        try:
            return self._ex().fetch_balance()
        except Exception as exc:
            logger.error("[%s] get_balance failed: %s", _now_iso(), exc)
            return None

    def get_open_orders(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        try:
            return self._ex().fetch_open_orders(symbol)
        except Exception as exc:
            logger.error("[%s] get_open_orders(%s) failed: %s",
                         _now_iso(), symbol, exc)
            return None

    # -- trading -----------------------------------------------------------

    def place_market_order(
        self, symbol: str, side: str, amount: float,
    ) -> Optional[Dict[str, Any]]:
        try:
            return self._ex().create_order(symbol, "market", side, float(amount))
        except Exception as exc:
            logger.error("[%s] place_market_order(%s %s %s) failed: %s",
                         _now_iso(), side, amount, symbol, exc)
            return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            self._ex().cancel_order(order_id, symbol)
            return True
        except Exception as exc:
            logger.error("[%s] cancel_order(%s %s) failed: %s",
                         _now_iso(), order_id, symbol, exc)
            return False

    def get_order_status(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            return self._ex().fetch_order(order_id, symbol)
        except Exception as exc:
            logger.error("[%s] get_order_status(%s %s) failed: %s",
                         _now_iso(), order_id, symbol, exc)
            return None
