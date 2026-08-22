"""Paper-only CCXT execution layer (logging-only — not wired to the live bot).

Submodules:

* :mod:`execution.exchange_client`  — defensive, sandbox-locked CCXT wrapper.
* :mod:`execution.order_manager`    — HAR-vol-targeted sizing + order building.
* :mod:`execution.position_tracker` — local-SQLite paper position book.
* :mod:`execution.execution_logger` — separate audit log of signals/orders.

``sandbox=True`` is enforced everywhere; there is no live-trading code path.
"""
from __future__ import annotations

__all__ = [
    "exchange_client",
    "order_manager",
    "position_tracker",
    "execution_logger",
]
