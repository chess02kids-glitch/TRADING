"""Phase 9A — breakout-direction continuation analysis (pure statistics).

Standalone analysis module. It does **not** connect to any database: it
receives a ``pd.DataFrame`` of forward-return data (produced by
``kronos_trading.alerts.forward_return_logger.get_phase9a_data`` and typically
exported to CSV) and tests the pre-registered hypothesis that a HAR breakout
bar's candle direction persists into the next 1/2/3 bars.

Public surface:

* :mod:`phase9a.direction_calculator` — hit rates + temporal-window splits.
* :mod:`phase9a.continuation_tester` — temporal stability + G1–G6 gates.
* :mod:`phase9a.dm_test` — one-sided Diebold-Mariano test vs a coin flip.
* :mod:`phase9a.phase9a_runner` — CLI that loads a CSV and prints the report.
"""
from __future__ import annotations

__all__ = [
    "direction_calculator",
    "continuation_tester",
    "dm_test",
    "phase9a_runner",
]
