"""Phase 9A — breakout-direction continuation analysis (pure statistics).

This package tests the pre-registered hypothesis that the candle direction of a
HAR *breakout* bar (``actual_range > 2 x har_predicted_range``) persists into
the next 1/2/3 bars. It is a *pure statistical* test: no machine-learning
models, no live trading, no writes to the ``har_predictions`` table.

Public surface:

* :mod:`phase9a.direction_calculator` — past-only breakout direction + forward
  returns from candle history.
* :mod:`phase9a.continuation_tester` — hit rates, temporal stability,
  degradation flag and the G1–G6 gate checks.
* :mod:`phase9a.dm_test` — Diebold-Mariano test of the directional signal vs a
  50/50 random baseline (one-sided, HAC / Newey-West standard errors).
* :mod:`phase9a.phase9a_runner` — CLI orchestrator that ties it together.
"""
from __future__ import annotations

__all__ = [
    "direction_calculator",
    "continuation_tester",
    "dm_test",
    "phase9a_runner",
]
