"""Agent 1 — Pattern Research Sandbox.

A self-contained research playground that is **completely separate** from the
main ``kronos_trading`` system:

* no database / Supabase connection,
* no secrets or API keys (KuCoin public CCXT endpoints only),
* nothing here is imported by the production codebase.

Sub-modules:

* :mod:`sandbox.pattern_research.data_loader` — 1h OHLCV loading (KuCoin
  public API, 730 days, disk cache, offline CSV support).
* :mod:`sandbox.pattern_research.patterns` — the four pattern families
  (momentum, candlestick, time-of-day, volume spike).
* :mod:`sandbox.pattern_research.validator` — Diebold-Mariano test, G1–G6
  gate checks and walk-forward stability (same criteria as Phase 9A).
* :mod:`sandbox.pattern_research.run_pattern_research` — CLI runner.
"""

__all__ = ["data_loader", "patterns", "validator"]
