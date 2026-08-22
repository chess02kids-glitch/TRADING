"""Execution audit log — signals, attempts, results, and skips.

A completely separate, in-memory + optional plain-text audit trail from the
``har_predictions`` research store. Every signal the execution layer sees,
every order it tries to place, every outcome and every skip is recorded with a
UTC timestamp and an ``event_type`` tag, so the full paper-trading decision
trail is replayable via :func:`get_log`.

* :func:`get_log` returns the structured list of all events.
* Plain-text lines are also appended to a log file when a path is configured
  via :func:`configure` (best-effort; defaults to in-memory only).
* :func:`reset` clears the in-memory log (test isolation).

Nothing here touches the live HAR bot or any database.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# In-memory structured log (the source of get_log()).
_LOG: List[Dict[str, Any]] = []
# Optional plain-text log file path (None = in-memory only).
_LOG_PATH: Optional[str] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def configure(log_path: Optional[str]) -> None:
    """Enable/disable plain-text file logging (``None`` = in-memory only)."""
    global _LOG_PATH
    _LOG_PATH = log_path


def reset() -> None:
    """Clear the in-memory log (used by tests for isolation)."""
    _LOG.clear()


def _append(event_type: str, **fields: Any) -> None:
    entry: Dict[str, Any] = {
        "timestamp": _now_iso(),
        "event_type": event_type,
    }
    entry.update(fields)
    _LOG.append(entry)
    if _LOG_PATH:
        try:
            with open(_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:  # pragma: no cover - best-effort file logging
            logger.warning("execution_logger file write failed: %s", exc)


def log_signal(signal: Any) -> None:
    """Record an incoming signal (dataclass or dict-like)."""
    _append("signal", signal=_jsonable(signal))


def log_skip(reason: str, signal: Any) -> None:
    """Record a skipped trade with the reason and the originating signal."""
    _append("skip", reason=str(reason), signal=_jsonable(signal))


def log_order_attempt(params: Any) -> None:
    """Record an order attempt (the built OrderParams)."""
    _append("order_attempt", order_params=_jsonable(params))


def log_order_result(result: Optional[Dict[str, Any]], success: bool) -> None:
    """Record an order outcome (CCXT order dict, or None) and success flag."""
    _append("order_result", result=_jsonable(result), success=bool(success))


def get_log() -> List[Dict[str, Any]]:
    """Return a copy of the structured event list (oldest first)."""
    return [dict(e) for e in _LOG]
