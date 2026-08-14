"""Phase 8 - public Binance USD-M derivatives data acquisition (read-only).

Fetches the three pre-registered derivatives features for the frozen HAR
experiment, using ONLY public market-data endpoints (no API key, no trading,
no order endpoints):

1. ``funding_t``  - last SETTLED perpetual funding rate (funding_time <= t)
                   via ``GET /fapi/v1/fundingRate``.
2. ``oi_chg22_t`` - 22-bar log change in aggregate open interest via
                   ``GET /futures/data/openInterestHist``.
3. ``basis_t``    - perpetual basis/premium = mark_price / spot_index - 1 via
                   ``GET /fapi/v1/premiumIndex``.

Every row is stamped with the exchange timestamp at which the value was
actually available. This module only stores/aligns data; it never computes a
forecast and never mutates the verified OHLCV dataset.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .types import Candle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DERIVATIVES_DIR = PROJECT_ROOT / "data" / "derivatives"

# Binance USD-M public market-data REST base (no auth).
FAPI_BASE = "https://fapi.binance.com"


class DerivativesDataError(RuntimeError):
    """Raised when derivatives data cannot be acquired or aligned safely."""


def _http_get(url: str, params: Dict[str, Any], timeout: int = 20) -> Any:
    import urllib.parse
    import urllib.request
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request("%s?%s" % (url, qs), headers={
        "User-Agent": "kronos-trading-research/0.1",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_funding_rate(symbol: str = "BTCUSDT", start_ms: int = 0,
                       limit: int = 1000) -> List[Dict[str, Any]]:
    """Funding-rate history (settled). Each row: fundingTime, fundingRate."""
    rows = _http_get(FAPI_BASE + "/fapi/v1/fundingRate",
                     {"symbol": symbol, "startTime": start_ms, "limit": limit})
    out = []
    for r in rows:
        out.append({
            "timestamp_ms": int(r["fundingTime"]),
            "funding_rate": float(r["fundingRate"]),
            "kind": "funding",
        })
    return out


def fetch_open_interest(symbol: str = "BTCUSDT", period: str = "1h",
                        start_ms: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
    """Aggregate open-interest history. Each row: timestamp, sumOpenInterest(Value)."""
    rows = _http_get(FAPI_BASE + "/futures/data/openInterestHist",
                     {"symbol": symbol, "period": period,
                      "startTime": start_ms, "limit": limit})
    out = []
    for r in rows:
        out.append({
            "timestamp_ms": int(r["timestamp"]),
            "open_interest": float(r["sumOpenInterest"]),
            "kind": "open_interest",
        })
    return out


def fetch_premium_index(symbol: str = "BTCUSDT") -> List[Dict[str, Any]]:
    """Current premium index (mark price vs index). Point-in-time snapshot."""
    rows = _http_get(FAPI_BASE + "/fapi/v1/premiumIndex", {"symbol": symbol})
    for r in rows:
        return [{
            "timestamp_ms": int(time.time() * 1000),  # fetched now
            "mark_price": float(r["markPrice"]),
            "index_price": float(r["indexPrice"]),
            "basis": float(r["markPrice"]) / float(r["indexPrice"]) - 1.0,
            "kind": "basis",
        }]
    return []


def fetch_derivatives(symbol: str = "BTCUSDT", period: str = "1h",
                      start_ms: int = 0) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch all three derivatives series for one USD-M symbol."""
    return {
        "funding": fetch_funding_rate(symbol, start_ms),
        "open_interest": fetch_open_interest(symbol, period, start_ms),
        "basis": fetch_premium_index(symbol),
    }


def save_derivatives(symbol: str, data: Dict[str, List[Dict[str, Any]]],
                     directory: Optional[Path] = None) -> Path:
    """Persist fetched derivatives data as JSON (gitignored under data/derivatives)."""
    d = directory or DERIVATIVES_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / ("%s_derivatives.json" % symbol)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return path


def load_derivatives(symbol: str, directory: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
    d = directory or DERIVATIVES_DIR
    path = d / ("%s_derivatives.json" % symbol)
    if not path.exists():
        raise DerivativesDataError(
            "derivatives data for %s not present at %s (run --fetch first)"
            % (symbol, path))
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Point-in-time alignment
# --------------------------------------------------------------------------- #
def as_of_series(rows: List[Dict[str, Any]], timestamps: List[int],
                 value_key: str,
                 max_staleness_ms: Optional[int] = None) -> List[Optional[float]]:
    """Return the most recent value with ``timestamp_ms <= t`` for each t.

    No forward-fill across a missing required observation: if the most recent
    observation is older than ``max_staleness_ms`` before ``t`` (a genuine gap),
    the value is None (a skip). With ``max_staleness_ms=None``, the last
    observation <= t is used (correct for a natively coarse but complete series).
    Future observations are never used.
    """
    sorted_rows = sorted(rows, key=lambda r: r["timestamp_ms"])
    ts_arr = [r["timestamp_ms"] for r in sorted_rows]
    val_arr = [r[value_key] for r in sorted_rows]
    out: List[Optional[float]] = []
    idx = 0
    for t in timestamps:
        while idx < len(ts_arr) and ts_arr[idx] <= t:
            idx += 1
        if idx == 0:
            out.append(None)
            continue
        src_ts = ts_arr[idx - 1]
        if max_staleness_ms is not None and (t - src_ts) > max_staleness_ms:
            out.append(None)  # genuine gap: do not forward-fill a stale value
        else:
            out.append(val_arr[idx - 1])
    return out


def align_derivatives(data: Dict[str, List[Dict[str, Any]]],
                      timestamps: List[int],
                      max_staleness_ms: Optional[Dict[str, int]] = None
                      ) -> Dict[str, List[Optional[float]]]:
    """Align funding / open-interest / basis to query timestamps (point-in-time).

    ``funding`` = last settled rate with fundingTime <= t (staleness bounded by
    the funding interval: a missing settled interval is a skip).
    ``open_interest`` / ``basis`` = last snapshot with timestamp <= t, bounded
    by their observation cadence (a missing snapshot is a skip, never
    forward-filled).
    """
    s = max_staleness_ms or {}
    return {
        "funding": as_of_series(data["funding"], timestamps, "funding_rate",
                                s.get("funding")),
        "open_interest": as_of_series(data["open_interest"], timestamps,
                                      "open_interest", s.get("open_interest")),
        "basis": as_of_series(data["basis"], timestamps, "basis", s.get("basis")),
    }


def oi_log_change_22(oi: List[Optional[float]]) -> List[Optional[float]]:
    """22-bar log change of open interest: log(OI_t / OI_{t-22}).

    Returns None where either value is missing (never interpolated).
    """
    import math
    out: List[Optional[float]] = [None] * len(oi)
    for t in range(22, len(oi)):
        a, b = oi[t], oi[t - 22]
        if a is None or b is None or a <= 0 or b <= 0:
            out[t] = None
        else:
            out[t] = math.log(a / b)
    return out
