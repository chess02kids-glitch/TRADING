"""Phase 8 (F-01) - public Binance USD-M funding-rate acquisition (read-only).

The pre-registered Phase 8 experiment is FUNDING-ONLY. It uses exactly two
external features derived from the settled perpetual funding rate:

    funding_mean_24h     = mean of settled funding rates in (t-24h, t]
    abs_funding_mean_24h = mean of |funding rate| in (t-24h, t]

Nothing else (no open interest, no basis/premium, no liquidations, no
long/short ratios). This module acquires and aligns funding data using ONLY
public market-data endpoints (no API key, no trading, no order endpoints).

The open-interest and premium-index fetchers below are retained as FUTURE
utilities only; they are never called by the F-01 execution path.
"""
from __future__ import annotations

import json
import time
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DERIVATIVES_DIR = PROJECT_ROOT / "data" / "derivatives"

# Binance USD-M public market-data REST base (no auth).
FAPI_BASE = "https://fapi.binance.com"

# Funding settles every 8 hours on Binance USD-M perpetuals. A settled rate
# older than one funding interval at prediction time is a genuine gap (skip).
FUNDING_INTERVAL_MS = 8 * 3600 * 1000
FUNDING_WINDOW_MS = 24 * 3600 * 1000  # 24-hour funding window for F-01 features


class DerivativesDataError(RuntimeError):
    """Raised when funding data cannot be acquired or aligned safely."""


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
                       end_ms: Optional[int] = None,
                       limit: int = 1000) -> List[Dict[str, Any]]:
    """Paginated settled funding-rate history (Binance USD-M).

    Binance's ``GET /fapi/v1/fundingRate`` returns at most ``limit`` (max 1000)
    rows per request. This fetches the COMPLETE history for ``[start_ms, end_ms]``
    by advancing chronologically:

    1. request from ``start_ms`` with ``limit`` rows;
    2. read the returned funding timestamps;
    3. advance the next request to (latest returned funding time + 1 ms);
    4. continue until the requested range is exhausted or the endpoint returns
       no rows.

    Guarantees: no silent truncation at 1000, no duplicate observations across
    pages (dedup by funding timestamp), chronological ordering, no future
    observations (each returned ``fundingTime`` must be within
    ``[start_ms, end_ms]``), no forward-fill/synthesis.

    ``end_ms`` defaults to the current time (point-in-time ceiling).
    """
    end_ms = end_ms if end_ms is not None else int(time.time() * 1000)
    limit = max(1, min(int(limit), 1000))

    collected: Dict[int, float] = {}
    cursor = start_ms
    seen_max: Optional[int] = None
    while cursor <= end_ms:
        page = _http_get(FAPI_BASE + "/fapi/v1/fundingRate",
                         {"symbol": symbol, "startTime": cursor,
                          "endTime": end_ms, "limit": limit})
        if not page:
            break

        page_ts: List[int] = []
        for r in page:
            ts = int(r["fundingTime"])
            if start_ms <= ts <= end_ms:  # point-in-time: never future, never pre-start
                collected[ts] = float(r["fundingRate"])
                page_ts.append(ts)
        if not page_ts:
            break

        last_ts = max(page_ts)
        if seen_max is not None and last_ts <= seen_max:
            # no forward progress (endpoint returned only already-seen rows)
            break
        seen_max = last_ts
        cursor = last_ts + 1

    return [{"timestamp_ms": ts, "funding_rate": rate, "kind": "funding"}
            for ts, rate in sorted(collected.items())]


def fetch_funding_only(symbol: str, start_ms: int,
                       end_ms: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    """F-01 data acquisition: complete settled funding history only."""
    return {"funding": fetch_funding_rate(symbol, start_ms, end_ms=end_ms)}


# --------------------------------------------------------------------------- #
# FUTURE utilities (NOT used by the funding-only F-01 experiment)
# --------------------------------------------------------------------------- #
def fetch_open_interest(symbol: str = "BTCUSDT", period: str = "1h",
                        start_ms: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
    """[future] Aggregate open-interest history. Not used by F-01."""
    rows = _http_get(FAPI_BASE + "/futures/data/openInterestHist",
                     {"symbol": symbol, "period": period,
                      "startTime": start_ms, "limit": limit})
    return [{"timestamp_ms": int(r["timestamp"]),
             "open_interest": float(r["sumOpenInterest"]),
             "kind": "open_interest"} for r in rows]


def fetch_premium_index(symbol: str = "BTCUSDT") -> List[Dict[str, Any]]:
    """[future] Premium index snapshot (mark vs index). Not used by F-01."""
    rows = _http_get(FAPI_BASE + "/fapi/v1/premiumIndex", {"symbol": symbol})
    for r in rows:
        return [{
            "timestamp_ms": int(time.time() * 1000),
            "mark_price": float(r["markPrice"]),
            "index_price": float(r["indexPrice"]),
            "basis": float(r["markPrice"]) / float(r["indexPrice"]) - 1.0,
            "kind": "basis",
        }]
    return []


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
def save_derivatives(symbol: str, data: Dict[str, List[Dict[str, Any]]],
                     directory: Optional[Path] = None) -> Path:
    """Persist fetched funding data as JSON (gitignored under data/derivatives)."""
    d = directory or DERIVATIVES_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / ("%s_derivatives.json" % symbol)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return path


def load_derivatives(symbol: str,
                     directory: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Load persisted funding data for one symbol (funding-only payload)."""
    d = directory or DERIVATIVES_DIR
    path = d / ("%s_derivatives.json" % symbol)
    if not path.exists():
        raise DerivativesDataError(
            "funding data for %s not present at %s (run --fetch first)"
            % (symbol, path))
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# F-01 feature derivation (point-in-time, no forward-fill)
# --------------------------------------------------------------------------- #
def funding_features_24h(funding_rows: List[Dict[str, Any]],
                         query_timestamps: List[int],
                         window_ms: int = FUNDING_WINDOW_MS,
                         max_staleness_ms: int = FUNDING_INTERVAL_MS
                         ) -> Dict[str, List[Optional[float]]]:
    """Compute funding_mean_24h and abs_funding_mean_24h for each query time t.

    Point-in-time rules (no future, no forward-fill, no interpolation):

    * uses only settled funding rates with ``funding_time <= t``;
    * the window is ``(t - window_ms, t]`` (24 hours);
    * if the most recent settled rate is older than ``max_staleness_ms`` before
      ``t`` (a missing settlement), the value is None (skip);
    * if there are no settled rates in the window, the value is None (skip).

    Returns ``{'funding_mean_24h': [...], 'abs_funding_mean_24h': [...]}`` with
    None for skipped timestamps.
    """
    rows = sorted(funding_rows, key=lambda r: r["timestamp_ms"])
    ts_arr = [r["timestamp_ms"] for r in rows]
    rate_arr = [float(r["funding_rate"]) for r in rows]

    mean_out: List[Optional[float]] = []
    abs_out: List[Optional[float]] = []
    for t in query_timestamps:
        j = bisect_right(ts_arr, t)  # first index with ts > t
        if j == 0:
            mean_out.append(None)
            abs_out.append(None)
            continue
        last_settled = ts_arr[j - 1]
        if t - last_settled > max_staleness_ms:
            # genuine gap: a settlement is missing -> skip, do not forward-fill
            mean_out.append(None)
            abs_out.append(None)
            continue
        lo = bisect_right(ts_arr, t - window_ms)  # first index with ts > t-window
        window_rates = rate_arr[lo:j]
        if not window_rates:
            mean_out.append(None)
            abs_out.append(None)
            continue
        mean_out.append(sum(window_rates) / len(window_rates))
        abs_out.append(sum(abs(r) for r in window_rates) / len(window_rates))

    return {"funding_mean_24h": mean_out, "abs_funding_mean_24h": abs_out}
