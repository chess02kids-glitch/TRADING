"""Phase 8 F-01 - funding pagination regression tests.

Uses a mocked Binance funding endpoint (no network) to prove that
``fetch_funding_rate`` retrieves the COMPLETE settled funding history via
chronological pagination, with no duplicates, no truncation, no future
observations, and deterministic output. F-01 funding-only behavior is asserted
to remain unchanged.
"""
import time

import pytest

import kronos_trading.derivatives_data as dd

H8 = 8 * 3600 * 1000
BASE = 1_700_000_000_000


class MockFundingAPI:
    """Simulates Binance GET /fapi/v1/fundingRate with limit-per-request."""

    def __init__(self, n_events, base=BASE, step=H8, limit_cap=1000):
        self.n_events = n_events
        self.base = base
        self.step = step
        self.limit_cap = limit_cap
        self.calls = []
        self.events = [
            {"fundingTime": base + i * step,
             "fundingRate": str(round((i % 11) * 1e-4, 10))}
            for i in range(n_events)
        ]

    def __call__(self, url, params):
        self.calls.append(dict(params))
        s = params["startTime"]
        e = params["endTime"]
        lim = params["limit"]
        rows = [ev for ev in self.events if s <= ev["fundingTime"] <= e]
        return rows[:lim]


@pytest.fixture
def mock_api(monkeypatch):
    def install(n_events, base=BASE, step=H8, limit_cap=1000):
        mock = MockFundingAPI(n_events, base, step, limit_cap)
        monkeypatch.setattr(dd, "_http_get", mock)
        return mock
    return install


def _ts(rows):
    return [r["timestamp_ms"] for r in rows]


# --------------------------------------------------------------------------- #
# 1. Pagination across multiple pages
# --------------------------------------------------------------------------- #
def test_pagination_across_multiple_pages(mock_api):
    n = 2500
    mock = mock_api(n)
    rows = dd.fetch_funding_rate("BTCUSDT", start_ms=BASE,
                                 end_ms=BASE + (n - 1) * H8)
    assert len(rows) == n
    assert len(mock.calls) == 3  # 1000 + 1000 + 500


# --------------------------------------------------------------------------- #
# 2. Correct timestamp advancement
# --------------------------------------------------------------------------- #
def test_timestamp_advancement(mock_api):
    n = 2500
    mock = mock_api(n)
    dd.fetch_funding_rate("BTCUSDT", start_ms=BASE,
                          end_ms=BASE + (n - 1) * H8)
    # first page starts at the requested start_ms
    assert mock.calls[0]["startTime"] == BASE
    for prev, cur in zip(mock.calls, mock.calls[1:]):
        # each subsequent page starts one ms after the previous page's last row
        prev_rows = [ev for ev in mock.events
                     if prev["startTime"] <= ev["fundingTime"] <= prev["endTime"]][:prev["limit"]]
        expected_next = max(r["fundingTime"] for r in prev_rows) + 1
        assert cur["startTime"] == expected_next


# --------------------------------------------------------------------------- #
# 3. No duplicate observations across pages
# --------------------------------------------------------------------------- #
def test_no_duplicates(mock_api):
    n = 3000
    mock_api(n)
    rows = dd.fetch_funding_rate("BTCUSDT", start_ms=BASE,
                                 end_ms=BASE + (n - 1) * H8)
    ts = _ts(rows)
    assert len(ts) == len(set(ts)) == n


# --------------------------------------------------------------------------- #
# 4. Chronological ordering
# --------------------------------------------------------------------------- #
def test_chronological_ordering(mock_api):
    n = 2100
    mock_api(n)
    rows = dd.fetch_funding_rate("BTCUSDT", start_ms=BASE,
                                 end_ms=BASE + (n - 1) * H8)
    ts = _ts(rows)
    assert ts == sorted(ts)


# --------------------------------------------------------------------------- #
# 5. Full-history retrieval beyond 1000 records
# --------------------------------------------------------------------------- #
def test_full_history_beyond_1000(mock_api):
    n = 2190  # ~730 days of 8h funding events
    mock_api(n)
    rows = dd.fetch_funding_rate("BTCUSDT", start_ms=BASE,
                                 end_ms=BASE + (n - 1) * H8)
    assert len(rows) == 2190  # not truncated at 1000


# --------------------------------------------------------------------------- #
# 6. Empty final page handling
# --------------------------------------------------------------------------- #
def test_empty_final_page(mock_api):
    n = 1500
    mock = mock_api(n)
    # end_ms far beyond the last event -> after all events are fetched, one
    # further request returns [] and the loop must terminate cleanly.
    end_ms = BASE + (n + 500) * H8
    rows = dd.fetch_funding_rate("BTCUSDT", start_ms=BASE, end_ms=end_ms)
    assert len(rows) == n
    assert len(mock.calls) == 3  # 1000 + 500 + 1 empty page


# --------------------------------------------------------------------------- #
# 7. Exact boundary behavior
# --------------------------------------------------------------------------- #
def test_exact_boundary(mock_api):
    n = 1000  # exactly one page worth
    mock = mock_api(n)
    end_ms = BASE + (n - 1) * H8
    rows = dd.fetch_funding_rate("BTCUSDT", start_ms=BASE, end_ms=end_ms)
    assert len(rows) == 1000
    assert max(_ts(rows)) == end_ms
    # the very next page would start at end_ms + 1 > end_ms -> loop ended
    assert len(mock.calls) == 1


def test_end_ms_is_respected(mock_api):
    n = 3000
    mock = mock_api(n)
    end_ms = BASE + 1500 * H8  # stop mid-history
    rows = dd.fetch_funding_rate("BTCUSDT", start_ms=BASE, end_ms=end_ms)
    assert len(rows) == 1501  # events at BASE .. BASE+1500*H8 inclusive
    assert all(ts <= end_ms for ts in _ts(rows))


# --------------------------------------------------------------------------- #
# 8. Deterministic output
# --------------------------------------------------------------------------- #
def test_deterministic_output(mock_api):
    n = 2000
    mock_api(n)
    r1 = dd.fetch_funding_rate("BTCUSDT", start_ms=BASE,
                               end_ms=BASE + (n - 1) * H8)
    mock_api(n)  # reset calls; a fresh mock returns identical data
    r2 = dd.fetch_funding_rate("BTCUSDT", start_ms=BASE,
                               end_ms=BASE + (n - 1) * H8)
    assert r1 == r2


# --------------------------------------------------------------------------- #
# 9. No future observations
# --------------------------------------------------------------------------- #
def test_no_future_observations(mock_api):
    n = 500
    mock = mock_api(n)
    end_ms = BASE + (n - 1) * H8
    rows = dd.fetch_funding_rate("BTCUSDT", start_ms=BASE, end_ms=end_ms)
    assert all(BASE <= ts <= end_ms for ts in _ts(rows))
    # a future settlement injected into the mock must NOT appear
    future = BASE + (n + 50) * H8
    mock.events.append({"fundingTime": future, "fundingRate": "0.999"})
    rows2 = dd.fetch_funding_rate("BTCUSDT", start_ms=BASE, end_ms=end_ms)
    assert all(ts <= end_ms for ts in _ts(rows2))
    assert len(rows2) == n


# --------------------------------------------------------------------------- #
# 10. F-01 funding-only behavior unchanged
# --------------------------------------------------------------------------- #
def test_fetch_funding_only_shape(mock_api):
    n = 700
    mock_api(n)
    data = dd.fetch_funding_only("BTCUSDT", start_ms=BASE,
                                 end_ms=BASE + (n - 1) * H8)
    assert set(data.keys()) == {"funding"}
    assert len(data["funding"]) == n
    assert all(r["kind"] == "funding" for r in data["funding"])


def test_funding_only_feeds_feature_builder(mock_api, monkeypatch):
    """End-to-end: paginated funding -> F-01 features still work (no OI/basis)."""
    from kronos_trading.derivatives_volatility import build_derivatives_features
    from kronos_trading.types import Candle
    import numpy as np

    n_funding = 2200
    mock = mock_api(n_funding)
    funding = dd.fetch_funding_only("BTCUSDT", start_ms=BASE,
                                    end_ms=BASE + (n_funding - 1) * H8)

    n_candles = 600
    rng = np.random.default_rng(0)
    candles = []
    close = 100.0
    for i in range(n_candles):
        r = max(0.05, 0.9 * 1.0 + rng.normal(0, 0.03))
        close = close * (1.0 + rng.normal(0, 0.001))
        t = BASE + i * 3600 * 1000
        candles.append(Candle(t, close - r / 2, close + r / 2,
                              close - r / 2, close, 1000.0))

    out = build_derivatives_features(candles, funding["funding"], 3600 * 1000)
    assert out["missing"] == 0  # full funding coverage -> no funding gaps
    assert out["valid"][24:].sum() > 0
    assert out["X"].shape[1] == 5  # HAR(3) + funding_mean_24h + abs_funding_mean_24h
