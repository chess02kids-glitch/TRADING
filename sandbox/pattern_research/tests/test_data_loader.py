"""Data-loader tests — fake CCXT exchange, no network."""
from __future__ import annotations

import os

import pandas as pd
import pytest

from sandbox.pattern_research import data_loader


class FakeExchange:
    """Minimal CCXT-like stub returning hourly rows from ``start_ms``."""

    def __init__(self, start_ms, n, bar_ms=3_600_000, extra_rows=None):
        self.rows = [[start_ms + i * bar_ms, 100.0 + i, 101.0 + i, 99.0 + i,
                      100.5 + i, 10.0 + i] for i in range(n)]
        if extra_rows:
            self.rows.extend(extra_rows)
        self.calls = []

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls.append((symbol, timeframe, since, limit))
        usable = [r for r in self.rows if r and r[0] is not None]
        rows = [r for r in usable if since is None or r[0] >= since]
        return rows[: (limit or len(rows))]


NOW = 1_700_000_000_000 // 3_600_000 * 3_600_000  # aligned to the hour


def test_fetch_paginates_and_drops_unclosed_bar():
    start = NOW - 200 * 3_600_000
    # last row opens exactly at NOW -> still forming -> must be dropped
    ex = FakeExchange(start, 201)
    df = data_loader.fetch_ohlcv("BTC/USDT", days=30, exchange=ex, now_ms=NOW, limit=50)
    assert len(df) == 200
    assert df.index[-1] == pd.to_datetime(NOW - 3_600_000, unit="ms", utc=True)
    assert len(ex.calls) > 1                     # paginated
    assert ex.calls[0][0] == "BTC/USDT"


def test_symbol_normalisation_and_dedupe():
    start = NOW - 100 * 3_600_000
    dup = [[start, 1.0, 2.0, 0.5, 1.5, 7.0]]      # duplicate ts, last wins
    ex = FakeExchange(start, 100, extra_rows=dup)
    df = data_loader.fetch_ohlcv("BTC", days=10, exchange=ex, now_ms=NOW)
    assert ex.calls[0][0] == "BTC/USDT"
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert df.iloc[0]["close"] == 1.5


def test_malformed_rows_are_skipped():
    start = NOW - 100 * 3_600_000
    ex = FakeExchange(start, 100, extra_rows=[[start + 500, "x"], None])
    df = data_loader.fetch_ohlcv("ETH/USDT", days=10, exchange=ex, now_ms=NOW)
    assert len(df) == 100  # the 2 junk rows are dropped, the 100 real bars survive


def test_insufficient_candles_raises():
    start = NOW - 10 * 3_600_000
    ex = FakeExchange(start, 10)
    with pytest.raises(data_loader.InsufficientCandlesError):
        data_loader.fetch_ohlcv("BTC/USDT", days=1, exchange=ex, now_ms=NOW)


def test_bad_arguments():
    with pytest.raises(ValueError):
        data_loader.fetch_ohlcv("BTC/USDT", timeframe="7m")
    with pytest.raises(ValueError):
        data_loader.fetch_ohlcv("BTC/USDT", days=0)


def test_csv_round_trip_and_cache(tmp_path):
    start = NOW - 300 * 3_600_000
    ex = FakeExchange(start, 300)
    cache_dir = str(tmp_path / "cache")
    first = data_loader.load_candles("BTC/USDT", days=20, exchange=ex, now_ms=NOW,
                                     cache_dir=cache_dir)
    path = data_loader.cache_path("BTC/USDT", "1h", 20, cache_dir)
    assert os.path.exists(path)

    # second call must hit the cache (no further exchange calls)
    n_calls = len(ex.calls)
    second = data_loader.load_candles("BTC/USDT", days=20, exchange=ex, now_ms=NOW,
                                      cache_dir=cache_dir)
    assert len(ex.calls) == n_calls
    pd.testing.assert_frame_equal(first, second)


def test_load_candles_from_explicit_csv(tmp_path):
    from sandbox.pattern_research.tools.make_synthetic_candles import make_synthetic_candles
    path = str(tmp_path / "synth.csv")
    make_synthetic_candles(200, seed=2).to_csv(path, index_label="timestamp")
    df = data_loader.load_candles("BTC/USDT", csv=path)
    assert list(df.columns) == data_loader.OHLCV_COLUMNS
    assert isinstance(df.index, pd.DatetimeIndex) and str(df.index.tz) == "UTC"


def test_load_csv_rejects_missing_columns(tmp_path):
    path = str(tmp_path / "bad.csv")
    pd.DataFrame({"timestamp": ["2024-01-01T00:00:00Z"], "close": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        data_loader.load_csv(path)


def test_sandbox_imports_no_production_or_db_modules():
    """Rule: the sandbox must not import the main system, Supabase or secrets."""
    import pathlib
    root = pathlib.Path(data_loader.__file__).parent
    banned = ("kronos_trading", "supabase", "psycopg", "sqlalchemy", "dotenv", "os.environ")
    for py in root.rglob("*.py"):
        if "tests" in py.parts:
            continue
        code_lines = [ln.strip() for ln in py.read_text(encoding="utf-8").splitlines()
                      if ln.strip().startswith(("import ", "from "))
                      or "os.environ" in ln or "getenv" in ln]
        for token in banned:
            for line in code_lines:
                assert token not in line, f"{py.name} references {token}: {line}"
