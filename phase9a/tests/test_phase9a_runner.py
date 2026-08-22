"""Unit tests for phase9a.phase9a_runner (pure core + isolated I/O provider)."""
from __future__ import annotations

import time

import pandas as pd
import pytest

from phase9a.phase9a_runner import (
    _build_parser,
    fetch_candle_history,
    format_results,
    run_analysis,
    save_results,
)


def _make_scenario(n_per_asset=40, continue_up=True):
    """Build synthetic breakout_rows + per-asset candles.

    Even-indexed bars are breakouts (open 100 -> close 110, UP). Odd bars are
    the t+1 forward bar; their close continues up (120) or reverses down (100)
    depending on ``continue_up``.
    """
    base = pd.Timestamp("2024-01-15T00:00:00Z")
    breakout_rows = []
    candles_by_asset = {}
    for asset in ("BTC/USDT", "ETH/USDT"):
        rows = []
        for k in range(n_per_asset):
            even = 2 * k
            ts_t = (base + pd.Timedelta(hours=even)).strftime("%Y-%m-%dT%H:%M:%SZ")
            ts_t1 = (base + pd.Timedelta(hours=even + 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            rows.append({"timestamp": ts_t, "open": 100.0, "high": 112.0,
                         "low": 99.0, "close": 110.0, "volume": 1.0})
            fclose = 120.0 if continue_up else 100.0
            rows.append({"timestamp": ts_t1, "open": 110.0, "high": 122.0,
                         "low": 99.0, "close": fclose, "volume": 1.0})
            breakout_rows.append({
                "timestamp": ts_t, "asset": asset, "timeframe": "1h",
                "har_predicted_range": 10.0, "actual_range": 30.0,
                "regime": "high", "breakout_flag": 1,
            })
        candles_by_asset[asset] = pd.DataFrame(rows)
    return pd.DataFrame(breakout_rows), candles_by_asset


# ---------------------------------------------------------------------------

class TestRunAnalysis:

    def test_signal_found_when_continuation_persists(self):
        br, candles = _make_scenario(n_per_asset=40, continue_up=True)
        res = run_analysis(br, candles, horizon=1)
        assert res["n_events"] == 80
        assert res["hit_rate"]["hit_rate"] == pytest.approx(1.0)
        assert res["dm"]["p_value"] < 0.05
        assert res["gates"]["verdict"] == "SIGNAL FOUND"
        assert res["gates"]["all_pass"] is True

    def test_closed_when_direction_reverses(self):
        br, candles = _make_scenario(n_per_asset=40, continue_up=False)
        res = run_analysis(br, candles, horizon=1)
        # 0% hit rate -> G1 fails, DM not significant -> CLOSED.
        assert res["hit_rate"]["hit_rate"] == pytest.approx(0.0)
        assert res["gates"]["verdict"] == "CLOSED"
        assert res["gates"]["all_pass"] is False

    def test_empty_breakout_rows(self):
        res = run_analysis(pd.DataFrame(), {}, horizon=1)
        assert res["n_events"] == 0
        assert res["gates"]["verdict"] == "CLOSED"

    def test_missing_candle_history_skips_asset(self):
        br, candles = _make_scenario(n_per_asset=40)
        del candles["ETH/USDT"]  # ETH has no candle history
        res = run_analysis(br, candles, horizon=1)
        # Only BTC analysed -> G3 (both assets) fails -> CLOSED.
        assert "ETH/USDT" not in res["hit_rate"]["by_asset"]
        assert res["gates"]["G3"] is False
        assert res["gates"]["verdict"] == "CLOSED"

    def test_horizon_carried_through(self):
        br, candles = _make_scenario(n_per_asset=40, continue_up=True)
        res = run_analysis(br, candles, horizon=2)
        assert res["horizon"] == 2


class TestFormatResults:

    def test_contains_expected_lines(self):
        br, candles = _make_scenario(n_per_asset=40, continue_up=True)
        res = run_analysis(br, candles, horizon=1)
        text = format_results(res)
        assert "PHASE 9A RESULTS" in text
        assert "Horizon: t+1" in text
        assert "VERDICT: SIGNAL FOUND" in text
        assert "Hit rate (BTC):" in text
        assert "Hit rate (ETH):" in text
        assert "G1 (hit rate > 55%): PASS" in text
        assert "DM statistic:" in text

    def test_handles_missing_asset(self):
        res = run_analysis(pd.DataFrame(), {}, horizon=1)
        text = format_results(res)
        assert "Hit rate (BTC): N/A" in text
        assert "VERDICT: CLOSED" in text


class TestSaveResults:

    def test_writes_markdown_file(self, tmp_path):
        br, candles = _make_scenario(n_per_asset=40, continue_up=True)
        res = run_analysis(br, candles, horizon=1)
        text = format_results(res)
        out = tmp_path / "report.md"
        path = save_results(text, str(out))
        assert path == str(out)
        content = out.read_text()
        assert "# Phase 9A Results" in content
        assert "SIGNAL FOUND" in content


class TestFetchCandleHistory:

    def test_drops_forming_bar_and_converts(self):
        class FakeExchange:
            def fetch_ohlcv(self, symbol, timeframe, limit):
                now_ms = int(time.time() * 1000)
                # 3 closed bars + 1 forming bar (open time within the last hour).
                return [
                    [now_ms - 4 * 3_600_000, 100, 105, 99, 102, 10],
                    [now_ms - 3 * 3_600_000, 102, 106, 100, 104, 11],
                    [now_ms - 2 * 3_600_000, 104, 107, 101, 103, 12],
                    [now_ms, 103, 108, 102, 106, 13],  # still forming
                ]

        df = fetch_candle_history("BTC/USDT", "1h", n=10, exchange=FakeExchange())
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert len(df) == 3  # forming bar dropped
        assert df["timestamp"].iloc[0].endswith("Z")


class TestCli:

    def test_defaults(self):
        args = _build_parser().parse_args([])
        assert args.horizon == 1
        assert args.asset == "both"
        assert args.dry_run is False

    def test_horizon_choices(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--horizon", "5"])
