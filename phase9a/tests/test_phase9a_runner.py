"""Tests for phase9a.phase9a_runner (CLI core: load, analyze, format, save)."""
from __future__ import annotations

import pandas as pd
import pytest

from phase9a.phase9a_runner import (
    analyze,
    filter_asset,
    format_report,
    load_data,
    main,
)


def _csv(tmp_path, n_each=20, hit=True):
    rows = []
    b = pd.Timestamp("2024-01-01T00:00:00Z")
    i = 0
    for _ in range(n_each):
        for asset in ("BTC/USDT", "ETH/USDT"):
            ts = (b + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            rows.append({"breakout_timestamp": ts, "asset": asset,
                         "breakout_direction": 1, "horizon": 1,
                         "forward_direction": 1 if hit else -1,
                         "forward_return": 0.01, "breakout_close_price": 100.0})
            i += 1
    df = pd.DataFrame(rows)
    path = tmp_path / "phase9a.csv"
    df.to_csv(path, index=False)
    return str(path), df


class TestLoadAndFilter:

    def test_load_csv(self, tmp_path):
        path, df = _csv(tmp_path, n_each=5)
        loaded = load_data(path)
        assert len(loaded) == len(df)
        assert "breakout_direction" in loaded.columns

    def test_filter_asset(self, tmp_path):
        _, df = _csv(tmp_path, n_each=5)
        assert len(filter_asset(df, "BTC/USDT")) == 5
        assert len(filter_asset(df, "both")) == 10


class TestAnalyzeAndFormat:

    def test_signal_found(self, tmp_path):
        _, df = _csv(tmp_path, n_each=35, hit=True)
        res = analyze(df, "both", horizon=1)
        assert res["gates"]["verdict"] == "SIGNAL FOUND"
        text = format_report(res)
        assert "PHASE 9A RESULTS" in text
        assert "VERDICT: SIGNAL FOUND" in text
        assert "Hit rate (BTC):" in text
        assert "G6 (n >= 30):" in text

    def test_closed_report(self, tmp_path):
        _, df = _csv(tmp_path, n_each=35, hit=False)
        res = analyze(df, "both", horizon=1)
        assert "VERDICT: CLOSED" in format_report(res)


class TestMain:

    def test_main_signal_found(self, tmp_path, capsys):
        path, _ = _csv(tmp_path, n_each=35, hit=True)
        out = tmp_path / "report.md"
        rc = main(["--data-file", path, "--asset", "both",
                   "--horizon", "1", "--output", str(out)])
        captured = capsys.readouterr()
        assert rc == 0
        assert "VERDICT: SIGNAL FOUND" in captured.out
        assert out.exists() and "SIGNAL FOUND" in out.read_text(encoding="utf-8")

    def test_main_not_enough_data(self, tmp_path, capsys):
        path, _ = _csv(tmp_path, n_each=5, hit=True)  # 5 each < 30
        rc = main(["--data-file", path, "--asset", "both"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "Not enough data yet" in captured.out
