"""Runner tests — offline, using synthetic candles."""
from __future__ import annotations

import os

import pandas as pd
import pytest

from sandbox.pattern_research import run_pattern_research as runner
from sandbox.pattern_research.tools.make_synthetic_candles import make_synthetic_candles


@pytest.fixture(scope="module")
def candles_by_asset():
    return {
        "BTC/USDT": make_synthetic_candles(4000, seed=1),
        "ETH/USDT": make_synthetic_candles(4000, seed=2, start_price=2000.0),
    }


def test_run_momentum_produces_report(candles_by_asset):
    report, evaluations = runner.run("momentum", list(candles_by_asset), 1,
                                     candles_by_asset, days=166)
    assert "# Pattern Research Results — momentum" in report
    assert len(evaluations) == 3
    for title, ev in evaluations:
        assert ev["n_events"] >= 0
        if not ev["skipped"]:
            assert ev["gates"]["verdict"] in ("SIGNAL FOUND", "CLOSED")
            assert set(ev["walk_forward"]) == set(candles_by_asset)


def test_run_all_covers_every_family(candles_by_asset):
    report, evaluations = runner.run("all", list(candles_by_asset), 1,
                                     candles_by_asset, days=166)
    titles = [t for t, _ in evaluations]
    assert any(t.startswith("momentum") for t in titles)
    assert any(t.startswith("candlestick") for t in titles)
    assert any(t.startswith("volume") for t in titles)
    assert any(t.startswith("time_of_day") for t in titles)
    assert "## Summary" in report and "| Signal | N | Hit rate | DM p | Verdict |" in report


def test_random_walk_data_should_not_pass_gates(candles_by_asset):
    """Sanity: a seeded random walk must not produce a 'SIGNAL FOUND'."""
    _report, evaluations = runner.run("all", list(candles_by_asset), 1,
                                      candles_by_asset, days=166)
    for title, ev in evaluations:
        if not ev["skipped"]:
            assert ev["gates"]["verdict"] == "CLOSED", title


def test_low_occurrence_pattern_is_skipped_and_documented():
    tiny = {"BTC/USDT": make_synthetic_candles(120, seed=3),
            "ETH/USDT": make_synthetic_candles(120, seed=4)}
    report, evaluations = runner.run("candlestick", list(tiny), 1, tiny, days=5)
    skipped = [(t, e) for t, e in evaluations if e["skipped"]]
    assert skipped, "expected at least one skipped pattern on a tiny sample"
    for title, ev in skipped:
        assert ev["gates"] is None
        assert "minimum" in ev["skip_reason"]
        assert "SKIPPED" in report


def test_time_of_day_is_evaluated_out_of_sample(candles_by_asset):
    ev = runner.evaluate_time_of_day(candles_by_asset, horizon=1)
    per_asset = ev["per_asset"]
    for asset, ctx in per_asset.items():
        assert ctx["n_train"] + ctx["n_test"] == len(candles_by_asset[asset])
        # every scored event must lie strictly inside the held-out window
        if ev["n_events"]:
            events = ev["results_df"]
            sub = events[events["asset"] == asset]
            if len(sub):
                assert str(sub["timestamp"].min()) >= ctx["test_start"]


def test_horizons_2_and_3_run(candles_by_asset):
    for horizon in (2, 3):
        _report, evaluations = runner.run("volume", list(candles_by_asset), horizon,
                                          candles_by_asset, days=166)
        assert evaluations


def test_cli_end_to_end_with_csv(tmp_path):
    btc = tmp_path / "btc.csv"
    eth = tmp_path / "eth.csv"
    make_synthetic_candles(2500, seed=5).to_csv(btc, index_label="timestamp")
    make_synthetic_candles(2500, seed=6, start_price=2000.0).to_csv(eth, index_label="timestamp")
    out = tmp_path / "results"
    code = runner.main([
        "--pattern", "momentum", "--asset", "both", "--horizon", "1",
        "--output", str(out), "--quiet",
        "--csv", f"BTC/USDT={btc},ETH/USDT={eth}",
    ])
    assert code == 0
    files = list(out.glob("pattern_research_momentum_both_h1_*.md"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "GATES" in text and "VERDICT" in text


def test_cli_single_asset(tmp_path):
    btc = tmp_path / "btc.csv"
    make_synthetic_candles(2500, seed=8).to_csv(btc, index_label="timestamp")
    out = tmp_path / "results"
    code = runner.main(["--pattern", "candlestick", "--asset", "BTC/USDT",
                        "--output", str(out), "--quiet", "--csv", f"BTC/USDT={btc}"])
    assert code == 0
    assert list(out.glob("pattern_research_candlestick_BTCUSDT_h1_*.md"))


def test_cli_reports_data_failure_cleanly(tmp_path, capsys):
    code = runner.main(["--pattern", "momentum", "--csv", "BTC/USDT=/nope/missing.csv",
                        "--output", str(tmp_path)])
    assert code == 2
    assert "Data load failed" in capsys.readouterr().err


def test_parse_csv_arg():
    assert runner._parse_csv_arg("BTC/USDT=a.csv, ETH/USDT=b.csv") == {
        "BTC/USDT": "a.csv", "ETH/USDT": "b.csv"}
    assert runner._parse_csv_arg(None) == {}
    with pytest.raises(ValueError):
        runner._parse_csv_arg("BTC/USDT")
