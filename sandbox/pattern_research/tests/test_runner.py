"""Runner tests — offline, using synthetic candles."""
from __future__ import annotations

import logging
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


# --- fade (mean-reversion) reading ------------------------------------------
def test_runner_supports_momentum_fade(candles_by_asset):
    _report, evaluations = runner.run("momentum_fade", list(candles_by_asset), 1,
                                      candles_by_asset, days=166)
    assert len(evaluations) == 1
    title, ev = evaluations[0]
    assert title.startswith("momentum_fade")
    assert ev["gates"] is not None                       # gates present
    assert set(ev["walk_forward"]) == set(candles_by_asset)  # walk-forward per asset

    # fade hit rate + continuation hit rate must sum to ~1.0 on the same events
    _rep, cont = runner.run("momentum", list(candles_by_asset), 1,
                            candles_by_asset, days=166)
    cont_ev = [e for t, e in cont if "combined" in t][0]
    fade_rate = ev["gates"]["details"]["hit_rate"]["overall_hit_rate"]
    cont_rate = cont_ev["gates"]["details"]["hit_rate"]["overall_hit_rate"]
    assert fade_rate + cont_rate == pytest.approx(1.0, abs=0.01)
    # expected honest outcome on random data: near 50%, no edge -> CLOSED
    assert fade_rate == pytest.approx(0.5, abs=0.05)
    assert ev["gates"]["verdict"] == "CLOSED"


def test_momentum_fade_is_not_bundled_into_all(candles_by_asset):
    """'all' must not score the same events twice via the inverse signal."""
    _report, evaluations = runner.run("all", list(candles_by_asset), 1,
                                      candles_by_asset, days=166)
    titles = [t for t, _ in evaluations]
    assert not any("fade" in t.lower() for t in titles)
    assert not any(t.startswith("momentum_fade") for t in titles)


# --- timeframe support -------------------------------------------------------
def test_report_states_the_timeframe_and_horizon_meaning(candles_by_asset):
    report_1h, _ = runner.run("momentum", list(candles_by_asset), 1,
                              candles_by_asset, days=166, timeframe="1h")
    assert "_Timeframe:_ 1h" in report_1h
    assert "_Horizon:_ t+1 = 1 hour forward" in report_1h
    assert "1h OHLCV" in report_1h

    report_4h, _ = runner.run("momentum", list(candles_by_asset), 1,
                              candles_by_asset, days=166, timeframe="4h")
    assert "_Timeframe:_ 4h" in report_4h
    assert "_Horizon:_ t+1 = 4 hours forward" in report_4h
    assert "4h OHLCV" in report_4h
    # the data table documents each asset's detected bar spacing
    assert "| Asset | Bars | Bar spacing |" in report_1h


def test_horizon_label_wording():
    assert runner.horizon_label(1, "1h") == "t+1 = 1 hour forward"
    assert runner.horizon_label(1, "4h") == "t+1 = 4 hours forward"
    assert runner.horizon_label(2, "4h") == "t+2 = 8 hours forward"
    assert runner.horizon_label(1, "1d") == "t+1 = 1 day forward"
    assert runner.horizon_label(3, "1d") == "t+3 = 3 days forward"


def test_report_warns_when_csv_spacing_contradicts_the_timeframe_flag(
        candles_by_asset, caplog):
    # the fixture data is 1h-spaced but the flag claims 4h — possible with --csv
    with caplog.at_level(logging.WARNING,
                         logger="sandbox.pattern_research.run_pattern_research"):
        report, _ = runner.run("momentum", list(candles_by_asset), 1,
                               candles_by_asset, days=166, timeframe="4h")
    assert "> **Warning:**" in report
    assert "bars of the loaded data" in report
    assert any("contradicts" in rec.message.lower() or "--timeframe" in rec.message
               for rec in caplog.records)


def test_cli_timeframe_flag_names_the_output_file(tmp_path):
    btc = tmp_path / "btc4h.csv"
    eth = tmp_path / "eth4h.csv"
    make_synthetic_candles(2500, seed=11, freq="4h").to_csv(btc, index_label="timestamp")
    make_synthetic_candles(2500, seed=12, freq="4h", start_price=2000.0).to_csv(
        eth, index_label="timestamp")
    out = tmp_path / "results"
    code = runner.main([
        "--pattern", "momentum_fade", "--asset", "both", "--horizon", "1",
        "--timeframe", "4h", "--output", str(out), "--quiet",
        "--csv", f"BTC/USDT={btc},ETH/USDT={eth}",
    ])
    assert code == 0
    files = list(out.glob("pattern_research_momentum_fade_both_4h_h1_*.md"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "_Timeframe:_ 4h" in text
    assert "> **Warning:**" not in text   # genuine 4h bars: no spacing warning


def test_cli_rejects_unsupported_timeframe(tmp_path):
    with pytest.raises(SystemExit):
        runner.main(["--pattern", "momentum", "--timeframe", "15m",
                     "--output", str(tmp_path), "--quiet"])


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
    files = list(out.glob("pattern_research_momentum_both_1h_h1_*.md"))
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
    assert list(out.glob("pattern_research_candlestick_BTCUSDT_1h_h1_*.md"))


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
