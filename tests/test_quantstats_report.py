"""Tests for scripts/run_quantstats_report.py.

All tests are deterministic and offline:
- No real Supabase / no real database / no real network.
- ``quantstats`` reports are mocked through ``_emit_quantstats_html``.
- Timestamps are fixed except where a helper explicitly needs "now"
  (``make_prediction_rows`` and the calibration-day test).

Covers: empty / length / UTC-index conversions, perfect and terrible HAR
scores, clipping, minimum-row guard, daily aggregation, zero actual-range
safety, lag-1 benchmark behavior, summary statistics, Telegram message
formatting, combined summary generation, and main() exit codes for missing
DB/Telegram configuration.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# --- path bootstrap (same as scripts/run_quantstats_report.py) --------------
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.run_quantstats_report as qr  # noqa: E402

UTC = timezone.utc
T0 = datetime(2024, 1, 15, tzinfo=UTC)
MOD = "scripts.run_quantstats_report"


def _iso(ts: datetime) -> str:
    """Fixed ISO8601 UTC string (no 'Z' ambiguity in tests)."""
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def make_single_row(
    timestamp: datetime,
    actual: float,
    predicted: float,
    asset: str = "BTC/USDT",
    regime: str = "high",
    breakout_flag: int = 0,
    abs_error: float | None = None,
) -> dict:
    """Build one Supabase-shaped prediction row."""
    if abs_error is None:
        abs_error = abs(actual - predicted)
    return {
        "id": 0,
        "timestamp": _iso(timestamp),
        "asset": asset,
        "timeframe": "1h",
        "har_predicted_range": float(predicted),
        "coef_b0": 50.0,
        "coef_b1": 0.3,
        "coef_b2": 0.3,
        "coef_b3": 0.3,
        "n_obs": 100,
        "regime": regime,
        "actual_range": float(actual),
        "prediction_error": float(actual - predicted),
        "abs_prediction_error": float(abs_error),
        "breakout_flag": int(breakout_flag),
        "created_at": _iso(timestamp),
    }


def _make_rows_over_days(
    hours: int,
    start: datetime = T0,
    asset: str = "BTC/USDT",
    har_wins: bool = True,
    actual_fn=None,
) -> list[dict]:
    """Generate ``hours`` hourly rows starting at ``start``."""
    rows = []
    for i in range(hours):
        ts = start + timedelta(hours=i)
        if actual_fn is None:
            actual = 400.0 + (i % 10) * 20.0
        else:
            actual = float(actual_fn(i))
        predicted = actual * (0.9 if har_wins else 1.2)
        rows.append(make_single_row(ts, actual, predicted, asset=asset))
    return rows


def make_prediction_rows(
    n: int = 50,
    asset: str = "BTC/USDT",
    har_wins: bool = True,
) -> list[dict]:
    """Build synthetic prediction rows matching the Supabase schema.

    If ``har_wins=True``, HAR errors are 10% smaller than persistence errors.
    Timestamps hourly from ``n`` hours ago. All rows have ``actual_range``.
    """
    rows = []
    base_time = datetime.now(timezone.utc) - timedelta(hours=n)

    prev_predicted = 500.0  # noqa: F841 - kept for parity with the spec helper

    for i in range(n):
        ts = base_time + timedelta(hours=i)
        actual = 400.0 + (i % 10) * 20.0
        predicted = actual * (0.9 if har_wins else 1.2)
        error = actual - predicted

        rows.append(
            {
                "timestamp": ts.isoformat(),
                "asset": asset,
                "timeframe": "1h",
                "har_predicted_range": predicted,
                "actual_range": actual,
                "prediction_error": error,
                "abs_prediction_error": abs(error),
                "breakout_flag": 1 if abs(error) > 2 * predicted else 0,
                "regime": "high",
                "created_at": ts.isoformat(),
                "coef_b0": 50.0,
                "coef_b1": 0.3,
                "coef_b2": 0.3,
                "coef_b3": 0.3,
                "n_obs": 100,
            }
        )
        prev_predicted = predicted

    return rows


# ---------------------------------------------------------------------------
# Data fetching / conversion basics
# ---------------------------------------------------------------------------

class TestReturnsConversionBasics:
    def test_fetch_returns_empty_on_no_rows(self):
        empty = qr.predictions_to_returns([])
        assert isinstance(empty, pd.Series)
        assert len(empty) == 0

        empty_bm = qr.predictions_to_benchmark([])
        assert isinstance(empty_bm, pd.Series)
        assert len(empty_bm) == 0

    def test_fetch_returns_correct_length(self):
        rows = _make_rows_over_days(24)
        series = qr.predictions_to_returns(rows)
        assert len(series) == 1  # one calendar day

    def test_fetch_returns_indexed_by_date(self):
        rows = _make_rows_over_days(48)
        series = qr.predictions_to_returns(rows)
        assert isinstance(series.index, pd.DatetimeIndex)
        assert len(series) == 2
        assert all(value.time() == pd.Timestamp("00:00:00").time()
                   for value in pd.Series(series.index))
        assert all(value.hour == 0 for value in series.index)


# ---------------------------------------------------------------------------
# predictions_to_returns
# ---------------------------------------------------------------------------

class TestPredictionsToReturns:
    def test_returns_perfect_prediction(self):
        row = make_single_row(T0, actual=100.0, predicted=100.0)
        assert qr.compute_har_score(row) == pytest.approx(1.0, abs=1e-9)

        # Perfect HAR with a bad first benchmark row gives a high
        # relative "return" on the first comparable day.
        rows = []
        for i in range(10):
            ts = T0 + timedelta(days=i, hours=i)
            actual = 100.0
            predicted = 0.0 if i == 0 else actual
            rows.append(make_single_row(ts, actual, predicted))
        series = qr.predictions_to_returns(rows)
        assert len(series) == 9
        assert series.iloc[0] == pytest.approx(1.0, abs=1e-9)

    def test_returns_terrible_prediction(self):
        row = make_single_row(T0, actual=100.0, predicted=1000.0)
        assert qr.compute_har_score(row) == pytest.approx(-1.0, abs=1e-9)

        # Consistently terrible HAR is still a valid, finite series. Because
        # the benchmark is lag-1 of the same HAR predictions, the relative
        # "return" is centered around zero (both strategies share the bias).
        rows = _make_rows_over_days(24, har_wins=False)
        series = qr.predictions_to_returns(rows)
        assert len(series) >= 1
        assert series.notna().all()
        assert series.abs().max() <= 2.0

    def test_returns_clipped_to_minus_one(self):
        row = make_single_row(T0, actual=100.0, predicted=5000.0)
        assert qr.compute_har_score(row) == -1.0

    def test_returns_requires_10_minimum(self):
        rows = make_prediction_rows(9)
        assert len(qr.predictions_to_returns(rows)) == 0

    def test_returns_daily_aggregation(self):
        rows = _make_rows_over_days(48)
        series = qr.predictions_to_returns(rows)
        assert len(series) == 2
        assert len(series.index.unique()) == 2

    def test_returns_index_is_utc(self):
        rows = make_prediction_rows(50)
        series = qr.predictions_to_returns(rows)
        assert str(series.index.tz) == "UTC"

    def test_returns_handles_zero_actual(self):
        rows = _make_rows_over_days(24)
        # One zero-actual row must be skipped safely (no divide-by-zero).
        rows[5]["actual_range"] = 0.0
        rows[5]["abs_prediction_error"] = 0.0
        series = qr.predictions_to_returns(rows)
        assert len(series) >= 1
        assert series.notna().all()


# ---------------------------------------------------------------------------
# predictions_to_benchmark
# ---------------------------------------------------------------------------

class TestPredictionsToBenchmark:
    def test_benchmark_uses_lag1(self):
        rows = []
        for i in range(10):
            ts = T0 + timedelta(days=i, hours=i)
            actual = 100.0
            predicted = 0.0 if i == 0 else actual
            rows.append(make_single_row(ts, actual, predicted))
        benchmark = qr.predictions_to_benchmark(rows)
        assert len(benchmark) == 9
        # Row 1 uses row 0's prediction (0.0), not its own (100.0).
        assert benchmark.iloc[0] == pytest.approx(0.0, abs=1e-9)
        # Row 2 uses row 1's prediction (100.0).
        assert benchmark.iloc[1] == pytest.approx(1.0, abs=1e-9)

    def test_benchmark_first_row_skipped(self):
        rows = []
        for i in range(10):
            ts = T0 + timedelta(days=i, hours=i)
            rows.append(make_single_row(ts, actual=100.0, predicted=100.0))
        benchmark = qr.predictions_to_benchmark(rows)
        assert len(benchmark) == 9
        assert pd.Timestamp(benchmark.index[0]).date() != rows[0]["timestamp"][:10]


# ---------------------------------------------------------------------------
# compute_summary_stats
# ---------------------------------------------------------------------------

class TestComputeSummaryStats:
    def _three_rows(self) -> list[dict]:
        return [
            make_single_row(T0 + timedelta(hours=0), 100.0, 90.0),
            make_single_row(T0 + timedelta(hours=1), 110.0, 100.0),
            make_single_row(T0 + timedelta(hours=2), 120.0, 100.0),
        ]

    def test_stats_correct_mae(self):
        stats = qr.compute_summary_stats(self._three_rows())
        assert stats["har_mae"] == pytest.approx((10 + 10 + 20) / 3)
        assert stats["persistence_mae"] == pytest.approx((20 + 20) / 2)
        assert stats["n_predictions"] == 3

    def test_stats_beats_persistence_true(self):
        stats = qr.compute_summary_stats(self._three_rows())
        assert stats["har_beats"] is True
        assert stats["improvement_pct"] == pytest.approx(33.3333, abs=0.01)
        assert stats["improvement_pct"] > 0

    def test_stats_beats_persistence_false(self):
        rows = [
            make_single_row(T0, actual=100.0, predicted=200.0),
            make_single_row(T0 + timedelta(hours=1), actual=110.0, predicted=300.0),
            make_single_row(T0 + timedelta(hours=2), actual=120.0, predicted=400.0),
        ]
        stats = qr.compute_summary_stats(rows)
        assert stats["har_beats"] is False
        assert stats["improvement_pct"] < 0

    def test_stats_breakout_count(self):
        rows = [
            make_single_row(T0, 100.0, 90.0, breakout_flag=1),
            make_single_row(T0 + timedelta(hours=1), 100.0, 101.0, breakout_flag=0),
            make_single_row(T0 + timedelta(hours=2), 100.0, 80.0, breakout_flag=1),
        ]
        stats = qr.compute_summary_stats(rows)
        assert stats["breakout_count"] == 2
        assert stats["breakout_rate"] == pytest.approx(2 / 3)

    def test_stats_regime_counts(self):
        regimes = ["low"] * 5 + ["medium"] * 3 + ["high"] * 2
        rows = [
            make_single_row(T0 + timedelta(hours=i), 100.0 + i, 100.0, regime=regimes[i])
            for i in range(10)
        ]
        stats = qr.compute_summary_stats(rows)
        assert stats["regime_counts"] == {"low": 5, "medium": 3, "high": 2}

    def test_stats_calibration_day(self):
        first_ts = datetime.now(timezone.utc) - timedelta(days=4, hours=1)
        rows = [make_single_row(first_ts, 100.0, 100.0)]
        stats = qr.compute_summary_stats(rows)
        assert stats["calibration_day"] == 5

    def test_stats_empty_input(self):
        assert qr.compute_summary_stats([]) is None
        assert qr.compute_summary_stats(None) is None


# ---------------------------------------------------------------------------
# Telegram message
# ---------------------------------------------------------------------------

class TestTelegramMessage:
    def _stats(self, har_mae=372.37, persistence_mae=402.19,
               beats=True, day=3) -> dict:
        return {
            "n_predictions": 44,
            "n_days": 3,
            "har_mae": har_mae,
            "persistence_mae": persistence_mae,
            "improvement_pct": (persistence_mae - har_mae)
            / persistence_mae * 100 if persistence_mae else 0.0,
            "har_beats": beats,
            "breakout_count": 4,
            "breakout_rate": 0.093,
            "regime_counts": {"low": 0, "medium": 0, "high": 44},
            "best_day_error": 300.0,
            "worst_day_error": 450.0,
            "mean_bias": -12.3,
            "calibration_day": day,
        }

    def test_message_contains_btc(self):
        msg = qr.build_quantstats_telegram_message(
            self._stats(), self._stats(), {}
        )
        assert "BTC/USDT" in msg

    def test_message_contains_eth(self):
        msg = qr.build_quantstats_telegram_message(
            self._stats(), self._stats(), {}
        )
        assert "ETH/USDT" in msg

    def test_message_shows_improvement(self):
        msg = qr.build_quantstats_telegram_message(
            self._stats(), self._stats(), {}
        )
        assert "+7.4%" in msg

    def test_message_shows_beats_yes(self):
        msg = qr.build_quantstats_telegram_message(
            self._stats(), self._stats(), {}
        )
        assert "✅" in msg

    def test_message_shows_beats_no(self):
        msg = qr.build_quantstats_telegram_message(
            self._stats(har_mae=500, persistence_mae=400, beats=False), {},
            {"btc": "reports/btc_har_report.html"},
        )
        assert "❌" in msg

    def test_message_contains_disclaimer(self):
        msg = qr.build_quantstats_telegram_message(
            self._stats(), self._stats(), {}
        )
        assert "Not financial advice" in msg


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

class TestGenerateHtmlReport:
    def _series(self) -> pd.Series:
        index = pd.date_range(start="2024-01-15", periods=12, freq="D", tz="UTC")
        return pd.Series([0.01] * 12, index=index, dtype=float)

    def test_generate_html_report_combined_created(self, tmp_path):
        results = qr.generate_html_report(
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            output_dir=str(tmp_path),
        )
        assert "combined" in results
        assert (tmp_path / "combined_summary.html").exists()

    def test_generate_html_report_quantstats_mocked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(qr, "QUANTSTATS_AVAILABLE", True)
        with patch.object(qr, "_emit_quantstats_html") as mock_emit:
            results = qr.generate_html_report(
                self._series(),
                self._series(),
                self._series(),
                self._series(),
                output_dir=str(tmp_path),
            )
        assert mock_emit.call_count == 2
        assert "btc" in results and "eth" in results
        # QuantStats is mocked, so only the reported paths are authoritative
        # (the real QuantStats call creates the actual HTML files in prod).
        assert results["btc"].endswith("btc_har_report.html")
        assert results["eth"].endswith("eth_har_report.html")

    def test_generate_html_report_no_quantstats_skips_btc_eth(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(qr, "QUANTSTATS_AVAILABLE", False)
        results = qr.generate_html_report(
            self._series(),
            self._series(),
            self._series(),
            self._series(),
            output_dir=str(tmp_path),
        )
        assert "btc" not in results
        assert "eth" not in results
        assert "combined" in results
        assert not (tmp_path / "btc_har_report.html").exists()


# ---------------------------------------------------------------------------
# Supabase fetch helpers (offline)
# ---------------------------------------------------------------------------

class TestFetchHelpers:
    def test_fetch_all_predictions_empty_url_returns_empty(self):
        assert qr.fetch_all_predictions("", "BTC/USDT", "1h") == []

    def test_fetch_both_assets_empty_url(self):
        result = qr.fetch_both_assets("")
        assert list(result.keys()) == ["BTC/USDT", "ETH/USDT"]
        assert result["BTC/USDT"] == []
        assert result["ETH/USDT"] == []


# ---------------------------------------------------------------------------
# main() environment checks
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_no_db_url_returns_1(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
        assert qr.main() == 1

    def test_main_no_telegram_returns_1(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://x")
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        assert qr.main() == 1
