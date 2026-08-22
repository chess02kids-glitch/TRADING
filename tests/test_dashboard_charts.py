"""Tests for the new dashboard.charts functions (Plotly figures, synthetic data)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from dashboard.charts import (
    daily_mae_frame,
    plot_coefficient_drift,
    plot_improvement_confidence,
)


def _coef_df(n=12):
    ts = pd.date_range("2026-08-10", periods=n, freq="D")
    return pd.DataFrame({
        "timestamp": ts,
        "coef_b0": [0.5 + 0.01 * i for i in range(n)],
        "coef_b1": [0.4 - 0.005 * i for i in range(n)],
        "coef_b2": [0.3 for _ in range(n)],
        "coef_b3": [0.1 + 0.002 * i for i in range(n)],
    })


def _mae_df(n=20):
    ts = pd.date_range("2026-08-01", periods=n, freq="D")
    return pd.DataFrame({
        "timestamp": ts,
        "har_mae": [10.0 + 0.1 * i for i in range(n)],
        "persistence_mae": [13.0 + 0.05 * i for i in range(n)],  # HAR better
    })


class TestPlotCoefficientDrift:

    def test_returns_figure_with_four_traces(self):
        fig = plot_coefficient_drift(_coef_df(), "BTC/USDT")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4  # B0, B1, B2, B3

    def test_empty_input_returns_empty_figure(self):
        fig = plot_coefficient_drift(pd.DataFrame(), "BTC/USDT")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_missing_columns_returns_empty_figure(self):
        df = pd.DataFrame({"timestamp": [pd.Timestamp("2026-08-10")]})
        fig = plot_coefficient_drift(df, "BTC/USDT")
        assert len(fig.data) == 0

    def test_title_mentions_stability(self):
        fig = plot_coefficient_drift(_coef_df(), "ETH/USDT")
        assert "Coefficient Stability" in fig.layout.title.text


class TestPlotImprovementConfidence:

    def test_returns_figure_with_band_mean_and_zero_line(self):
        fig = plot_improvement_confidence(_mae_df(), "BTC/USDT")
        assert isinstance(fig, go.Figure)
        # band (fill) + rolling mean line + zero line
        assert len(fig.data) == 3
        names = [tr.name for tr in fig.data]
        assert any("confidence band" in (n or "") for n in names)
        assert any("improvement" in (n or "").lower() for n in names)

    def test_empty_input_returns_empty_figure(self):
        fig = plot_improvement_confidence(pd.DataFrame(), "BTC/USDT")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_improvement_positive_when_har_beats_persistence(self):
        # HAR MAE lower than persistence -> improvement > 0 in the mean trace
        df = _mae_df(20)
        fig = plot_improvement_confidence(df, "BTC/USDT")
        mean_trace = [tr for tr in fig.data if tr.name and "improvement" in tr.name.lower()][0]
        ys = [v for v in mean_trace.y if v is not None and pd.notna(v)]
        assert all(y > 0 for y in ys)

    def test_title_mentions_ci(self):
        fig = plot_improvement_confidence(_mae_df(), "ETH/USDT")
        assert "Improvement" in fig.layout.title.text


class TestDailyMaeFrame:

    def test_builds_daily_frame(self):
        # 48 hourly completed predictions -> ~2 daily rows
        ts = pd.date_range("2026-08-01", periods=48, freq="h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": ts,
            "har_predicted_range": [100.0] * 48,
            "actual_range": [110.0] * 48,
            "abs_prediction_error": [10.0] * 48,
        })
        daily = daily_mae_frame(df)
        assert set(daily.columns) == {"timestamp", "har_mae", "persistence_mae"}
        assert len(daily) >= 1
        assert daily["har_mae"].iloc[0] == pytest.approx(10.0)
