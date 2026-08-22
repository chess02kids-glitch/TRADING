"""Unit tests for dashboard.charts — Plotly figures only."""

import pandas as pd
import plotly.graph_objects as go

from dashboard.charts import (
    BACKGROUND,
    HAR_COLOR,
    base_layout,
    chart_calibration_gauge,
    chart_har_coefficients,
    chart_mae_over_time,
    chart_predicted_vs_actual,
    chart_prediction_errors,
    chart_regime_distribution,
)


def _sample_predictions(n=10, with_breakout=False):
    ts = pd.date_range("2026-08-20", periods=n, freq="h", tz="UTC")
    predicted = [100.0 + i for i in range(n)]
    actual = [110.0 + i for i in range(n)]
    flags = [0] * n
    if with_breakout:
        flags[-1] = 1
        actual[-1] = predicted[-1] * 3
    return pd.DataFrame({
        "timestamp": ts,
        "har_predicted_range": predicted,
        "actual_range": actual,
        "abs_prediction_error": [
            abs(a - p) for a, p in zip(actual, predicted)
        ],
        "prediction_error": [
            a - p for a, p in zip(actual, predicted)
        ],
        "breakout_flag": flags,
        "coef_b0": [1.0] * n,
        "coef_b1": [0.4 + i * 0.01 for i in range(n)],
        "coef_b2": [0.3] * n,
        "coef_b3": [0.2] * n,
        "regime": ["high"] * n,
    })


def test_chart_predicted_vs_actual_empty():
    fig = chart_predicted_vs_actual(pd.DataFrame(), "BTC/USDT")
    assert isinstance(fig, go.Figure)
    texts = [a.text for a in fig.layout.annotations]
    assert any("No completed predictions yet" in t for t in texts)


def test_chart_predicted_vs_actual_with_data():
    fig = chart_predicted_vs_actual(_sample_predictions(8), "BTC/USDT")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2


def test_chart_predicted_vs_actual_breakouts():
    fig = chart_predicted_vs_actual(
        _sample_predictions(8, with_breakout=True), "BTC/USDT")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
    assert fig.data[2].name == "Breakout"


def test_chart_mae_empty():
    fig = chart_mae_over_time(pd.DataFrame(), "BTC/USDT")
    assert isinstance(fig, go.Figure)


def test_chart_mae_with_data():
    fig = chart_mae_over_time(_sample_predictions(12), "ETH/USDT")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.data[0].name == "HAR MAE"
    assert fig.data[1].name == "Persistence MAE"


def test_chart_regime_distribution_empty():
    fig = chart_regime_distribution({})
    assert isinstance(fig, go.Figure)
    texts = [a.text for a in fig.layout.annotations]
    assert any("No regime data yet" in t for t in texts)


def test_chart_regime_distribution_with_data():
    fig = chart_regime_distribution({"high": 40, "low": 3})
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "pie"


def test_chart_prediction_errors_empty():
    fig = chart_prediction_errors(pd.DataFrame(), "BTC/USDT")
    assert isinstance(fig, go.Figure)


def test_chart_calibration_gauge():
    fig = chart_calibration_gauge(7, 30)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "indicator"
    assert fig.data[0].value == 7


def test_chart_har_coefficients_empty():
    fig = chart_har_coefficients(pd.DataFrame(), "BTC/USDT")
    assert isinstance(fig, go.Figure)


def test_base_layout_has_dark_background():
    layout = base_layout("Test")
    assert layout["paper_bgcolor"] == "#0e1117"
    assert layout["paper_bgcolor"] == BACKGROUND


def test_chart_predicted_has_correct_colors():
    fig = chart_predicted_vs_actual(_sample_predictions(5), "BTC/USDT")
    assert fig.data[0].line.color == "#4f8ef7"
    assert fig.data[0].line.color == HAR_COLOR


def test_chart_har_coefficients_with_data():
    fig = chart_har_coefficients(_sample_predictions(6), "ETH/USDT")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3


def test_chart_prediction_errors_with_data():
    fig = chart_prediction_errors(_sample_predictions(15), "BTC/USDT")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "histogram"
