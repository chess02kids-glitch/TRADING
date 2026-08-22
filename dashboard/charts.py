"""
Chart building functions using Plotly.
All functions return plotly Figure objects.
No Streamlit imports here — just data → chart.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


BACKGROUND = "#0e1117"
CARD_BG = "#1e2130"
GRID_COLOR = "#2a2a2a"
TEXT_COLOR = "#e0e0e0"
HAR_COLOR = "#4f8ef7"
ACTUAL_COLOR = "#00ff88"
PERSISTENCE_COLOR = "#ffaa00"
BREAKOUT_COLOR = "#ff4444"


def base_layout(title: str = "") -> dict:
    """Return base Plotly layout dict."""
    return dict(
        title=title,
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            showgrid=True),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            showgrid=True),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            bgcolor=CARD_BG,
            bordercolor=GRID_COLOR),
    )


def _empty_figure(message: str, title: str) -> go.Figure:
    """Return a dark empty-state figure with a centered annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color=TEXT_COLOR))
    fig.update_layout(**base_layout(title))
    return fig


def chart_predicted_vs_actual(
    df: pd.DataFrame,
    asset: str = "BTC/USDT",
) -> go.Figure:
    """
    Line chart: HAR predicted range vs
    actual realized range over time.

    Only shows rows where actual_range
    is not null.
    """
    title = f"{asset} — Predicted vs Actual Range"
    if df is None or df.empty or "actual_range" not in df.columns:
        return _empty_figure(
            "No completed predictions yet", title)

    completed = df[
        df["actual_range"].notna()
    ].copy()

    fig = go.Figure()

    if completed.empty:
        fig.add_annotation(
            text="No completed predictions yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16,
                      color=TEXT_COLOR))
        fig.update_layout(
            **base_layout(
                f"{asset} — Predicted vs Actual Range"))
        return fig

    # HAR predicted line
    fig.add_trace(go.Scatter(
        x=completed["timestamp"],
        y=completed["har_predicted_range"],
        name="HAR Predicted",
        line=dict(color=HAR_COLOR, width=2),
        mode="lines",
    ))

    # Actual range line
    fig.add_trace(go.Scatter(
        x=completed["timestamp"],
        y=completed["actual_range"],
        name="Actual Range",
        line=dict(color=ACTUAL_COLOR, width=2),
        mode="lines",
    ))

    # Breakout markers
    if "breakout_flag" in completed.columns:
        breakouts = completed[
            completed["breakout_flag"] == 1]
        if not breakouts.empty:
            fig.add_trace(go.Scatter(
                x=breakouts["timestamp"],
                y=breakouts["actual_range"],
                name="Breakout",
                mode="markers",
                marker=dict(
                    color=BREAKOUT_COLOR,
                    size=10,
                    symbol="star"),
            ))

    fig.update_layout(
        **base_layout(
            f"{asset} — HAR Predicted vs Actual Range ($)"))
    return fig


def chart_mae_over_time(
    df: pd.DataFrame,
    asset: str = "BTC/USDT",
) -> go.Figure:
    """
    Line chart: Daily HAR MAE vs
    Persistence MAE over time.
    """
    if (
        df is None
        or df.empty
        or "actual_range" not in df.columns
    ):
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for MAE chart",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14,
                      color=TEXT_COLOR))
        fig.update_layout(
            **base_layout("Daily MAE Comparison"))
        return fig

    completed = df[
        df["actual_range"].notna()
    ].copy()

    if len(completed) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for MAE chart",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14,
                      color=TEXT_COLOR))
        fig.update_layout(
            **base_layout("Daily MAE Comparison"))
        return fig

    # Compute persistence MAE
    completed = completed.sort_values(
        "timestamp").copy()
    completed["persistence_pred"] = (
        completed["har_predicted_range"].shift(1))
    completed["persistence_error"] = (
        completed["actual_range"] -
        completed["persistence_pred"]
    ).abs()

    # Daily aggregation
    completed["date"] = (
        completed["timestamp"]
        .dt.floor("D"))
    daily = completed.groupby("date").agg(
        har_mae=("abs_prediction_error", "mean"),
        persistence_mae=(
            "persistence_error", "mean"),
    ).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily["date"],
        y=daily["har_mae"],
        name="HAR MAE",
        line=dict(color=HAR_COLOR, width=2),
        mode="lines+markers",
    ))

    fig.add_trace(go.Scatter(
        x=daily["date"],
        y=daily["persistence_mae"],
        name="Persistence MAE",
        line=dict(
            color=PERSISTENCE_COLOR,
            width=2,
            dash="dash"),
        mode="lines+markers",
    ))

    fig.update_layout(
        **base_layout(
            f"{asset} — Daily MAE: HAR vs Persistence"))
    return fig


def chart_regime_distribution(
    regime_counts: dict,
) -> go.Figure:
    """
    Pie chart: Distribution of
    low/medium/high volatility regimes.
    """
    from dashboard.config import REGIME_COLORS

    if not regime_counts:
        fig = go.Figure()
        fig.add_annotation(
            text="No regime data yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14,
                      color=TEXT_COLOR))
        fig.update_layout(
            paper_bgcolor=BACKGROUND,
            plot_bgcolor=BACKGROUND,
            font=dict(color=TEXT_COLOR),
            title="Regime Distribution",
        )
        return fig

    labels = list(regime_counts.keys())
    values = list(regime_counts.values())
    colors = [
        REGIME_COLORS.get(r, "#888888")
        for r in labels
    ]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.4,
        textinfo="label+percent",
        textfont=dict(color=TEXT_COLOR),
    )])

    fig.update_layout(
        paper_bgcolor=BACKGROUND,
        font=dict(color=TEXT_COLOR),
        title="Volatility Regime Distribution",
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    return fig


def chart_prediction_errors(
    df: pd.DataFrame,
    asset: str = "BTC/USDT",
) -> go.Figure:
    """
    Histogram of prediction errors
    (actual - predicted).
    """
    if (
        df is None
        or df.empty
        or "prediction_error" not in df.columns
    ):
        fig = go.Figure()
        fig.update_layout(
            **base_layout("Prediction Error Distribution"))
        return fig

    completed = df[
        df["prediction_error"].notna()
    ].copy()

    if completed.empty:
        fig = go.Figure()
        fig.update_layout(
            **base_layout("Prediction Error Distribution"))
        return fig

    fig = go.Figure(data=[go.Histogram(
        x=completed["prediction_error"],
        nbinsx=30,
        marker_color=HAR_COLOR,
        opacity=0.8,
        name="Prediction Error",
    )])

    # Add vertical line at zero
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color=BREAKOUT_COLOR,
        annotation_text="Zero bias",
        annotation_position="top",
    )

    fig.update_layout(
        **base_layout(
            f"{asset} — Prediction Error Distribution"),
        xaxis_title="Error ($)",
        yaxis_title="Count",
    )
    return fig


def chart_calibration_gauge(
    calibration_day: int,
    total_days: int = 30,
) -> go.Figure:
    """
    Gauge chart showing calibration progress.
    """
    pct = min(100, calibration_day / total_days * 100)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=calibration_day,
        number=dict(suffix=f" / {total_days} days"),
        delta=dict(reference=total_days),
        gauge=dict(
            axis=dict(
                range=[0, total_days],
                tickcolor=TEXT_COLOR),
            bar=dict(color=HAR_COLOR),
            bgcolor=CARD_BG,
            bordercolor=GRID_COLOR,
            steps=[
                dict(range=[0, 10],
                     color="#2a2a2a"),
                dict(range=[10, 20],
                     color="#1e3a2a"),
                dict(range=[20, 30],
                     color="#1a4a2a"),
            ],
            threshold=dict(
                line=dict(
                    color=ACTUAL_COLOR,
                    width=4),
                thickness=0.75,
                value=total_days,
            ),
        ),
        title=dict(
            text="Calibration Progress",
            font=dict(color=TEXT_COLOR)),
    ))

    fig.update_layout(
        paper_bgcolor=BACKGROUND,
        font=dict(color=TEXT_COLOR),
        height=250,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def chart_har_coefficients(
    df: pd.DataFrame,
    asset: str = "BTC/USDT",
) -> go.Figure:
    """
    Line chart showing how HAR model
    coefficients evolve over time.
    B1, B2, B3 (the range persistence
    coefficients).
    """
    if df.empty or "coef_b1" not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            **base_layout("HAR Coefficients"))
        return fig

    df_sorted = df.sort_values(
        "timestamp").copy()

    fig = go.Figure()

    coef_info = [
        ("coef_b1", "B1 (previous bar)",
         "#4f8ef7"),
        ("coef_b2", "B2 (5-bar mean)",
         "#00ff88"),
        ("coef_b3", "B3 (22-bar mean)",
         "#ffaa00"),
    ]

    for col, name, color in coef_info:
        if col in df_sorted.columns:
            fig.add_trace(go.Scatter(
                x=df_sorted["timestamp"],
                y=df_sorted[col],
                name=name,
                line=dict(color=color,
                          width=2),
                mode="lines",
            ))

    fig.update_layout(
        **base_layout(
            f"{asset} — HAR Model Coefficients "
            f"Over Time"))
    return fig
