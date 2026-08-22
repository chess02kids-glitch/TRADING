"""
HAR Research Platform — Streamlit Dashboard

A live monitoring dashboard for the
HAR volatility calibration study.

Shows:
- Calibration progress (day N of 30)
- HAR vs Persistence accuracy
- Regime distribution
- Breakout events
- Prediction history

This is a RESEARCH MONITORING TOOL only.
No trading signals. No financial advice.
"""

import sys
from pathlib import Path

# Allow `streamlit run app.py` from dashboard/
# and `streamlit run dashboard/app.py` from repo root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd
from datetime import datetime, timezone

# Page configuration MUST be first Streamlit call
st.set_page_config(
    page_title="HAR Research Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from dashboard.data_loader import (
    fetch_calibration_summary,
    fetch_predictions,
    fetch_breakouts,
)
from dashboard.charts import (
    chart_predicted_vs_actual,
    chart_mae_over_time,
    chart_regime_distribution,
    chart_prediction_errors,
    chart_calibration_gauge,
    chart_har_coefficients,
)
from dashboard.utils import (
    format_mae,
    format_pct,
    compute_improvement_pct,
    beats_to_text,
    get_regime_emoji,
    get_dominant_regime,
    format_timestamp_short,
)
from dashboard.config import (
    ASSETS,
    TIMEFRAME,
    CALIBRATION_TOTAL_DAYS,
    AUTO_REFRESH_SECONDS,
    RECENT_PREDICTIONS_ROWS,
    BREAKOUT_TABLE_ROWS,
    REGIME_COLORS,
)


def apply_custom_css():
    """Apply dark theme CSS."""
    st.markdown("""
    <style>
        .main { background-color: #0e1117; }
        .metric-card {
            background: #1e2130;
            border-radius: 10px;
            padding: 15px;
            margin: 5px;
            border: 1px solid #2a2a2a;
        }
        .good { color: #00ff88; font-weight: bold; }
        .bad { color: #ff4444; font-weight: bold; }
        .neutral { color: #888888; }
        .big-metric {
            font-size: 2em;
            font-weight: bold;
        }
        .section-header {
            color: #4f8ef7;
            border-bottom: 1px solid #2a2a2a;
            padding-bottom: 5px;
            margin-top: 20px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.5em;
        }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render dashboard header."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📊 HAR Research Platform")
        st.caption(
            "Kronos Crypto Quantitative Research — "
            "Volatility Forecasting Study")
    with col2:
        now = datetime.now(timezone.utc)
        st.caption(
            f"🕐 {now.strftime('%Y-%m-%d %H:%M')} UTC")
        st.caption("🔄 Auto-refreshes every 5 min")

    st.markdown(
        "> ⚠️ **Research tool only.** "
        "Not financial advice. "
        "No trades are placed.",
        unsafe_allow_html=False)
    st.divider()


def render_calibration_progress(
    btc_summary: dict,
    eth_summary: dict,
):
    """Render top-level calibration status."""
    st.markdown(
        "### 🎯 Calibration Progress",
        unsafe_allow_html=False)

    cal_day = btc_summary.get(
        "calibration_day", 1)
    days_remaining = btc_summary.get(
        "days_remaining",
        CALIBRATION_TOTAL_DAYS)

    # Progress bar
    progress = min(
        1.0,
        cal_day / CALIBRATION_TOTAL_DAYS)
    st.progress(
        progress,
        text=f"Day {cal_day} of "
             f"{CALIBRATION_TOTAL_DAYS} "
             f"({days_remaining} days remaining)")

    # Gauge chart
    gauge = chart_calibration_gauge(
        cal_day, CALIBRATION_TOTAL_DAYS)
    st.plotly_chart(
        gauge,
        use_container_width=True,
        key="calibration_gauge")

    # Overall status
    btc_beats = btc_summary.get("har_beats")
    eth_beats = eth_summary.get("har_beats")

    if btc_beats is True and eth_beats is True:
        st.success(
            "✅ **ON TRACK** — HAR beating "
            "persistence on both assets")
    elif btc_beats is False or eth_beats is False:
        st.error(
            "❌ **AT RISK** — HAR not beating "
            "persistence on one or more assets")
    else:
        st.info(
            "⏳ **COLLECTING DATA** — "
            "Need 24+ completed predictions "
            "for calibration stats")


def render_asset_metrics(
    asset: str,
    summary: dict,
):
    """Render key metrics for one asset."""
    st.markdown(
        f"#### {asset}",
        unsafe_allow_html=False)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        har_mae = summary.get("har_mae")
        st.metric(
            "HAR MAE",
            format_mae(har_mae),
            help="Mean Absolute Error of HAR predictions")

    with col2:
        persist_mae = summary.get("persistence_mae")
        st.metric(
            "Persistence MAE",
            format_mae(persist_mae),
            help="MAE of naive persistence baseline")

    with col3:
        improvement = compute_improvement_pct(
            summary.get("har_mae"),
            summary.get("persistence_mae"))
        delta_color = (
            "normal" if (improvement or 0) > 0
            else "inverse")
        st.metric(
            "Improvement",
            format_pct(improvement),
            delta=format_pct(improvement),
            delta_color=delta_color,
            help="How much better HAR is vs persistence")

    with col4:
        beats_label, _ = beats_to_text(
            summary.get("har_beats"))
        st.metric(
            "Status",
            beats_label,
            help="Whether HAR is currently "
                 "beating persistence")

    # Second row
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            "Predictions",
            summary.get("total_predictions", 0),
            help="Total predictions logged")

    with col6:
        st.metric(
            "Completed",
            summary.get("completed", 0),
            help="Predictions with actual range filled")

    with col7:
        breakout_count = summary.get(
            "breakout_count", 0)
        breakout_rate = summary.get(
            "breakout_rate", 0)
        st.metric(
            "Breakouts",
            f"{breakout_count} "
            f"({breakout_rate:.1%})",
            help="Bars where actual > 2× predicted")

    with col8:
        bias = summary.get("mean_bias")
        bias_str = (
            f"${bias:+.1f}" if bias else "N/A")
        st.metric(
            "Mean Bias",
            bias_str,
            help="Positive = overestimating range")


def render_charts_section(
    asset: str,
    df: pd.DataFrame,
    regime_counts: dict,
):
    """Render charts for one asset."""
    st.markdown(
        f"### 📈 {asset} Charts",
        unsafe_allow_html=False)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Predicted vs Actual",
        "Daily MAE",
        "Regime Distribution",
        "Error Distribution",
        "HAR Coefficients",
    ])

    with tab1:
        fig = chart_predicted_vs_actual(
            df, asset)
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"pred_actual_{asset}")

    with tab2:
        fig = chart_mae_over_time(df, asset)
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"mae_{asset}")

    with tab3:
        fig = chart_regime_distribution(
            regime_counts)
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"regime_{asset}")

    with tab4:
        fig = chart_prediction_errors(df, asset)
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"errors_{asset}")

    with tab5:
        fig = chart_har_coefficients(df, asset)
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"coefs_{asset}")


def render_breakouts_section(
    breakouts_df: pd.DataFrame,
):
    """Render breakout events table."""
    st.markdown(
        "### 🚨 Breakout Events",
        unsafe_allow_html=False)

    if breakouts_df.empty:
        st.info(
            "No breakout events recorded yet. "
            "Breakouts occur when actual range "
            "> 2× HAR predicted range.")
        return

    st.caption(
        f"Last {len(breakouts_df)} breakout events "
        f"(actual range > 2× HAR predicted)")

    # Format for display
    display_df = breakouts_df.copy()
    if "timestamp" in display_df.columns:
        display_df["Time"] = display_df[
            "timestamp"].apply(
                format_timestamp_short)
    display_df = display_df.rename(columns={
        "asset": "Asset",
        "har_predicted_range": "Predicted ($)",
        "actual_range": "Actual ($)",
        "ratio": "Ratio",
    })

    cols_to_show = [
        "Time", "Asset", "Predicted ($)",
        "Actual ($)", "Ratio"]
    cols_exist = [
        c for c in cols_to_show
        if c in display_df.columns]

    st.dataframe(
        display_df[cols_exist].round(2),
        use_container_width=True,
        hide_index=True,
    )


def render_recent_predictions(
    asset: str,
    df: pd.DataFrame,
):
    """Render recent predictions table."""
    st.markdown(
        f"### 📋 {asset} — Recent Predictions",
        unsafe_allow_html=False)

    if df.empty:
        st.info("No prediction data yet.")
        return

    # Show last N rows
    recent = df.tail(
        RECENT_PREDICTIONS_ROWS
    ).sort_values(
        "timestamp",
        ascending=False).copy()

    # Format for display
    display_cols = {
        "timestamp": "Time",
        "har_predicted_range": "Predicted ($)",
        "actual_range": "Actual ($)",
        "abs_prediction_error": "Error ($)",
        "regime": "Regime",
        "breakout_flag": "Breakout",
    }

    display_df = recent[[
        c for c in display_cols.keys()
        if c in recent.columns
    ]].rename(columns=display_cols)

    if "Time" in display_df.columns:
        display_df["Time"] = display_df[
            "Time"].apply(format_timestamp_short)

    if "Breakout" in display_df.columns:
        display_df["Breakout"] = (
            display_df["Breakout"].apply(
                lambda x: "🚨" if x == 1 else ""))

    if "Regime" in display_df.columns:
        display_df["Regime"] = (
            display_df["Regime"].apply(
                lambda r: f"{get_regime_emoji(r)} {r}"
                if r else "—"))

    st.dataframe(
        display_df.round(2),
        use_container_width=True,
        hide_index=True,
    )


def render_sidebar():
    """Render sidebar with info and controls."""
    with st.sidebar:
        st.markdown("### ⚙️ Dashboard Info")
        st.markdown(
            "**HAR Research Platform**\n\n"
            "Monitoring a 30-day calibration study "
            "of the HAR volatility model on "
            "BTC/USDT and ETH/USDT.")

        st.divider()

        st.markdown("### 📊 Research Status")
        st.markdown(
            "**Model:** HAR "
            "(Heterogeneous Autoregressive)\n\n"
            "**Validated:** DM p ≈ 2.15e-26\n\n"
            "**Signal:** Volatility magnitude\n\n"
            "**NOT predicted:** Price direction")

        st.divider()

        st.markdown("### 🔗 Links")
        st.markdown(
            "- [GitHub Repository]"
            "(https://github.com/chess02kids-glitch/TRADING)\n"
            "- [Supabase Dashboard]"
            "(https://supabase.com)\n"
            "- [Telegram Alerts]"
            "(https://telegram.org)")

        st.divider()

        st.caption(
            "⚠️ Research tool only.\n"
            "Not financial advice.\n"
            "No trades are placed.")

        # Manual refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()


@st.cache_data(
    ttl=AUTO_REFRESH_SECONDS,
    show_spinner="Loading data...")
def load_all_data():
    """
    Load all dashboard data.
    Cached for AUTO_REFRESH_SECONDS.
    st.cache_data handles the refresh.
    """
    btc_summary = fetch_calibration_summary(
        "BTC/USDT", TIMEFRAME)
    eth_summary = fetch_calibration_summary(
        "ETH/USDT", TIMEFRAME)
    btc_df = fetch_predictions(
        "BTC/USDT", TIMEFRAME,
        limit=720)
    eth_df = fetch_predictions(
        "ETH/USDT", TIMEFRAME,
        limit=720)
    breakouts = fetch_breakouts(
        limit=BREAKOUT_TABLE_ROWS)

    return (btc_summary, eth_summary,
            btc_df, eth_df, breakouts)


def main():
    """Main dashboard entry point."""
    apply_custom_css()
    render_sidebar()
    render_header()

    # Check DB connection
    from dashboard.data_loader import get_db_url
    if not get_db_url():
        st.error(
            "❌ **SUPABASE_DB_URL not set.** "
            "Set this environment variable to "
            "connect to the database.")
        st.info(
            "For local development: "
            "Create a `.env` file or set "
            "the environment variable directly.")
        st.stop()

    # Load data (cached)
    with st.spinner("Loading calibration data..."):
        (btc_summary, eth_summary,
         btc_df, eth_df,
         breakouts_df) = load_all_data()

    # Calibration progress section
    render_calibration_progress(
        btc_summary, eth_summary)

    st.divider()

    # Asset tabs
    btc_tab, eth_tab = st.tabs([
        "₿ BTC/USDT", "Ξ ETH/USDT"])

    with btc_tab:
        render_asset_metrics("BTC/USDT", btc_summary)
        st.divider()
        render_charts_section(
            "BTC/USDT",
            btc_df,
            btc_summary.get("regime_counts", {}))
        render_recent_predictions(
            "BTC/USDT", btc_df)

    with eth_tab:
        render_asset_metrics("ETH/USDT", eth_summary)
        st.divider()
        render_charts_section(
            "ETH/USDT",
            eth_df,
            eth_summary.get("regime_counts", {}))
        render_recent_predictions(
            "ETH/USDT", eth_df)

    st.divider()

    # Breakouts section (all assets)
    render_breakouts_section(breakouts_df)

    st.divider()

    # Footer
    st.caption(
        "HAR Research Platform | "
        "Kronos Crypto Quantitative Research | "
        "Research tool only — not financial advice")


if __name__ == "__main__":
    main()
