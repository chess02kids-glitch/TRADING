"""
Helper functions for the dashboard.
Pure functions only. No DB access.
"""

import pandas as pd
from datetime import datetime, timezone


def format_large_number(
    value: float | None,
    decimals: int = 2,
    suffix: str = "",
) -> str:
    """
    Format a large number for display.
    e.g. 1500000 → "1.50M"
    """
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.{decimals}f}M{suffix}"
    if abs(value) >= 1_000:
        return f"{value/1_000:.{decimals}f}K{suffix}"
    return f"{value:.{decimals}f}{suffix}"


def format_mae(value: float | None) -> str:
    """Format MAE for display."""
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def format_pct(value: float | None) -> str:
    """Format percentage for display."""
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def compute_improvement_pct(
    har_mae: float | None,
    persistence_mae: float | None,
) -> float | None:
    """
    Compute percentage improvement of HAR
    over persistence.
    Positive = HAR is better.
    """
    if har_mae is None or persistence_mae is None:
        return None
    if persistence_mae == 0:
        return None
    return (
        (persistence_mae - har_mae)
        / persistence_mae * 100
    )


def get_calibration_progress_pct(
    calibration_day: int,
    total_days: int = 30,
) -> float:
    """
    Returns calibration progress as
    percentage (0.0 to 100.0).
    """
    return min(100.0,
               calibration_day / total_days * 100)


def get_regime_emoji(regime: str) -> str:
    """Return emoji for regime."""
    mapping = {
        "low": "🟢",
        "medium": "🟡",
        "high": "🔴",
    }
    return mapping.get(
        str(regime).lower(), "⚪")


def get_dominant_regime(
    regime_counts: dict,
) -> str:
    """Return the most frequent regime."""
    if not regime_counts:
        return "unknown"
    return max(
        regime_counts,
        key=regime_counts.get)


def beats_to_text(
    har_beats: bool | None,
) -> tuple[str, str]:
    """
    Returns (label, color) for beats status.
    """
    if har_beats is None:
        return "Insufficient data", "#888888"
    if har_beats:
        return "✅ HAR beats persistence", "#00ff88"
    return "❌ HAR losing to persistence", "#ff4444"


def format_timestamp_short(ts) -> str:
    """Format timestamp to short string."""
    if ts is None or pd.isna(ts):
        return "N/A"
    try:
        if hasattr(ts, "strftime"):
            return ts.strftime("%Y-%m-%d %H:%M")
        return str(ts)[:16]
    except Exception:
        return str(ts)[:16]
