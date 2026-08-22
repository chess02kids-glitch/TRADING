"""Unit tests for dashboard.utils — pure helpers, no DB."""

from datetime import datetime, timezone

import pandas as pd

from dashboard.utils import (
    beats_to_text,
    compute_improvement_pct,
    format_large_number,
    format_mae,
    format_pct,
    format_timestamp_short,
    get_calibration_progress_pct,
    get_dominant_regime,
    get_regime_emoji,
)


def test_format_mae_normal():
    assert format_mae(372.37) == "$372.37"


def test_format_mae_none():
    assert format_mae(None) == "N/A"


def test_format_pct_positive():
    assert format_pct(7.4) == "+7.4%"


def test_format_pct_negative():
    assert format_pct(-3.2) == "-3.2%"


def test_compute_improvement_positive():
    result = compute_improvement_pct(370, 420)
    assert result is not None
    assert abs(result - ((420 - 370) / 420 * 100)) < 1e-9
    assert abs(result - 11.904761904761905) < 0.01


def test_compute_improvement_negative():
    result = compute_improvement_pct(420, 370)
    assert result is not None
    assert result < 0


def test_compute_improvement_none_inputs():
    assert compute_improvement_pct(None, 10) is None
    assert compute_improvement_pct(10, None) is None
    assert compute_improvement_pct(None, None) is None


def test_compute_improvement_zero_persist():
    assert compute_improvement_pct(10, 0) is None


def test_calibration_progress_pct():
    assert get_calibration_progress_pct(15, 30) == 50.0


def test_calibration_progress_capped():
    assert get_calibration_progress_pct(35, 30) == 100.0


def test_regime_emoji_low():
    assert get_regime_emoji("low") == "🟢"


def test_regime_emoji_high():
    assert get_regime_emoji("high") == "🔴"


def test_regime_emoji_unknown():
    assert get_regime_emoji("unknown") == "⚪"


def test_beats_to_text_true():
    label, color = beats_to_text(True)
    assert "✅" in label
    assert color == "#00ff88"


def test_beats_to_text_false():
    label, color = beats_to_text(False)
    assert "❌" in label
    assert color == "#ff4444"


def test_beats_to_text_none():
    label, color = beats_to_text(None)
    assert label == "Insufficient data"
    assert color == "#888888"


def test_format_timestamp_short_valid():
    ts = datetime(2026, 8, 22, 14, 0, tzinfo=timezone.utc)
    assert format_timestamp_short(ts) == "2026-08-22 14:00"


def test_format_timestamp_short_none():
    assert format_timestamp_short(None) == "N/A"
    assert format_timestamp_short(pd.NaT) == "N/A"


def test_get_dominant_regime():
    assert get_dominant_regime({"high": 40, "low": 3}) == "high"


def test_format_large_number_millions():
    assert format_large_number(1_500_000) == "1.50M"


def test_format_large_number_thousands():
    assert format_large_number(1500) == "1.50K"


def test_format_large_number_none():
    assert format_large_number(None) == "N/A"


def test_format_large_number_small():
    assert format_large_number(12.3) == "12.30"


def test_regime_emoji_medium():
    assert get_regime_emoji("medium") == "🟡"


def test_get_dominant_regime_empty():
    assert get_dominant_regime({}) == "unknown"


def test_format_pct_none():
    assert format_pct(None) == "N/A"


def test_format_mae_thousands_separator():
    assert format_mae(1372.5) == "$1,372.50"
