"""Tests for the HAR weekly research summary (no network or database access).

Deterministic by construction:
- psycopg / DB access functions are mocked at the script module level
- Telegram sends are mocked
- all rows are synthetic with fixed timestamps relative to now
"""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from kronos_trading.alerts.telegram_sender import SendResult, TelegramConfig

from scripts import run_weekly_summary as ws

SCRIPT = "scripts.run_weekly_summary"
CONFIG = TelegramConfig("test-token", "test-chat")
SUCCESS = SendResult(True, 42, None, 1)
FAILURE = SendResult(False, None, "telegram error", 3)


def make_row(
    ts_iso: str,
    predicted: float = 100.0,
    actual: float = 100.0,
    regime: str | None = "medium",
    breakout: int = 0,
    asset: str = "BTC/USDT",
) -> dict:
    """One prediction row shaped like the har_predictions schema."""
    return {
        "id": 1,
        "timestamp": ts_iso,
        "asset": asset,
        "timeframe": "1h",
        "har_predicted_range": predicted,
        "coef_b0": 1.0,
        "coef_b1": 0.5,
        "coef_b2": 0.3,
        "coef_b3": 0.2,
        "n_obs": 100,
        "regime": regime,
        "actual_range": actual,
        "prediction_error": actual - predicted,
        "abs_prediction_error": abs(actual - predicted),
        "breakout_flag": breakout,
        "created_at": ts_iso,
    }


def make_week_rows(
    predicted: float | list = 100.0,
    actual: float | list = 100.0,
    regime: str | None = "medium",
    breakout: int = 0,
    n: int = 30,
    start_hours_ago: int = 0,
    asset: str = "BTC/USDT",
) -> list[dict]:
    """Newest-first rows (matching get_prediction_history ordering)."""
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(n):
        ts = now - timedelta(hours=start_hours_ago + i)
        pred = predicted[i % len(predicted)] if isinstance(predicted, list) else predicted
        act = actual[i % len(actual)] if isinstance(actual, list) else actual
        b = breakout[i % len(breakout)] if isinstance(breakout, list) else breakout
        rows.append(make_row(ts.strftime("%Y-%m-%dT%H:%M:%SZ"), pred, act, regime, b, asset))
    return rows


def week_stats(**overrides):
    """Build a full weekly-stats dict from synthetic rows."""
    stats = ws.compute_weekly_stats(make_week_rows(n=30))
    assert stats is not None
    stats.update(overrides)
    return stats


def make_progress(**overrides):
    progress = {
        "btc_wins": 2,
        "btc_total": 2,
        "eth_wins": 1,
        "eth_total": 2,
        "degrading": False,
        "verdict": "ON TRACK",
    }
    progress.update(overrides)
    return progress


def make_insufficient(n_predictions=10, n_completed=10):
    return {
        "insufficient": True,
        "n_predictions": n_predictions,
        "n_completed": n_completed,
    }


# ─── compute_weekly_stats ──────────────────────────────────────────────────


def test_weekly_stats_correct_mae():
    """Known rows: predicted=100, actual=90 -> HAR MAE = 10.0."""
    rows = make_week_rows(predicted=100.0, actual=90.0, n=24)
    stats = ws.compute_weekly_stats(rows)
    assert stats is not None
    assert stats["har_mae"] == 10.0
    assert stats["n_obs"] == 24


def test_weekly_stats_beats_persistence():
    """Perfect predictions vs lag-1 misalignment -> HAR wins."""
    # predicted alternates 100/150 and actual == predicted -> HAR MAE 0,
    # persistence MAE 50 (previous prediction is always off by 50).
    preds = [100.0, 150.0] * 15
    rows = make_week_rows(predicted=preds, actual=preds, n=30)
    stats = ws.compute_weekly_stats(rows)
    assert stats["har_mae"] == 0.0
    assert stats["persistence_mae"] == 50.0
    assert stats["har_beats_persistence"] is True


def test_weekly_stats_loses_persistence():
    """Predictions one step out of phase -> HAR loses."""
    preds = [100.0, 150.0] * 15
    actuals = [150.0, 100.0] * 15
    rows = make_week_rows(predicted=preds, actual=actuals, n=30)
    stats = ws.compute_weekly_stats(rows)
    assert stats["har_mae"] == 50.0
    assert stats["persistence_mae"] == 0.0
    assert stats["har_beats_persistence"] is False


def test_weekly_stats_returns_none_insufficient():
    assert ws.compute_weekly_stats(make_week_rows(n=23)) is None


def test_weekly_stats_returns_none_empty():
    assert ws.compute_weekly_stats([]) is None


def test_weekly_stats_breakout_count():
    rows = make_week_rows(n=30, breakout=[1, 0, 0])
    stats = ws.compute_weekly_stats(rows)
    assert stats["breakout_count"] == 10  # every 3rd row


def test_weekly_stats_breakout_rate():
    rows = make_week_rows(n=30, breakout=[1, 0, 0])
    stats = ws.compute_weekly_stats(rows)
    assert stats["breakout_rate"] == 10 / 30


def test_weekly_stats_mean_bias_positive():
    """actual > predicted -> positive bias (underestimation)."""
    rows = make_week_rows(predicted=100.0, actual=110.0, n=24)
    stats = ws.compute_weekly_stats(rows)
    assert stats["mean_bias"] == 10.0


def test_weekly_stats_mean_bias_negative():
    """actual < predicted -> negative bias (overestimation)."""
    rows = make_week_rows(predicted=100.0, actual=90.0, n=24)
    stats = ws.compute_weekly_stats(rows)
    assert stats["mean_bias"] == -10.0


def test_weekly_stats_profit_factor():
    """Errors +10 and -5 alternate -> PF = 120 / 60 = 2.0."""
    rows = make_week_rows(
        predicted=100.0, actual=[110.0, 95.0], n=24
    )
    stats = ws.compute_weekly_stats(rows)
    assert stats["profit_factor"] == 2.0


def test_weekly_stats_profit_factor_none_when_no_negatives():
    """All errors positive -> no denominator -> PF None."""
    rows = make_week_rows(predicted=100.0, actual=110.0, n=24)
    stats = ws.compute_weekly_stats(rows)
    assert stats["profit_factor"] is None


def test_weekly_stats_worst_best_ratio():
    """actuals 250, 300, rest 100 vs predicted 100 -> worst 3.0 best 1.0."""
    actuals = [250.0, 300.0] + [100.0] * 28
    stats = ws.compute_weekly_stats(
        make_week_rows(predicted=100.0, actual=actuals, n=30)
    )
    assert stats["worst_ratio"] == 3.0
    assert stats["best_ratio"] == 1.0


def test_weekly_stats_nonpositive_predicted_skipped_for_ratios():
    """A zero/negative prediction is counted in totals but never in ratios."""
    rows = make_week_rows(predicted=100.0, actual=100.0, n=30)
    rows[0]["har_predicted_range"] = 0.0
    rows[0]["actual_range"] = 999.0
    rows[1]["har_predicted_range"] = -5.0
    rows[1]["actual_range"] = 999.0
    stats = ws.compute_weekly_stats(rows)
    assert stats["n_obs"] == 30
    assert stats["worst_ratio"] == 1.0  # 999/0 and 999/-5 excluded
    assert stats["best_ratio"] == 1.0


def test_weekly_stats_drops_rows_with_missing_values():
    rows = make_week_rows(n=30)
    rows[0]["actual_range"] = None
    rows[1]["har_predicted_range"] = None
    stats = ws.compute_weekly_stats(rows)
    assert stats["n_obs"] == 28


# ─── compute_regime_distribution ───────────────────────────────────────────


def test_regime_distribution_counts():
    rows = []
    for i in range(20):
        regime = "low" if i < 10 else ("medium" if i < 15 else "high")
        rows.append(make_row(f"2026-01-01T{i:02d}:00:00Z", regime=regime))
    dist = ws.compute_regime_distribution(rows)
    assert dist["low"] == 10 and dist["medium"] == 5 and dist["high"] == 5
    assert dist["unknown"] == 0
    assert dist["low_pct"] == 0.50
    assert dist["medium_pct"] == 0.25
    assert dist["high_pct"] == 0.25


def test_regime_distribution_all_unknown():
    rows = make_week_rows(regime=None, n=10)
    dist = ws.compute_regime_distribution(rows)
    assert dist["unknown"] == 10
    assert dist["low_pct"] == 0.0
    assert dist["medium_pct"] == 0.0
    assert dist["high_pct"] == 0.0


def test_regime_distribution_mixed_unknown():
    rows = make_week_rows(regime="low", n=10)
    rows[0]["regime"] = None
    rows[1]["regime"] = "weird-value"
    dist = ws.compute_regime_distribution(rows)
    assert dist["low"] == 8
    assert dist["unknown"] == 2


def test_regime_distribution_empty():
    dist = ws.compute_regime_distribution([])
    assert dist == {
        "low": 0, "medium": 0, "high": 0, "unknown": 0,
        "low_pct": 0.0, "medium_pct": 0.0, "high_pct": 0.0,
    }


# ─── fetch_week_predictions windowing ──────────────────────────────────────


def _rows_at_offsets(offsets):
    now = datetime.now(timezone.utc)
    return [
        make_row((now - timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ"))
        for h in offsets
    ]


def test_fetch_window_this_week():
    rows = _rows_at_offsets([10, 100, 200, 300, 400])
    with patch(f"{SCRIPT}.get_prediction_history", return_value=rows):
        result = ws.fetch_week_predictions(
            "postgresql://fake", "BTC/USDT", "1h", 0, 168
        )
    assert len(result) == 2  # 10h and 100h ago


def test_fetch_window_previous_week():
    rows = _rows_at_offsets([10, 100, 200, 300, 400])
    with patch(f"{SCRIPT}.get_prediction_history", return_value=rows):
        result = ws.fetch_week_predictions(
            "postgresql://fake", "BTC/USDT", "1h", 168, 336
        )
    assert len(result) == 2  # 200h and 300h ago


def test_fetch_window_skips_unparseable_timestamps():
    rows = _rows_at_offsets([10, 100])
    rows[0]["timestamp"] = "not-a-date"
    with patch(f"{SCRIPT}.get_prediction_history", return_value=rows):
        result = ws.fetch_week_predictions(
            "postgresql://fake", "BTC/USDT", "1h", 0, 168
        )
    assert len(result) == 1


def test_fetch_pending_window_uses_pending_query():
    rows = _rows_at_offsets([5, 300])
    with patch(f"{SCRIPT}.get_pending_predictions", return_value=rows):
        result = ws.fetch_pending_window(
            "postgresql://fake", "BTC/USDT", "1h", 0, 168
        )
    assert len(result) == 1


# ─── compute_weeks_beating / calibration progress ─────────────────────────


def test_compute_weeks_beating_counts():
    """2 full chunks: one perfect week (beats), one phase-shifted (loses)."""
    perfect = make_week_rows(predicted=[100.0, 150.0] * 84, actual=[100.0, 150.0] * 84, n=168, start_hours_ago=168)
    losing = make_week_rows(predicted=[100.0, 150.0] * 84, actual=[150.0, 100.0] * 84, n=168, start_hours_ago=0)
    with patch(f"{SCRIPT}.get_prediction_history", return_value=perfect + losing):
        result = ws.compute_weeks_beating("postgresql://fake", "BTC/USDT", "1h")
    assert result == {"wins": 1, "total": 2}


def test_compute_weeks_beating_skips_partial_weeks():
    rows = make_week_rows(n=20)  # one partial chunk, < 24 rows
    with patch(f"{SCRIPT}.get_prediction_history", return_value=rows):
        result = ws.compute_weeks_beating("postgresql://fake", "BTC/USDT", "1h")
    assert result == {"wins": 0, "total": 0}


def test_calibration_progress_degrading():
    current = week_stats(har_mae=500.0)
    previous = week_stats(har_mae=100.0)
    progress = ws.compute_calibration_progress(
        current, previous, current, previous,
        {"wins": 1, "total": 2}, {"wins": 1, "total": 2},
    )
    assert progress["degrading"] is True
    assert progress["verdict"] == "AT RISK"


def test_calibration_progress_on_track():
    current = week_stats(har_mae=100.0)
    previous = week_stats(har_mae=200.0)
    progress = ws.compute_calibration_progress(
        current, previous, current, previous,
        {"wins": 2, "total": 2}, {"wins": 1, "total": 2},
    )
    assert progress["degrading"] is False
    assert progress["verdict"] == "ON TRACK"
    assert progress["btc_wins"] == 2
    assert progress["btc_total"] == 2


# ─── format_weekly_message ─────────────────────────────────────────────────


def test_message_contains_btc():
    msg = ws.format_weekly_message(
        week_stats(), week_stats(), None, None, 5, make_progress(),
    )
    assert "BTC/USDT" in msg


def test_message_contains_eth():
    msg = ws.format_weekly_message(
        week_stats(), week_stats(), None, None, 5, make_progress(),
    )
    assert "ETH/USDT" in msg


def test_message_beats_naive_yes():
    stats = week_stats(har_beats_persistence=True)
    msg = ws.format_weekly_message(stats, stats, None, None, 5, make_progress())
    assert "✅ YES" in msg


def test_message_beats_naive_no():
    stats = week_stats(har_beats_persistence=False)
    msg = ws.format_weekly_message(stats, stats, None, None, 5, make_progress())
    assert "❌ NO" in msg


def test_message_no_data_handled():
    msg = ws.format_weekly_message(None, None, None, None, 1, make_progress())
    assert "Insufficient data" in msg or "first weekly report" in msg


def test_message_first_week_comparison():
    msg = ws.format_weekly_message(
        week_stats(), week_stats(), None, None, 5, make_progress(),
    )
    assert "First week" in msg


def test_message_week_over_week_better():
    current = week_stats(har_mae=100.0)
    previous = week_stats(har_mae=200.0)
    msg = ws.format_weekly_message(
        current, current, previous, previous, 5, make_progress(),
    )
    assert "better" in msg
    assert "100.00 vs 200.00 last week" in msg


def test_message_week_over_week_worse():
    current = week_stats(har_mae=200.0)
    previous = week_stats(har_mae=100.0)
    msg = ws.format_weekly_message(
        current, current, previous, previous, 5, make_progress(),
    )
    assert "worse" in msg


def test_message_regime_percentages():
    msg = ws.format_weekly_message(
        week_stats(), week_stats(), None, None, 5, make_progress(),
    )
    assert "Low:" in msg
    assert "Medium:" in msg
    assert "High:" in msg


def test_message_unknown_regime_line_when_present():
    stats = week_stats()
    stats["regime"] = {
        "low": 0, "medium": 0, "high": 0, "unknown": 30,
        "low_pct": 0.0, "medium_pct": 0.0, "high_pct": 0.0,
    }
    msg = ws.format_weekly_message(stats, stats, None, None, 5, make_progress())
    assert "Unknown:" in msg


def test_message_contains_disclaimer():
    msg = ws.format_weekly_message(
        week_stats(), week_stats(), None, None, 5, make_progress(),
    )
    assert "Not financial advice" in msg
    assert "No trades are placed" in msg


def test_message_no_data_at_all():
    msg = ws.format_weekly_message(
        make_insufficient(0, 0), make_insufficient(0, 0),
        None, None, 1, make_progress(),
    )
    assert "first weekly report" in msg


def test_message_insufficient_shows_counts():
    msg = ws.format_weekly_message(
        make_insufficient(12, 10), week_stats(), None, None, 5, make_progress(),
    )
    assert "Predictions this week: 12" in msg
    assert "Completed: 10" in msg
    assert "Insufficient data for accuracy stats" in msg


def test_message_never_raises_on_garbage():
    garbage = {"har_mae": "not-a-number", "insufficient": False}
    msg = ws.format_weekly_message(garbage, garbage, None, None, 5, {})
    assert isinstance(msg, str)
    assert len(msg) > 0


def test_message_profit_factor_na_when_none():
    stats = week_stats(profit_factor=None)
    msg = ws.format_weekly_message(stats, stats, None, None, 5, make_progress())
    assert "Profit factor:   N/A" in msg


# ─── send_weekly_report ────────────────────────────────────────────────────


def test_send_weekly_report_uses_plain_text():
    with patch(f"{SCRIPT}.send_message", return_value=SUCCESS) as send:
        result = ws.send_weekly_report(CONFIG, "hello")
    assert result.success
    _, kwargs = send.call_args
    assert kwargs.get("parse_mode") is None


# ─── main() ────────────────────────────────────────────────────────────────


def test_main_no_telegram_returns_1():
    with patch(
        "kronos_trading.alerts.telegram_sender.TelegramConfig.from_env",
        side_effect=EnvironmentError("missing token"),
    ):
        assert ws.main() == 1


def test_main_db_init_failure_returns_1():
    with patch(
        "kronos_trading.alerts.telegram_sender.TelegramConfig.from_env",
        return_value=CONFIG,
    ), patch(f"{SCRIPT}.initialize_db", side_effect=RuntimeError("db down")):
        assert ws.main() == 1


def test_main_success_returns_0():
    rows = make_week_rows(n=30)
    with patch(
        "kronos_trading.alerts.telegram_sender.TelegramConfig.from_env",
        return_value=CONFIG,
    ), patch(f"{SCRIPT}.initialize_db", return_value=None), patch(
        f"{SCRIPT}.fetch_week_predictions", return_value=rows
    ), patch(f"{SCRIPT}.fetch_pending_window", return_value=[]), patch(
        f"{SCRIPT}.compute_weeks_beating", return_value={"wins": 1, "total": 1}
    ), patch(f"{SCRIPT}.compute_calibration_day", return_value=5), patch(
        f"{SCRIPT}.send_weekly_report", return_value=SUCCESS
    ) as send:
        assert ws.main() == 0
        assert send.call_count == 1


def test_main_telegram_failure_returns_1():
    rows = make_week_rows(n=30)
    with patch(
        "kronos_trading.alerts.telegram_sender.TelegramConfig.from_env",
        return_value=CONFIG,
    ), patch(f"{SCRIPT}.initialize_db", return_value=None), patch(
        f"{SCRIPT}.fetch_week_predictions", return_value=rows
    ), patch(f"{SCRIPT}.fetch_pending_window", return_value=[]), patch(
        f"{SCRIPT}.compute_weeks_beating", return_value={"wins": 1, "total": 1}
    ), patch(f"{SCRIPT}.compute_calibration_day", return_value=5), patch(
        f"{SCRIPT}.send_weekly_report", return_value=FAILURE
    ):
        assert ws.main() == 1
