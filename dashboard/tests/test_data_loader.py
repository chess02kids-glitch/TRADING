"""Unit tests for dashboard.data_loader — DB access is fully mocked."""

import sys
import types
from datetime import timedelta

import pandas as pd

from dashboard.data_loader import (
    fetch_all_assets,
    fetch_breakouts,
    fetch_calibration_summary,
    fetch_predictions,
    get_db_url,
)


def _row(
    ts="2026-08-22T10:00:00+00:00",
    created="2026-08-22T10:05:00+00:00",
    predicted=400.0,
    actual=380.0,
    abs_err=20.0,
    err=-20.0,
    breakout=0,
    regime="high",
    asset="BTC/USDT",
):
    return {
        "id": 1,
        "timestamp": ts,
        "asset": asset,
        "timeframe": "1h",
        "har_predicted_range": predicted,
        "coef_b0": 1.0,
        "coef_b1": 0.2,
        "coef_b2": 0.3,
        "coef_b3": 0.4,
        "n_obs": 100,
        "regime": regime,
        "actual_range": actual,
        "prediction_error": err,
        "abs_prediction_error": abs_err,
        "breakout_flag": breakout,
        "created_at": created,
    }


def _install_psycopg_mock(monkeypatch, rows, capture=None, boom=None):
    """Inject a fake psycopg module used by data_loader internals."""

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, query, params=None):
            if capture is not None:
                capture.append((query, params))
            if boom is not None:
                raise boom

        def fetchall(self):
            return rows

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def cursor(self):
            return Cursor()

    fake = types.ModuleType("psycopg")

    def connect(*a, **k):
        if boom is not None and not capture:
            raise boom
        return Conn()

    fake.connect = connect
    rows_mod = types.ModuleType("psycopg.rows")
    rows_mod.dict_row = object()
    monkeypatch.setitem(sys.modules, "psycopg", fake)
    monkeypatch.setitem(sys.modules, "psycopg.rows", rows_mod)
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://example.invalid/db")


def test_get_db_url_from_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://example")
    assert get_db_url() == "postgresql://example"


def test_get_db_url_none_when_missing(monkeypatch):
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    assert get_db_url() is None


def test_fetch_predictions_empty_no_url(monkeypatch):
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    df = fetch_predictions()
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_fetch_predictions_returns_dataframe(monkeypatch):
    _install_psycopg_mock(monkeypatch, [_row()])
    df = fetch_predictions()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    for col in (
        "timestamp",
        "asset",
        "har_predicted_range",
        "actual_range",
        "created_at",
    ):
        assert col in df.columns


def test_fetch_predictions_completed_only_filter(monkeypatch):
    captured = []
    _install_psycopg_mock(monkeypatch, [_row()], capture=captured)
    fetch_predictions(completed_only=True)
    assert captured
    query, params = captured[0]
    assert "actual_range IS NOT NULL" in query
    assert params[0] == "BTC/USDT"


def test_fetch_predictions_timestamp_parsed(monkeypatch):
    _install_psycopg_mock(monkeypatch, [_row()])
    df = fetch_predictions()
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert str(df["timestamp"].dt.tz) != "None"


def test_fetch_predictions_sorted_ascending(monkeypatch):
    later = _row(ts="2026-08-22T12:00:00+00:00")
    earlier = _row(ts="2026-08-22T08:00:00+00:00")
    _install_psycopg_mock(monkeypatch, [later, earlier])
    df = fetch_predictions()
    assert list(df["timestamp"]) == sorted(df["timestamp"])
    assert df.iloc[0]["timestamp"] < df.iloc[-1]["timestamp"]


def test_fetch_all_assets_returns_dict(monkeypatch):
    monkeypatch.setattr(
        "dashboard.data_loader.fetch_predictions",
        lambda **kwargs: pd.DataFrame({"asset": [kwargs["asset"]]}),
    )
    result = fetch_all_assets()
    assert isinstance(result, dict)
    assert set(result) == {"BTC/USDT", "ETH/USDT"}


def test_fetch_calibration_summary_empty(monkeypatch):
    monkeypatch.setattr(
        "dashboard.data_loader.fetch_predictions",
        lambda **kwargs: pd.DataFrame(),
    )
    summary = fetch_calibration_summary()
    assert summary["total_predictions"] == 0
    assert summary["completed"] == 0
    assert summary["har_mae"] is None
    assert summary["har_beats"] is None
    assert summary["breakout_count"] == 0
    assert summary["regime_counts"] == {}
    assert summary["first_prediction_ts"] is None


def _completed_frame(n, predicted, actual, created_at=None):
    ts = pd.date_range("2026-08-01", periods=n, freq="h", tz="UTC")
    if created_at is None:
        created = ts
    else:
        created = pd.Series([created_at] * n)
    pred = list(predicted) if not isinstance(predicted, (int, float)) else [predicted] * n
    act = list(actual) if not isinstance(actual, (int, float)) else [actual] * n
    return pd.DataFrame({
        "timestamp": ts,
        "created_at": created,
        "har_predicted_range": pred,
        "actual_range": act,
        "abs_prediction_error": [abs(a - p) for a, p in zip(act, pred)],
        "prediction_error": [a - p for a, p in zip(act, pred)],
        "breakout_flag": [0] * n,
        "regime": ["high"] * n,
    })


def test_fetch_calibration_summary_har_beats(monkeypatch):
    # Alternating predictions close to actuals → HAR MAE ~0,
    # persistence (lag-1 pred) is far from actual.
    n = 24
    predicted = [100.0 if i % 2 == 0 else 200.0 for i in range(n)]
    actual = list(predicted)
    monkeypatch.setattr(
        "dashboard.data_loader.fetch_predictions",
        lambda **kwargs: _completed_frame(n, predicted, actual),
    )
    summary = fetch_calibration_summary()
    assert summary["har_mae"] is not None
    assert summary["persistence_mae"] is not None
    assert summary["har_mae"] < summary["persistence_mae"]
    assert summary["har_beats"] is True


def test_fetch_calibration_summary_not_beats(monkeypatch):
    # HAR is consistently off; lag-1 prediction matches later actuals.
    n = 24
    predicted = [100.0] * n
    actual = [200.0] + [100.0] * (n - 1)
    monkeypatch.setattr(
        "dashboard.data_loader.fetch_predictions",
        lambda **kwargs: _completed_frame(n, predicted, actual),
    )
    summary = fetch_calibration_summary()
    assert summary["har_mae"] is not None
    assert summary["persistence_mae"] is not None
    assert summary["har_mae"] > summary["persistence_mae"]
    assert summary["har_beats"] is False


def test_fetch_calibration_summary_insufficient(monkeypatch):
    monkeypatch.setattr(
        "dashboard.data_loader.fetch_predictions",
        lambda **kwargs: _completed_frame(10, 100.0, 101.0),
    )
    summary = fetch_calibration_summary()
    assert summary["completed"] == 10
    assert summary["har_mae"] is None
    assert summary["har_beats"] is None


def test_fetch_calibration_day_computed(monkeypatch):
    # Implementation: day = int(elapsed_days) + 1.
    # 4 full days elapsed → calibration day 5.
    first = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=4)
    monkeypatch.setattr(
        "dashboard.data_loader.fetch_predictions",
        lambda **kwargs: _completed_frame(24, 100.0, 101.0, created_at=first),
    )
    summary = fetch_calibration_summary()
    assert summary["calibration_day"] == 5
    assert summary["days_remaining"] == 25


def test_fetch_breakouts_returns_dataframe(monkeypatch):
    captured = []
    rows = [{
        "timestamp": "2026-08-22T11:00:00+00:00",
        "asset": "BTC/USDT",
        "timeframe": "1h",
        "har_predicted_range": 100.0,
        "actual_range": 250.0,
        "ratio": 2.50,
        "created_at": "2026-08-22T11:05:00+00:00",
    }]
    _install_psycopg_mock(monkeypatch, rows, capture=captured)
    df = fetch_breakouts(limit=20)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "ratio" in df.columns
    assert captured
    assert captured[0][1] == (20,)


def test_fetch_predictions_handles_exception(monkeypatch):
    _install_psycopg_mock(
        monkeypatch,
        [],
        boom=RuntimeError("db down"),
    )
    df = fetch_predictions()
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_fetch_breakouts_empty_no_url(monkeypatch):
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    df = fetch_breakouts()
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_fetch_predictions_empty_result_set(monkeypatch):
    _install_psycopg_mock(monkeypatch, [])
    df = fetch_predictions()
    assert df.empty
