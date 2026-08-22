"""
Tests for HAR regime filter.

All tests:
- Deterministic
- No real DB connections
- No real network calls
- Mock psycopg
- Fast (< 1 second total)
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from freqtrade_har.strategies.har_regime_filter import (
    get_latest_har_prediction,
    get_regime,
    is_tradeable_regime,
    is_high_volatility,
    is_low_volatility,
    HARPrediction,
    REGIME_LOW,
    REGIME_MEDIUM,
    REGIME_HIGH,
    REGIME_UNKNOWN,
    MAX_PREDICTION_AGE_HOURS,
)


def make_fresh_timestamp() -> str:
    """Timestamp 30 minutes ago — not stale."""
    dt = datetime.now(timezone.utc) - timedelta(
        minutes=30)
    return dt.isoformat()


def make_stale_timestamp() -> str:
    """Timestamp 3 hours ago — stale."""
    dt = datetime.now(timezone.utc) - timedelta(
        hours=3)
    return dt.isoformat()


def make_mock_row(
    regime="low",
    fresh=True,
    asset="BTC/USDT",
    timeframe="1h",
    predicted_range=500.0,
):
    """Build a mock DB row dict."""
    ts = (make_fresh_timestamp()
          if fresh else make_stale_timestamp())
    return {
        "timestamp": "2026-08-22T10:00:00Z",
        "asset": asset,
        "timeframe": timeframe,
        "har_predicted_range": predicted_range,
        "regime": regime,
        "created_at": ts,
    }


def mock_psycopg_connect(row=None):
    """
    Create a mock psycopg connection
    that returns the given row.
    """
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = row
    mock_cur.__enter__ = lambda s: s
    mock_cur.__exit__ = MagicMock(
        return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(
        return_value=False)

    return mock_conn


# ─── get_latest_har_prediction tests ─────────

class TestGetLatestHARPrediction:

    def test_returns_none_no_db_url(self):
        """No URL → returns None, no error."""
        result = get_latest_har_prediction(
            db_url=None)
        assert result is None

    def test_returns_none_empty_db_url(self):
        """Empty string URL → returns None."""
        result = get_latest_har_prediction(
            db_url="")
        assert result is None

    def test_returns_none_on_db_exception(self):
        """DB connection failure → None."""
        with patch("psycopg.connect",
                   side_effect=Exception("conn fail")):
            result = get_latest_har_prediction(
                db_url="postgresql://fake")
        assert result is None

    def test_returns_none_no_rows(self):
        """No rows in DB → None."""
        mock_conn = mock_psycopg_connect(row=None)
        with patch("psycopg.connect",
                   return_value=mock_conn):
            result = get_latest_har_prediction(
                db_url="postgresql://fake")
        assert result is None

    def test_returns_prediction_fresh_low(self):
        """Fresh low regime → HARPrediction."""
        row = make_mock_row(
            regime="low", fresh=True)
        mock_conn = mock_psycopg_connect(row=row)
        with patch("psycopg.connect",
                   return_value=mock_conn):
            result = get_latest_har_prediction(
                db_url="postgresql://fake")
        assert result is not None
        assert result.regime == "low"
        assert result.is_stale is False

    def test_returns_prediction_fresh_medium(self):
        """Fresh medium regime → HARPrediction."""
        row = make_mock_row(
            regime="medium", fresh=True)
        mock_conn = mock_psycopg_connect(row=row)
        with patch("psycopg.connect",
                   return_value=mock_conn):
            result = get_latest_har_prediction(
                db_url="postgresql://fake")
        assert result is not None
        assert result.regime == "medium"

    def test_returns_prediction_fresh_high(self):
        """Fresh high regime → HARPrediction."""
        row = make_mock_row(
            regime="high", fresh=True)
        mock_conn = mock_psycopg_connect(row=row)
        with patch("psycopg.connect",
                   return_value=mock_conn):
            result = get_latest_har_prediction(
                db_url="postgresql://fake")
        assert result is not None
        assert result.regime == "high"
        assert result.is_stale is False

    def test_stale_prediction_flagged(self):
        """3h old prediction → is_stale=True."""
        row = make_mock_row(
            regime="low", fresh=False)
        mock_conn = mock_psycopg_connect(row=row)
        with patch("psycopg.connect",
                   return_value=mock_conn):
            result = get_latest_har_prediction(
                db_url="postgresql://fake")
        assert result is not None
        assert result.is_stale is True
        assert result.age_hours > MAX_PREDICTION_AGE_HOURS

    def test_age_hours_computed_correctly(self):
        """30 min old → age_hours ≈ 0.5."""
        row = make_mock_row(
            regime="medium", fresh=True)
        mock_conn = mock_psycopg_connect(row=row)
        with patch("psycopg.connect",
                   return_value=mock_conn):
            result = get_latest_har_prediction(
                db_url="postgresql://fake")
        assert result is not None
        assert 0.3 < result.age_hours < 0.7

    def test_all_fields_present(self):
        """All HARPrediction fields populated."""
        row = make_mock_row(
            regime="high",
            fresh=True,
            predicted_range=750.0)
        mock_conn = mock_psycopg_connect(row=row)
        with patch("psycopg.connect",
                   return_value=mock_conn):
            result = get_latest_har_prediction(
                db_url="postgresql://fake")
        assert result.timestamp is not None
        assert result.asset == "BTC/USDT"
        assert result.timeframe == "1h"
        assert result.har_predicted_range == 750.0
        assert result.regime == "high"
        assert isinstance(result.age_hours, float)
        assert isinstance(result.is_stale, bool)

    def test_psycopg_import_error(self):
        """psycopg not installed → None."""
        with patch.dict("sys.modules",
                        {"psycopg": None}):
            result = get_latest_har_prediction(
                db_url="postgresql://fake")
        assert result is None

    def test_none_regime_becomes_unknown(self):
        """None regime in DB → unknown."""
        row = make_mock_row(
            regime=None, fresh=True)
        mock_conn = mock_psycopg_connect(row=row)
        with patch("psycopg.connect",
                   return_value=mock_conn):
            result = get_latest_har_prediction(
                db_url="postgresql://fake")
        assert result is not None
        assert result.regime == REGIME_UNKNOWN


# ─── get_regime tests ────────────────────────

class TestGetRegime:

    def test_unknown_when_no_url(self):
        result = get_regime(db_url=None)
        assert result == REGIME_UNKNOWN

    def test_unknown_when_db_fails(self):
        with patch("psycopg.connect",
                   side_effect=Exception("fail")):
            result = get_regime(
                db_url="postgresql://fake")
        assert result == REGIME_UNKNOWN

    def test_unknown_when_stale(self):
        row = make_mock_row(
            regime="low", fresh=False)
        mock_conn = mock_psycopg_connect(row=row)
        with patch("psycopg.connect",
                   return_value=mock_conn):
            result = get_regime(
                db_url="postgresql://fake")
        assert result == REGIME_UNKNOWN

    def test_returns_low(self):
        row = make_mock_row(
            regime="low", fresh=True)
        mock_conn = mock_psycopg_connect(row=row)
        with patch("psycopg.connect",
                   return_value=mock_conn):
            result = get_regime(
                db_url="postgresql://fake")
        assert result == REGIME_LOW

    def test_returns_medium(self):
        row = make_mock_row(
            regime="medium", fresh=True)
        mock_conn = mock_psycopg_connect(row=row)
        with patch("psycopg.connect",
                   return_value=mock_conn):
            result = get_regime(
                db_url="postgresql://fake")
        assert result == REGIME_MEDIUM

    def test_returns_high(self):
        row = make_mock_row(
            regime="high", fresh=True)
        mock_conn = mock_psycopg_connect(row=row)
        with patch("psycopg.connect",
                   return_value=mock_conn):
            result = get_regime(
                db_url="postgresql://fake")
        assert result == REGIME_HIGH


# ─── is_tradeable_regime tests ───────────────

class TestIsTradeableRegime:

    def _mock_get_regime(self, regime):
        return patch(
            "freqtrade_har.strategies"
            ".har_regime_filter.get_regime",
            return_value=regime)

    def test_low_allowed_by_default(self):
        with self._mock_get_regime("low"):
            assert is_tradeable_regime(
                db_url="x") is True

    def test_medium_allowed_by_default(self):
        with self._mock_get_regime("medium"):
            assert is_tradeable_regime(
                db_url="x") is True

    def test_high_blocked_by_default(self):
        with self._mock_get_regime("high"):
            assert is_tradeable_regime(
                db_url="x") is False

    def test_unknown_treated_as_medium(self):
        """Unknown → medium → allowed by default."""
        with self._mock_get_regime("unknown"):
            assert is_tradeable_regime(
                db_url="x") is True

    def test_unknown_blocked_when_high_only(self):
        """Unknown → medium → not in [high]."""
        with self._mock_get_regime("unknown"):
            assert is_tradeable_regime(
                db_url="x",
                allow_regimes=["high"]) is False

    def test_high_allowed_when_specified(self):
        with self._mock_get_regime("high"):
            assert is_tradeable_regime(
                db_url="x",
                allow_regimes=["high"]) is True

    def test_custom_allow_list(self):
        with self._mock_get_regime("low"):
            assert is_tradeable_regime(
                db_url="x",
                allow_regimes=["medium",
                               "high"]) is False


# ─── is_high/low_volatility tests ────────────

class TestVolatilityHelpers:

    def _mock_regime(self, regime):
        return patch(
            "freqtrade_har.strategies"
            ".har_regime_filter.get_regime",
            return_value=regime)

    def test_is_high_true(self):
        with self._mock_regime("high"):
            assert is_high_volatility(
                db_url="x") is True

    def test_is_high_false_when_low(self):
        with self._mock_regime("low"):
            assert is_high_volatility(
                db_url="x") is False

    def test_is_low_true(self):
        with self._mock_regime("low"):
            assert is_low_volatility(
                db_url="x") is True

    def test_is_low_false_when_high(self):
        with self._mock_regime("high"):
            assert is_low_volatility(
                db_url="x") is False

    def test_is_high_false_unknown(self):
        with self._mock_regime("unknown"):
            assert is_high_volatility(
                db_url="x") is False

    def test_is_low_false_unknown(self):
        with self._mock_regime("unknown"):
            assert is_low_volatility(
                db_url="x") is False
