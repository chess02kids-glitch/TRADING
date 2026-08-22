"""
Tests for the winner strategies (Volatility Breakout family).

All tests:
- Deterministic (seeded synthetic data)
- No network, no DB, no freqtrade backtest engine
- Fast (< 1 second total)

CRITICAL REGRESSION TEST:
The channel-exit bug. An earlier implementation used

    low.rolling(N).min()

which INCLUDES the current bar. Because a bar's
close is always >= its own low, the condition

    close < lowest_low

was structurally impossible — the exit signal
never fired. The correct formulation excludes
the current bar:

    low.rolling(N).min().shift(1)

test_exit_signal_fires_on_channel_break guards
this exact regression.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from freqtrade_har.strategies.winner_baseline import WinnerBaseline
from freqtrade_har.strategies.winner_har_filtered import (
    WinnerHARFiltered,
    HAR_AVAILABLE as HAR_AVAILABLE_B,
)
from freqtrade_har.strategies.winner_har_inverse import (
    WinnerHARInverse,
    HAR_AVAILABLE as HAR_AVAILABLE_C,
)


def make_frame(n=300, start=100.0, drift=0.05):
    """
    Synthetic OHLCV frame:
    gentle deterministic uptrend so that
    close > EMA 200 holds after warmup.
    """
    idx = pd.date_range(
        "2024-01-01", periods=n, freq="1h", tz="UTC")
    close = start + drift * np.arange(n)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = np.full(n, 100.0)
    return pd.DataFrame({
        "date": idx,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def make_expansion_bar(df, i, jump=3.0):
    """
    Replace bar i with a large bullish expansion
    bar (range >> 2x the quiet 1.0 range).
    """
    prev_close = df["close"].iloc[i - 1]
    open_ = prev_close
    close = open_ + jump
    df.loc[df.index[i], ["open", "high", "low", "close"]] = [
        open_,
        max(open_, close) + 0.2,
        min(open_, close) - 0.2,
        close,
    ]
    return df


def populate(strategy, df, pair="BTC/USDT"):
    df = strategy.populate_indicators(df.copy(), {"pair": pair})
    df = strategy.populate_entry_trend(df, {"pair": pair})
    df = strategy.populate_exit_trend(df, {"pair": pair})
    return df


STRATEGIES = [
    (WinnerBaseline, "baseline"),
    (WinnerHARFiltered, "har_filtered"),
    (WinnerHARInverse, "har_inverse"),
]

TEST_CONFIG = {
    "max_open_trades": 3,
    "stake_currency": "USDT",
    "stake_amount": 100,
    "tradable_balance_ratio": 0.99,
    "dry_run": True,
    "dry_run_wallet": 1000,
    "timeframe": "1h",
    "exchange": {"name": "kucoin"},
    "telegram": {"enabled": False},
    "api_server": {"enabled": False},
}


def make_strategy(cls):
    return cls(config=TEST_CONFIG.copy())


from contextlib import contextmanager


@contextmanager
def regime_allowed(cls):
    """
    Force the HAR regime check to ALLOW entries
    so entry-logic tests exercise the strategy
    conditions, not the DB fallback path.
    Baseline (no HAR) is unaffected.
    """
    if cls is WinnerHARFiltered:
        with patch(
            "freqtrade_har.strategies.winner_har_filtered"
            ".is_tradeable_regime",
            return_value=True,
        ):
            yield
    elif cls is WinnerHARInverse:
        with patch(
            "freqtrade_har.strategies.winner_har_inverse"
            ".is_tradeable_regime",
            return_value=True,
        ):
            yield
    else:
        yield


class TestInterface:

    @pytest.mark.parametrize("cls,_", STRATEGIES)
    def test_interface_attributes(self, cls, _):
        s = make_strategy(cls)
        assert s.INTERFACE_VERSION == 3
        assert s.timeframe == "1h"
        assert s.can_short is False
        assert s.stoploss == -0.05
        assert s.minimal_roi == {}
        assert s.trailing_stop is False

    def test_har_available_flags(self):
        """HAR module must be importable in tests."""
        assert HAR_AVAILABLE_B is True
        assert HAR_AVAILABLE_C is True


class TestIndicators:

    @pytest.mark.parametrize("cls,_", STRATEGIES)
    def test_indicator_columns_present(self, cls, _):
        df = make_frame()
        df = make_strategy(cls).populate_indicators(df, {"pair": "BTC/USDT"})
        for col in ["candle_range", "avg_range",
                    "ema_trend", "lowest_low"]:
            assert col in df.columns, f"missing {col}"
        assert df["candle_range"].notna().all()


class TestEntryLogic:

    @pytest.mark.parametrize("cls,_", STRATEGIES)
    def test_entry_fires_on_expansion_bar(self, cls, _):
        df = make_frame(n=250)
        make_expansion_bar(df, 240)  # big bullish bar
        with regime_allowed(cls):
            df = populate(make_strategy(cls), df)
        entries = df["enter_long"].fillna(0)
        assert entries.iloc[240] == 1, (
            "expansion bar in uptrend must trigger entry")
        assert entries.sum() == 1

    @pytest.mark.parametrize("cls,_", STRATEGIES)
    def test_no_entry_without_expansion(self, cls, _):
        """Quiet bars: range condition blocks entry."""
        df = make_frame(n=250)
        with regime_allowed(cls):
            df = populate(make_strategy(cls), df)
        entries = df["enter_long"].fillna(0)
        assert entries.sum() == 0

    @pytest.mark.parametrize("cls,_", STRATEGIES)
    def test_no_entry_below_ema_trend(self, cls, _):
        """Expansion bar below EMA 200: no entry."""
        df = make_frame(n=250, start=200.0, drift=-0.5)
        make_expansion_bar(df, 240)
        with regime_allowed(cls):
            df = populate(make_strategy(cls), df)
        entries = df["enter_long"].fillna(0)
        assert entries.sum() == 0

    def test_volume_zero_blocks_entry(self):
        df = make_frame(n=250)
        make_expansion_bar(df, 240)
        df.loc[df.index[240], "volume"] = 0.0
        df = populate(make_strategy(WinnerBaseline), df)
        assert df["enter_long"].fillna(0).sum() == 0


class TestExitLogic:

    def test_exit_signal_fires_on_channel_break(self):
        """
        REGRESSION TEST (the exit bug):
        after an uptrend, a bar that closes below
        the PRIOR 5-bar lows must set exit_long.
        The buggy rolling-min-without-shift
        implementation fails this test because
        close < min(including current low) is
        impossible.
        """
        df = make_frame(n=250)
        make_expansion_bar(df, 200)
        # five consecutive falling bars that
        # close below the prior channel lows
        for k in range(1, 6):
            i = 200 + k
            prev_close = df["close"].iloc[i - 1]
            close = prev_close - 2.0
            open_ = prev_close
            df.loc[df.index[i], ["open", "high", "low", "close"]] = [
                open_,
                max(open_, close) + 0.1,
                min(open_, close) - 0.1,
                close,
            ]
        df = populate(make_strategy(WinnerBaseline), df)
        exits = df["exit_long"].fillna(0)
        assert exits.iloc[205] == 1, (
            "channel break must trigger exit — "
            "regression of the shift(1) bug")

    def test_no_exit_without_channel_break(self):
        """Small pullback above channel low: no exit."""
        df = make_frame(n=250)
        make_expansion_bar(df, 200)
        # mild pullback that stays above prior lows
        for k in range(1, 4):
            i = 200 + k
            prev_close = df["close"].iloc[i - 1]
            close = prev_close - 0.3
            open_ = prev_close
            df.loc[df.index[i], ["open", "high", "low", "close"]] = [
                open_,
                max(open_, close) + 0.1,
                min(open_, close) - 0.1,
                close,
            ]
        df = populate(make_strategy(WinnerBaseline), df)
        exits = df["exit_long"].fillna(0)
        assert exits.sum() == 0


class TestHARVariants:

    def _signal_frame(self):
        df = make_frame(n=250)
        make_expansion_bar(df, 240)
        return df

    def test_filtered_blocks_entry_when_high_regime(self):
        df = self._signal_frame()
        with patch(
            "freqtrade_har.strategies.winner_har_filtered"
            ".is_tradeable_regime",
            return_value=False,
        ):
            df = populate(make_strategy(WinnerHARFiltered), df)
        assert df["enter_long"].fillna(0).sum() == 0, (
            "HIGH regime must block entries")

    def test_filtered_allows_entry_when_tradeable(self):
        df = self._signal_frame()
        with patch(
            "freqtrade_har.strategies.winner_har_filtered"
            ".is_tradeable_regime",
            return_value=True,
        ):
            df = populate(make_strategy(WinnerHARFiltered), df)
        assert df["enter_long"].fillna(0).iloc[240] == 1

    def test_inverse_blocks_entry_when_not_high(self):
        df = self._signal_frame()
        with patch(
            "freqtrade_har.strategies.winner_har_inverse"
            ".is_tradeable_regime",
            return_value=False,
        ):
            df = populate(make_strategy(WinnerHARInverse), df)
        assert df["enter_long"].fillna(0).sum() == 0

    def test_exits_never_blocked_by_har(self):
        """
        HAR failure / block must NEVER affect exits.
        """
        df = self._signal_frame()
        for k in range(1, 6):
            i = 200 + k
            prev_close = df["close"].iloc[i - 1]
            close = prev_close - 2.0
            df.loc[df.index[i], ["open", "high", "low", "close"]] = [
                prev_close,
                prev_close + 0.1,
                min(prev_close, close) - 0.1,
                close,
            ]
        with patch(
            "freqtrade_har.strategies.winner_har_filtered"
            ".is_tradeable_regime",
            side_effect=Exception("DB down"),
        ):
            df = populate(make_strategy(WinnerHARFiltered), df)
        exits = df["exit_long"].fillna(0)
        assert exits.iloc[205] == 1, (
            "exits must never be blocked by HAR")
