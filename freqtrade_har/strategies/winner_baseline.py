"""
Winner: Volatility Breakout — Baseline (A)

What it does:
- Enters long when a candle's range
  (high - low) exceeds 2.0x the average range
  of the last 20 candles AND the candle is
  bullish (close > open) AND close > EMA 200.
- Exits when close breaks below the lowest low
  of the last 10 candles. No ROI cap — winners
  are allowed to run (trend following).
- Stoploss -5% below entry.

Why it should work (hypothesis):
The baseline failed because its exits capped
winners at +0.5-4% while losses ran to -5% —
a payoff structure that loses even at 79% win
rate. A trend-following system inverts that
structure: small frequent losses (failed
breakouts) and occasional large winners (trend
capture). In a window where the market moved
+68%, a long-only breakout system should capture
a fraction of that move. Range expansion is the
same volatility signal HAR models — this is the
most natural candidate for HAR integration.

HAR variant: A (baseline — no HAR filter).

Paper trading only. No live orders.
"""

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
)
from pandas import DataFrame


class WinnerBaseline(IStrategy):
    """
    Range-expansion breakout, long only.
    Candidate 3.

    Entry: range > 2x avg range, bullish bar,
           close > EMA 200
    Exit: close < 10-bar low OR stoploss
    """

    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False

    # No ROI cap: trend following needs to
    # let winners run. Exits via signal/stop.
    minimal_roi = {}

    stoploss = -0.05
    trailing_stop = False

    # Pre-registered parameters (standard
    # breakout values, chosen a priori)
    range_period = IntParameter(
        10, 30, default=20, space="buy",
        optimize=False)
    range_mult = DecimalParameter(
        1.5, 3.0, default=2.0, space="buy",
        optimize=False)
    ema_trend = IntParameter(
        150, 250, default=200, space="buy",
        optimize=False)
    exit_lookback = IntParameter(
        5, 20, default=10, space="sell",
        optimize=False)

    startup_candle_count = 220

    def populate_indicators(
        self,
        dataframe: DataFrame,
        metadata: dict,
    ) -> DataFrame:
        dataframe["candle_range"] = (
            dataframe["high"] - dataframe["low"]
        )
        dataframe["avg_range"] = dataframe[
            "candle_range"
        ].rolling(window=self.range_period.value).mean()
        dataframe["ema_trend"] = dataframe["close"].ewm(
            span=self.ema_trend.value,
            adjust=False,
        ).mean()
        dataframe["lowest_low"] = dataframe["low"].rolling(
            window=self.exit_lookback.value,
            min_periods=1,
        ).min().shift(1)
        return dataframe

    def populate_entry_trend(
        self,
        dataframe: DataFrame,
        metadata: dict,
    ) -> DataFrame:
        """
        Entry: volatility expansion + bullish bar
        + uptrend. Momentum breakout.
        """
        dataframe.loc[
            (dataframe["candle_range"]
             > self.range_mult.value * dataframe["avg_range"])
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["close"] > dataframe["ema_trend"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(
        self,
        dataframe: DataFrame,
        metadata: dict,
    ) -> DataFrame:
        """
        Exit: close below the 10-bar low.
        Channel exit — trend over / stop-out.
        """
        dataframe.loc[
            dataframe["close"] < dataframe["lowest_low"],
            "exit_long",
        ] = 1
        return dataframe
