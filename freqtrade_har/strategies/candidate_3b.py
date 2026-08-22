"""
Candidate 3b: Volatility Breakout — Fast Rotation
(OPTION B modification of candidate_3, PRE-REGISTERED)

What changed (and why — mechanism, not curve-fit):
- range_mult: 2.0x -> 1.5x
  (1.5x average range is a ~1.5-sigma expansion,
  a standard lower threshold for breakout entries)
- exit_lookback: 10 -> 5 bars
  (5-bar low is the minimum channel exit that
  still requires a real pullback)

Why this modification is pre-registered:
Candidate 3 (the selected winner) had POSITIVE
expectancy (+11.21%, PF 2.35, DD 8.31%) but only
18 trades in 2 years. The binding constraint was
diagnosed as position rotation speed: avg trade
duration of 75 days means each pair completes
only ~9 cycles in the whole window. The
pre-registered MUST-HAVE is >= 50 trades. Both
changes widen the entry funnel and shorten the
holding period — they target TRADE FREQUENCY,
the pre-registered requirement, not profit.
Parameter values are standard breakout values
chosen BEFORE this retest.

Everything else is identical to candidate_3:
- Entry: range > mult x avg range(20), bullish
  bar (close > open), close > EMA 200.
- Exit: close < lowest low of last N bars.
- No ROI cap. Stoploss -5%.

HAR variant: A (baseline — no HAR filter).

Paper trading only. No live orders.
"""

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
)
from pandas import DataFrame


class VolatilityBreakoutFast(IStrategy):
    """
    Range-expansion breakout, long only.
    Candidate 3b — fast-rotation variant.

    Entry: range > 1.5x avg range, bullish bar,
           close > EMA 200
    Exit: close < 5-bar low OR stoploss
    """

    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False

    # No ROI cap: trend following needs to
    # let winners run. Exits via signal/stop.
    minimal_roi = {}

    stoploss = -0.05
    trailing_stop = False

    # Pre-registered modified parameters
    range_period = IntParameter(
        10, 30, default=20, space="buy",
        optimize=False)
    range_mult = DecimalParameter(
        1.5, 3.0, default=1.5, space="buy",
        optimize=False)
    ema_trend = IntParameter(
        150, 250, default=200, space="buy",
        optimize=False)
    exit_lookback = IntParameter(
        5, 20, default=5, space="sell",
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
        Exit: close below the N-bar low.
        Channel exit — trend over / stop-out.
        """
        dataframe.loc[
            dataframe["close"] < dataframe["lowest_low"],
            "exit_long",
        ] = 1
        return dataframe
