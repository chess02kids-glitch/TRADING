"""
Strategy A: RSI Baseline (No HAR Filter)

The benchmark strategy.
Simple RSI mean-reversion.
No volatility regime filter applied.
Trades in all market conditions.

Purpose: Establish baseline performance
before adding HAR filter.

Paper trading only. No live orders.
"""

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
)
from pandas import DataFrame
import talib.abstract as ta


class RSIBaseline(IStrategy):
    """
    RSI strategy with no HAR filter.
    Strategy A — baseline comparison.

    Entry: RSI oversold
    Exit: RSI overbought OR ROI OR stoploss
    """

    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False

    # Exit after reaching profit targets
    minimal_roi = {
        "120": 0.005,
        "60": 0.01,
        "30": 0.02,
        "0": 0.04,
    }

    stoploss = -0.05
    trailing_stop = False

    # Hyperparameters
    rsi_period = IntParameter(
        10, 30, default=14, space="buy",
        optimize=False)
    rsi_buy = IntParameter(
        20, 40, default=30, space="buy",
        optimize=False)
    rsi_sell = IntParameter(
        60, 80, default=70, space="sell",
        optimize=False)

    def populate_indicators(
        self,
        dataframe: DataFrame,
        metadata: dict,
    ) -> DataFrame:
        dataframe["rsi"] = ta.RSI(
            dataframe,
            timeperiod=self.rsi_period.value,
        )
        return dataframe

    def populate_entry_trend(
        self,
        dataframe: DataFrame,
        metadata: dict,
    ) -> DataFrame:
        """Enter when RSI oversold."""
        dataframe.loc[
            (dataframe["rsi"] < self.rsi_buy.value)
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(
        self,
        dataframe: DataFrame,
        metadata: dict,
    ) -> DataFrame:
        """Exit when RSI overbought."""
        dataframe.loc[
            dataframe["rsi"] > self.rsi_sell.value,
            "exit_long",
        ] = 1
        return dataframe
