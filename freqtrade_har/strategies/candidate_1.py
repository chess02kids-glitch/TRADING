"""
Candidate 1: RSI + EMA200 Trend Filter

What it does:
- Same RSI(14) mean-reversion entry as the baseline.
- ADDITION: long entries only when close > EMA 200.
- Same ROI/stoploss structure as the baseline
  (exits unchanged so the only variable is the
  entry filter).

Why it should work (hypothesis):
The baseline RSI strategy failed because it took
oversold entries during downtrends — RSI
mean-reversion fought the prevailing trend and
those trades hit the -5% stop. In an uptrend,
oversold dips are mean-reverting pullbacks; in a
downtrend they are falling knives. EMA 200 is the
standard institutional long-term trend filter.
Filtering entries by it should cut the large
losses while keeping most of the wins.

HAR variant: A (baseline — no HAR filter).

Paper trading only. No live orders.
"""

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
)
from pandas import DataFrame
import talib.abstract as ta


class RSITrendFilter(IStrategy):
    """
    RSI mean-reversion restricted to uptrends
    (close > EMA 200). Candidate 1.

    Entry: RSI oversold AND close > EMA 200
    Exit: RSI overbought OR ROI OR stoploss
    """

    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False

    # Same exit structure as the RSI baseline
    minimal_roi = {
        "120": 0.005,
        "60": 0.01,
        "30": 0.02,
        "0": 0.04,
    }

    stoploss = -0.05
    trailing_stop = False

    # Pre-registered parameters (standard values,
    # chosen BEFORE seeing any backtest results)
    rsi_period = IntParameter(
        10, 30, default=14, space="buy",
        optimize=False)
    rsi_buy = IntParameter(
        20, 40, default=30, space="buy",
        optimize=False)
    rsi_sell = IntParameter(
        60, 80, default=70, space="sell",
        optimize=False)
    ema_trend = IntParameter(
        150, 250, default=200, space="buy",
        optimize=False)

    # EMA 200 needs warmup data before the
    # backtest window starts
    startup_candle_count = 220

    def populate_indicators(
        self,
        dataframe: DataFrame,
        metadata: dict,
    ) -> DataFrame:
        dataframe["rsi"] = ta.RSI(
            dataframe,
            timeperiod=self.rsi_period.value,
        )
        dataframe["ema_trend"] = ta.EMA(
            dataframe,
            timeperiod=self.ema_trend.value,
        )
        return dataframe

    def populate_entry_trend(
        self,
        dataframe: DataFrame,
        metadata: dict,
    ) -> DataFrame:
        """
        Entry: RSI oversold AND close > EMA 200.
        Uptrend filter only — never short.
        """
        dataframe.loc[
            (dataframe["rsi"] < self.rsi_buy.value)
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
        """Exit: RSI overbought (same as baseline)."""
        dataframe.loc[
            dataframe["rsi"] > self.rsi_sell.value,
            "exit_long",
        ] = 1
        return dataframe
