import numpy as np
import pandas as pd
from datetime import datetime
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
from freqtrade.persistence import Trade

class HARStopBaseline(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False

    minimal_roi = {
        "120": 0.005,
        "60": 0.01,
        "30": 0.02,
        "0": 100
    }

    stoploss = -0.05
    trailing_stop = False

    range_period = IntParameter(10, 30, default=20, space="buy", optimize=False)
    range_mult = DecimalParameter(1.5, 3.0, default=2.0, space="buy", optimize=False)
    ema_trend = IntParameter(150, 250, default=200, space="buy", optimize=False)
    exit_lookback = IntParameter(5, 20, default=10, space="sell", optimize=False)

    startup_candle_count = 220

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["candle_range"] = dataframe["high"] - dataframe["low"]
        dataframe["avg_range"] = dataframe["candle_range"].rolling(window=self.range_period.value).mean()
        dataframe["ema_trend"] = dataframe["close"].ewm(span=self.ema_trend.value, adjust=False).mean()
        dataframe["lowest_low"] = dataframe["low"].rolling(window=self.exit_lookback.value, min_periods=1).min().shift(1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["candle_range"] > self.range_mult.value * dataframe["avg_range"])
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["close"] > dataframe["ema_trend"])
            & (dataframe["volume"] > 0),
            "enter_long"
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            dataframe["close"] < dataframe["lowest_low"],
            "exit_long"
        ] = 1
        return dataframe
