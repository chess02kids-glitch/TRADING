import numpy as np
import pandas as pd
from datetime import datetime
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
from freqtrade.persistence import Trade
from kronos_trading.volatility_baselines import har_forecast, HAR_MIN_TRAIN

def compute_har_predictions(highs: pd.Series, lows: pd.Series) -> pd.Series:
    ranges = highs - lows
    predictions = pd.Series(np.nan, index=ranges.index, dtype=float)
    for i in range(HAR_MIN_TRAIN, len(ranges)):
        hist_ranges = ranges.iloc[:i]
        try:
            pred = har_forecast(hist_ranges)
            predictions.iloc[i] = pred
        except Exception:
            pass
    return predictions

class HARStopDynamic(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False
    
    use_custom_stoploss = True

    minimal_roi = {
        "120": 0.005,
        "60": 0.01,
        "30": 0.02,
        "0": 100
    }

    stoploss = -0.99
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
        
        dataframe["har_range"] = compute_har_predictions(dataframe["high"], dataframe["low"])
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

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        trade_date = trade.open_date_utc
        
        try:
            har_val = dataframe.loc[dataframe["date"] == trade_date, "har_range"].values
            if len(har_val) > 0 and not np.isnan(har_val[0]) and har_val[0] > 0 and trade.open_rate > 0:
                stop = -(1.5 * har_val[0] / trade.open_rate)
                return stop
        except Exception:
            pass
            
        return -0.05
