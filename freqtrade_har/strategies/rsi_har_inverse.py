"""
Strategy C: RSI + Inverse HAR Filter (Control)

The control strategy.
OPPOSITE of Strategy B:
Only trades during HIGH volatility.
Skips LOW and MEDIUM.

Research purpose:
If Strategy B (skip high vol) beats A,
and Strategy C (only high vol) loses to A,
then HAR regime has genuine value.

If C beats A, the hypothesis is wrong.
High volatility might actually be BETTER
for RSI entries (e.g. mean reversion).

Paper trading only. No live orders.
"""

import logging
from freqtrade.strategy import (
    IStrategy,
    IntParameter,
)
from pandas import DataFrame
import talib.abstract as ta

logger = logging.getLogger(__name__)

try:
    from freqtrade_har.strategies.har_regime_filter import (
        is_tradeable_regime,
        REGIME_HIGH,
    )
    HAR_AVAILABLE = True
except ImportError:
    HAR_AVAILABLE = False


class RSIHARInverse(IStrategy):
    """
    RSI strategy with inverse HAR filter.
    Strategy C — control experiment.

    ONLY trades in HIGH volatility regime.
    """

    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False

    minimal_roi = {
        "120": 0.005,
        "60": 0.01,
        "30": 0.02,
        "0": 0.04,
    }

    stoploss = -0.05
    trailing_stop = False

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
        """
        INVERSE: only enter in HIGH regime.
        """
        pair = metadata.get("pair", "BTC/USDT")

        regime_ok = True
        if HAR_AVAILABLE:
            try:
                regime_ok = is_tradeable_regime(
                    asset=pair,
                    timeframe=self.timeframe,
                    allow_regimes=[REGIME_HIGH],
                )
            except Exception as e:
                logger.warning(
                    f"HAR check failed: {e}.")
                regime_ok = True

        if not regime_ok:
            dataframe["enter_long"] = 0
            return dataframe

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
        dataframe.loc[
            dataframe["rsi"] > self.rsi_sell.value,
            "exit_long",
        ] = 1
        return dataframe
