"""
Strategy B: RSI + HAR Regime Filter

The experimental strategy.
Same RSI logic as Strategy A.
Adds HAR volatility regime as entry filter.

HAR filter logic:
- LOW regime:    trade normally
- MEDIUM regime: trade normally
- HIGH regime:   SKIP entry
- UNKNOWN:       treat as MEDIUM (safe fallback)

Research hypothesis:
Filtering out HIGH volatility entries
improves risk-adjusted returns vs baseline.

IMPORTANT: HAR filter only blocks ENTRIES.
It never blocks EXITS.
Once in a trade, we always allow exits.

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

# Import HAR filter
# Graceful fallback if not available
try:
    from freqtrade_har.strategies.har_regime_filter import (
        is_tradeable_regime,
        REGIME_LOW,
        REGIME_MEDIUM,
    )
    HAR_AVAILABLE = True
except ImportError:
    HAR_AVAILABLE = False
    logger.warning(
        "har_regime_filter not found. "
        "Strategy B running without HAR filter.")


class RSIHARFiltered(IStrategy):
    """
    RSI strategy filtered by HAR regime.
    Strategy B — experimental variant.

    Only enters trades in low/medium volatility.
    Skips high volatility entries.
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

    # Regimes that allow entry
    ALLOWED_REGIMES = [REGIME_LOW, REGIME_MEDIUM] \
        if HAR_AVAILABLE else ["low", "medium"]

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
        Entry: RSI oversold AND HAR regime ok.

        NOTE ON BACKTESTING:
        During freqtrade backtesting, this runs
        against historical data with no live DB.
        The HAR filter reads Supabase in real time
        which is unavailable during backtest.
        Therefore backtest results = RSI only.
        Live paper trading = RSI + HAR filter.
        Both measurements are needed.
        """
        pair = metadata.get("pair", "BTC/USDT")

        # Check HAR regime
        # This only works in live paper trading
        # During backtesting: DB unavailable
        # so filter is bypassed automatically
        regime_ok = True
        if HAR_AVAILABLE:
            try:
                regime_ok = is_tradeable_regime(
                    asset=pair,
                    timeframe=self.timeframe,
                    allow_regimes=self.ALLOWED_REGIMES,
                )
                if not regime_ok:
                    logger.info(
                        f"HAR BLOCKING {pair}: "
                        f"high volatility regime")
            except Exception as e:
                logger.warning(
                    f"HAR check failed: {e}. "
                    f"Allowing entry (fallback).")
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
        """
        Exit: RSI overbought.
        HAR filter NEVER blocks exits.
        Always allow position closure.
        """
        dataframe.loc[
            dataframe["rsi"] > self.rsi_sell.value,
            "exit_long",
        ] = 1
        return dataframe
