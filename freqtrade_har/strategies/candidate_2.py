"""
Candidate 2: Bollinger Band Mean Reversion

What it does:
- Enters long when close < lower Bollinger band
  (20-period SMA, 2.0 standard deviations).
- Exits when close > middle band (the 20-period
  SMA). No ROI cap — the middle band is the
  profit target.
- Stoploss -5% below entry.

Why it should work (hypothesis):
Bollinger bands are volatility-normalized: the
entry zone widens when volatility is high, so the
signal is comparable across regimes (this is the
same principle HAR uses — range relative to
recent range). Mean reversion to the middle band
captures a target proportional to current
volatility instead of a fixed +0.5-4% ROI, which
was the baseline's structural weakness (small
wins vs -5% losses). In trending-up markets,
lower-band touches are buyable pullbacks.

HAR variant: A (baseline — no HAR filter).

Paper trading only. No live orders.
"""

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
)
from pandas import DataFrame


class BollingerReversion(IStrategy):
    """
    Bollinger lower-band mean reversion.
    Candidate 2.

    Entry: close < lower band
    Exit: close > middle band OR stoploss
    """

    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False

    # No ROI cap: the middle band is the target.
    # This is intentional strategy design —
    # a fixed small ROI repeats the baseline's
    # win/loss asymmetry problem.
    minimal_roi = {}

    stoploss = -0.05
    trailing_stop = False

    # Pre-registered parameters (standard values:
    # 20-period SMA with 2.0 sigma bands)
    bb_period = IntParameter(
        10, 30, default=20, space="buy",
        optimize=False)
    bb_std = DecimalParameter(
        1.5, 3.0, default=2.0, space="buy",
        optimize=False)

    startup_candle_count = 40

    def populate_indicators(
        self,
        dataframe: DataFrame,
        metadata: dict,
    ) -> DataFrame:
        period = self.bb_period.value
        stddev = self.bb_std.value

        dataframe["bb_mid"] = dataframe["close"].rolling(
            window=period).mean()
        dataframe["bb_std"] = dataframe["close"].rolling(
            window=period).std(ddof=0)
        dataframe["bb_lower"] = (
            dataframe["bb_mid"] - stddev * dataframe["bb_std"]
        )

        # Make sure the middle band never
        # triggers an exit on the entry bar
        # (band values are computed from close,
        # so they are known at candle close)
        return dataframe

    def populate_entry_trend(
        self,
        dataframe: DataFrame,
        metadata: dict,
    ) -> DataFrame:
        """
        Entry: close below the lower band.
        Mean-reversion entry — buy the dip.
        """
        dataframe.loc[
            (dataframe["close"] < dataframe["bb_lower"])
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
        Exit: close back above the middle band.
        Target = middle band (volatility-scaled).
        """
        dataframe.loc[
            dataframe["close"] > dataframe["bb_mid"],
            "exit_long",
        ] = 1
        return dataframe
