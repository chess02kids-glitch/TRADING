"""
HAR Volatility Actor for NautilusTrader.

Reads HAR predictions and publishes
volatility signals to strategies.

In backtest: reads pre-computed predictions
             from the data loading stage
In live: reads current predictions from
         Supabase (same code, different data)

This is the key advantage of NautilusTrader:
Same actor code works in both modes.
"""

try:
    from nautilus_trader.core.data import Data
    from nautilus_trader.model.data import Bar
    from nautilus_trader.trading.actor import Actor
    from nautilus_trader.common.component import Logger
    class _TestData(Data): pass
except Exception:
    class Data: pass
    class Bar: pass
    class Actor:
        def __init__(self, *args, **kwargs): pass
        def subscribe_bars(self, *args, **kwargs): pass
        def publish_data(self, *args, **kwargs): pass
    class Logger: pass
from dataclasses import dataclass
import pandas as pd


@dataclass
class HARVolatilitySignal(Data):
    """
    Signal published by HARVolatilityActor.
    Consumed by volatility targeting strategies.
    """
    instrument_id: str
    predicted_range: float
    current_price: float
    volatility_estimate: float  # range / price
    regime: str                 # low/medium/high
    ts_event: int               # nanoseconds UTC
    ts_init: int


class HARVolatilityActor(Actor):
    """
    Reads HAR predictions and publishes
    HARVolatilitySignal data to the engine.

    Configuration:
      predictions: dict mapping
        InstrumentId → pd.Series of predictions
      target_vol: float (the target volatility)
    """

    def __init__(
        self,
        predictions: dict,
        target_vol: float = 0.02,
    ):
        super().__init__()
        self._predictions = predictions
        self._target_vol = target_vol

    def on_start(self) -> None:
        """Subscribe to bar data."""
        for instrument_id in (
            self._predictions.keys()):
            self.subscribe_bars(
                bar_type=self._get_bar_type(
                    instrument_id))

    def on_bar(self, bar: Bar) -> None:
        """
        On each bar, look up HAR prediction
        and publish a volatility signal.
        """
        instrument_id = str(
            bar.bar_type.instrument_id)

        if instrument_id not in (
            self._predictions):
            return

        preds = self._predictions[instrument_id]
        bar_time = pd.Timestamp(
            bar.ts_event, unit="ns", tz="UTC")

        # Find nearest prediction
        pred = self._lookup_prediction(
            preds, bar_time)

        if pred is None or pred <= 0:
            return

        close = float(bar.close)
        if close <= 0:
            return

        vol_estimate = pred / close

        # Classify regime
        regime = self._classify_regime(
            vol_estimate)

        signal = HARVolatilitySignal(
            instrument_id=instrument_id,
            predicted_range=pred,
            current_price=close,
            volatility_estimate=vol_estimate,
            regime=regime,
            ts_event=bar.ts_event,
            ts_init=bar.ts_init,
        )

        self.publish_data(
            data_type=type(signal),
            data=signal)

    def _lookup_prediction(
        self,
        preds: pd.Series,
        bar_time: pd.Timestamp,
    ) -> float | None:
        """Find nearest prediction to bar time."""
        try:
            if preds.empty:
                return None
            idx = preds.index.get_indexer(
                [bar_time], method="nearest")
            if idx[0] < 0:
                return None
            val = preds.iloc[idx[0]]
            if pd.isna(val):
                return None
            return float(val)
        except Exception:
            return None

    def _classify_regime(
        self,
        vol_estimate: float,
    ) -> str:
        """
        Classify volatility regime.
        LOW:    vol < 1% per bar
        MEDIUM: 1% <= vol < 3%
        HIGH:   vol >= 3%
        """
        if vol_estimate < 0.01:
            return "low"
        elif vol_estimate < 0.03:
            return "medium"
        else:
            return "high"

    def _get_bar_type(self, instrument_id: str):
        """Build bar type for subscription."""
        from nautilus_trader.model.data import (
            BarType, BarSpecification)
        from nautilus_trader.model.enums import (
            BarAggregation, PriceType)
        from nautilus_trader.model.identifiers import (
            InstrumentId)
        return BarType(
            instrument_id=InstrumentId.from_str(
                instrument_id),
            bar_spec=BarSpecification(
                step=1,
                aggregation=BarAggregation.HOUR,
                price_type=PriceType.LAST),
        )
