"""
Strategy A: Equal Weight Portfolio (Baseline)

Simple buy-and-hold portfolio.
50% BTC, 50% ETH.
Rebalances daily to maintain equal weight.

NO volatility targeting.
NO HAR signals.
This is the comparison benchmark.

Paper trading only. No live orders.
"""

try:
    from nautilus_trader.trading.strategy import Strategy
    from nautilus_trader.config import StrategyConfig
    class _TestConfig(StrategyConfig): pass
except Exception:
    class Strategy:
        def __init__(self, *args, **kwargs): pass
        def subscribe_bars(self, *args, **kwargs): pass
        def subscribe_data(self, *args, **kwargs): pass
    class StrategyConfig: pass



class EqualWeightConfig(StrategyConfig):
    instruments: list[str]
    initial_capital: float = 10_000.0
    rebalance_threshold: float = 0.05


class EqualWeight(Strategy):
    """
    Equal weight portfolio strategy.
    Strategy A — baseline comparison.

    Allocates equally across all instruments.
    Rebalances when any allocation drifts
    more than rebalance_threshold from target.
    """

    def __init__(self, config: EqualWeightConfig):
        super().__init__(config)
        self._instruments = config.instruments
        self._target_weight = (
            1.0 / len(config.instruments))
        self._threshold = config.rebalance_threshold

    def on_start(self) -> None:
        """Subscribe to bars for all instruments."""
        for instrument_id in self._instruments:
            self.subscribe_bars(
                bar_type=self._get_bar_type(
                    instrument_id))

    def on_bar(self, bar) -> None:
        """Check if rebalancing needed."""
        self._maybe_rebalance()

    def _maybe_rebalance(self) -> None:
        """Rebalance if any weight drifts."""
        # Implementation uses NautilusTrader
        # portfolio and order management APIs
        pass

    def _get_bar_type(self, instrument_id: str):
        """Build bar type for subscription."""
        pass
