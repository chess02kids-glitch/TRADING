"""
Strategy B: HAR Volatility Targeting

Portfolio strategy using HAR predicted range
to size positions.

Core formula:
  allocation = target_vol / vol_estimate
  vol_estimate = HAR_predicted_range / price

High HAR volatility → smaller allocation
Low HAR volatility → larger allocation

This is Kelly-adjacent volatility targeting.
Theoretical basis: strong (risk parity,
institutional risk management).

HAR predicts MAGNITUDE only.
This strategy does NOT need direction.
This is the most honest use of HAR.

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



class HARVolTargetingConfig(StrategyConfig):
    instruments: list[str]
    target_vol: float = 0.02
    min_allocation: float = 0.05
    max_allocation: float = 1.00
    rebalance_threshold: float = 0.05
    initial_capital: float = 10_000.0


class HARVolTargeting(Strategy):
    """
    Volatility targeting strategy using
    HAR predicted ranges.
    Strategy B — experimental variant.

    Subscribes to HARVolatilitySignal
    published by HARVolatilityActor.

    When signal arrives:
      1. Compute target allocation
      2. Compare to current allocation
      3. Rebalance if drift > threshold
    """

    def __init__(
        self, config: HARVolTargetingConfig):
        super().__init__(config)
        self._target_vol = config.target_vol
        self._min_alloc = config.min_allocation
        self._max_alloc = config.max_allocation
        self._threshold = (
            config.rebalance_threshold)
        self._current_allocations = {}
        self._last_signals = {}

    def on_start(self) -> None:
        """Subscribe to HAR signals."""
        from nautilus_har.actors.har_volatility_actor \
            import HARVolatilitySignal
        self.subscribe_data(HARVolatilitySignal)

    def on_data(self, data) -> None:
        """
        Process HAR volatility signal.
        Compute target allocation and rebalance.
        """
        from nautilus_har.actors.har_volatility_actor \
            import HARVolatilitySignal
        if not isinstance(
            data, HARVolatilitySignal):
            return

        # Store latest signal
        self._last_signals[
            data.instrument_id] = data

        # Compute target allocation
        target = self._compute_allocation(
            data.volatility_estimate)

        current = self._current_allocations.get(
            data.instrument_id, 0.5)

        # Rebalance if drift exceeds threshold
        if abs(target - current) > self._threshold:
            self._rebalance(
                data.instrument_id,
                target,
                data.current_price)

    def _compute_allocation(
        self,
        vol_estimate: float,
    ) -> float:
        """
        Compute target allocation.
        allocation = target_vol / vol_estimate
        Clipped to [min_alloc, max_alloc]
        """
        if vol_estimate <= 0:
            return 0.5  # default if no signal

        allocation = (
            self._target_vol / vol_estimate)
        return max(
            self._min_alloc,
            min(self._max_alloc, allocation))

    def _rebalance(
        self,
        instrument_id: str,
        target_alloc: float,
        price: float,
    ) -> None:
        """Execute rebalancing orders."""
        # NautilusTrader order execution
        # Updates self._current_allocations
        pass
