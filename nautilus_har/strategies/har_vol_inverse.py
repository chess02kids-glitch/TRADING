"""
Strategy C: Inverse HAR Volatility (Control)

OPPOSITE of Strategy B.
More allocation when volatility is HIGH.
Less allocation when volatility is LOW.

allocation = vol_estimate / target_vol
(inverse of Strategy B formula)

Research purpose:
If B beats A AND C loses to A:
  → Volatility targeting direction is correct
  → HAR is being used correctly

If C beats A:
  → Our hypothesis is wrong
  → More exposure in high vol is better
    (momentum effect?)

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



class HARVolInverseConfig(StrategyConfig):
    instruments: list[str]
    target_vol: float = 0.02
    min_allocation: float = 0.05
    max_allocation: float = 1.00
    rebalance_threshold: float = 0.05
    initial_capital: float = 10_000.0


class HARVolInverse(Strategy):
    """
    Inverse volatility targeting.
    Strategy C — control experiment.

    Increases allocation during high vol.
    Decreases during low vol.
    OPPOSITE of Strategy B.
    """

    def __init__(
        self, config: HARVolInverseConfig):
        super().__init__(config)
        self._target_vol = config.target_vol
        self._min_alloc = config.min_allocation
        self._max_alloc = config.max_allocation
        self._threshold = (
            config.rebalance_threshold)

    def on_start(self) -> None:
        from nautilus_har.actors.har_volatility_actor \
            import HARVolatilitySignal
        self.subscribe_data(HARVolatilitySignal)

    def on_data(self, data) -> None:
        from nautilus_har.actors.har_volatility_actor \
            import HARVolatilitySignal
        if not isinstance(
            data, HARVolatilitySignal):
            return

        # INVERSE formula
        target = self._compute_inverse_allocation(
            data.volatility_estimate)

        self._rebalance_if_needed(
            data.instrument_id,
            target,
            data.current_price)

    def _compute_inverse_allocation(
        self,
        vol_estimate: float,
    ) -> float:
        """
        INVERSE: more in high vol
        allocation = vol_estimate / target_vol
        Clipped to [min, max]
        """
        if vol_estimate <= 0:
            return 0.5

        allocation = (
            vol_estimate / self._target_vol)
        return max(
            self._min_alloc,
            min(self._max_alloc, allocation))

    def _rebalance_if_needed(
        self, instrument_id, target, price):
        pass
