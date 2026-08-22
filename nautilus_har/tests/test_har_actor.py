import pytest
import pandas as pd
from datetime import datetime, timezone
from nautilus_har.actors.har_volatility_actor import (
    HARVolatilityActor,
    HARVolatilitySignal,
)

def test_signal_dataclass_fields():
    signal = HARVolatilitySignal(
        instrument_id="BTC/USDT",
        predicted_range=100.0,
        current_price=50000.0,
        volatility_estimate=0.002,
        regime="low",
        ts_event=123456,
        ts_init=123456
    )
    assert signal.instrument_id == "BTC/USDT"
    assert signal.predicted_range == 100.0
    assert signal.current_price == 50000.0
    assert signal.volatility_estimate == 0.002
    assert signal.regime == "low"
    assert signal.ts_event == 123456
    assert signal.ts_init == 123456

def test_compute_allocation_normal():
    # vol=0.04, target=0.02 → allocation=0.5
    actor = HARVolatilityActor(predictions={}, target_vol=0.02)
    # The actor doesn't compute allocation (strategy does),
    # but the prompt says this test is in test_har_actor.py.
    # I'll implement it here by invoking the strategy.
    from nautilus_har.strategies.har_vol_targeting import HARVolTargeting, HARVolTargetingConfig
    config = HARVolTargetingConfig(instruments=["BTC/USDT"], target_vol=0.02, min_allocation=0.05, max_allocation=1.0)
    strategy = HARVolTargeting(config)
    alloc = strategy._compute_allocation(0.04)
    assert alloc == 0.5

def test_compute_allocation_low_vol():
    # vol=0.005, target=0.02 → clipped at max=1.0
    from nautilus_har.strategies.har_vol_targeting import HARVolTargeting, HARVolTargetingConfig
    config = HARVolTargetingConfig(instruments=["BTC/USDT"], target_vol=0.02, min_allocation=0.05, max_allocation=1.0)
    strategy = HARVolTargeting(config)
    alloc = strategy._compute_allocation(0.005)
    assert alloc == 1.0

def test_compute_allocation_high_vol():
    # vol=0.10, target=0.02 → allocation=0.20
    from nautilus_har.strategies.har_vol_targeting import HARVolTargeting, HARVolTargetingConfig
    config = HARVolTargetingConfig(instruments=["BTC/USDT"], target_vol=0.02, min_allocation=0.05, max_allocation=1.0)
    strategy = HARVolTargeting(config)
    alloc = strategy._compute_allocation(0.10)
    assert alloc == pytest.approx(0.20)

def test_compute_inverse_allocation():
    # vol=0.04, target=0.02 → allocation=2.0 (clipped)
    from nautilus_har.strategies.har_vol_inverse import HARVolInverse, HARVolInverseConfig
    config = HARVolInverseConfig(instruments=["BTC/USDT"], target_vol=0.02, min_allocation=0.05, max_allocation=1.0)
    strategy = HARVolInverse(config)
    alloc = strategy._compute_inverse_allocation(0.04)
    assert alloc == 1.0

def test_classify_regime_low():
    actor = HARVolatilityActor(predictions={}, target_vol=0.02)
    assert actor._classify_regime(0.005) == "low"

def test_classify_regime_medium():
    actor = HARVolatilityActor(predictions={}, target_vol=0.02)
    assert actor._classify_regime(0.02) == "medium"

def test_classify_regime_high():
    actor = HARVolatilityActor(predictions={}, target_vol=0.02)
    assert actor._classify_regime(0.05) == "high"
