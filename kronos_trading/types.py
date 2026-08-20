"""Shared data types for the paper-only Kronos trading research system."""
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List


@dataclass(frozen=True)
class Candle:
    """One OHLCV candle (UTC open time in milliseconds)."""
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Prediction:
    """Structured prediction result.

    The first block of fields is the original Phase 1/2 contract and is kept
    stable for backwards compatibility with the signal engine and backtester.

    The optional fields added for Phase 3 describe what the real Kronos model
    actually produced (Kronos emits open/high/low/close/volume/amount per
    horizon step - nothing else is invented here).
    """
    # --- stable contract ---
    symbol: str
    timeframe: str
    prediction_timestamp_ms: int
    horizon: int
    predicted_close: float
    expected_return: float
    input_start_ms: int
    input_end_ms: int
    generated_at_ms: int
    model_version: str
    device: str
    inference_ms: float
    confidence: Optional[float] = None

    # --- Phase 3 structured fields (all optional for mock/backward compat) ---
    context_length: Optional[int] = None
    prediction_timestamps_ms: Optional[List[int]] = None
    predicted_open: Optional[float] = None
    predicted_high: Optional[float] = None
    predicted_low: Optional[float] = None
    predicted_volume: Optional[float] = None
    predicted_amount: Optional[float] = None
    predicted_ohlcv: Optional[List[Dict[str, float]]] = None
    model_name: Optional[str] = None
    model_revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    dtype: Optional[str] = None
    peak_vram_bytes: Optional[int] = None

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Signal:
    symbol: str
    timestamp_ms: int
    side: str
    expected_return: float
    reason: str
