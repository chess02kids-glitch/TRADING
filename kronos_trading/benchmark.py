"""Real full-model inference benchmark for Kronos.

Separates one-time costs (model loading, first-inference warmup) from steady
state latency, synchronises CUDA around every timed region, and reports real
measurements only - nothing is fabricated.
"""
import statistics
import time
from typing import Dict, List, Optional

from .model import ModelManager, KronosRealPredictor
from .pipeline import PredictionPipeline
from .preprocess import closed, validate_context
from .types import Candle


def _cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _reset_peak_vram():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _peak_vram() -> Optional[int]:
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except ImportError:
        pass
    return None


def _cpu_rss_bytes() -> Optional[int]:
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    except Exception:
        try:
            import psutil
            return int(psutil.Process().memory_info().rss)
        except Exception:
            return None


def measure_model_load(manager: ModelManager) -> Dict:
    _cuda_sync()
    start = time.perf_counter()
    manager.load()
    _cuda_sync()
    load_s = time.perf_counter() - start
    return {
        "load_seconds": load_s,
        "available": manager.available,
        "error": manager.error,
        "report": manager.report(),
    }


def run_benchmark(manager: ModelManager,
                  candles: List[Candle],
                  timeframe: str = "1h",
                  context_length: int = 512,
                  horizon: int = 1,
                  now_ms: Optional[int] = None,
                  warmed_runs: int = 10,
                  seed: int = 123,
                  deterministic: bool = True) -> Dict:
    """Benchmark real inference end-to-end.

    Returns a dictionary of measurements. ``manager`` should already be loaded
    (load time is measured separately via ``measure_model_load``).
    """
    import time as _time
    if not manager.available:
        return {
            "status": "unavailable",
            "error": manager.error,
            "report": manager.report(),
        }
    now_ms = now_ms or int(_time.time() * 1000)
    ctx = validate_context(closed(candles, timeframe, now_ms), timeframe,
                           context_length)

    predictor = KronosRealPredictor(manager)
    pipeline = PredictionPipeline(predictor)

    # First (cold) inference - includes any lazy cuDNN/autotuning initialisation.
    _reset_peak_vram()
    _cuda_sync()
    start = time.perf_counter()
    first_pred = pipeline.predict("BENCH", timeframe, ctx, context_length,
                                  horizon, now_ms, seed=seed,
                                  deterministic=deterministic)
    _cuda_sync()
    first_ms = (time.perf_counter() - start) * 1000.0

    # Warmed steady-state latency.
    latencies: List[float] = []
    for _ in range(warmed_runs):
        _cuda_sync()
        start = time.perf_counter()
        pipeline.predict("BENCH", timeframe, ctx, context_length, horizon,
                         now_ms, seed=seed, deterministic=deterministic)
        _cuda_sync()
        latencies.append((time.perf_counter() - start) * 1000.0)

    peak = _peak_vram()

    return {
        "status": "ok",
        "symbol_context": "synthetic/benchmark context",
        "context_candles": len(ctx),
        "horizon": horizon,
        "first_inference_ms": first_ms,
        "warmed_runs": warmed_runs,
        "warmed_latency_ms": {
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
        },
        "peak_vram_bytes": peak,
        "cpu_rss_bytes": _cpu_rss_bytes(),
        "first_prediction": first_pred.asdict(),
        "report": manager.report(),
    }
