"""Phase 3 - Real Kronos inference: validation, gating and correctness.

These tests do NOT require the model weights:

* data correctness (closed-candle enforcement, gap / NaN / inf / OHLC / dup /
  unsupported-timeframe / insufficient-context handling);
* mock-vs-real gating (the real path never silently falls back to mock);
* device resolution and CPU fallback;
* seed/determinism helpers.

Real-weight tests (deterministic inference, structured output) are present but
``pytest.skip`` when the weights are not available in the environment, so the
suite stays green offline and runs fully on a machine with the model present.
"""
import math

import numpy as np
import pytest

from kronos_trading.model import (
    ModelManager,
    ModelUnavailableError,
    KronosRealPredictor,
    DeterministicMockPredictor,
    resolve_device,
    set_seed,
)
from kronos_trading.pipeline import PredictionPipeline
from kronos_trading.preprocess import (
    closed,
    validate_context,
    to_kronos_frame,
    future_timestamps,
    TF,
)
from kronos_trading.types import Candle

H = TF['1h']
BASE = 1_700_000_000_000


def mk(n, base=BASE, step=H):
    """Deterministically increasing, OHLC-valid candles."""
    return [Candle(base + i * step, 100.0 + i, 101.0 + i, 99.0 + i,
                   100.5 + i, 10.0) for i in range(n)]


# --------------------------------------------------------------------------- #
# 10. Look-ahead / closed-candle enforcement
# --------------------------------------------------------------------------- #
def test_forming_candle_is_excluded_and_context_ends_at_latest_closed():
    candles = mk(10)
    forming = Candle(candles[-1].timestamp_ms + H, 110, 111, 109, 110.5, 10)
    # "now" is halfway through the forming candle, so it is not closed yet.
    now_ms = forming.timestamp_ms + H // 2

    closed_only = closed(candles + [forming], '1h', now_ms)
    assert len(closed_only) == 10
    assert closed_only[-1].timestamp_ms == candles[-1].timestamp_ms

    ctx = validate_context(closed_only, '1h', 5)
    assert ctx[-1].timestamp_ms == candles[-1].timestamp_ms

    pipe = PredictionPipeline(DeterministicMockPredictor())
    pred = pipe.predict('BTC/USDT', '1h', candles + [forming], 5, 1, now_ms)
    assert pred.input_end_ms == candles[-1].timestamp_ms
    assert pred.prediction_timestamp_ms == forming.timestamp_ms


def test_pipeline_context_never_includes_forming_candle_end_to_end():
    candles = mk(64)
    forming = Candle(candles[-1].timestamp_ms + H, 200, 201, 199, 200.5, 1)
    now_ms = forming.timestamp_ms + H - 1  # 1 ms before close: still forming
    pipe = PredictionPipeline(DeterministicMockPredictor())
    pred = pipe.predict('ETH/USDT', '1h', candles + [forming], 64, 1, now_ms)
    assert pred.input_end_ms == candles[-1].timestamp_ms
    assert pred.input_end_ms < pred.prediction_timestamp_ms


# --------------------------------------------------------------------------- #
# 11. Error handling - never silently substitute mock data
# --------------------------------------------------------------------------- #
def test_validate_context_gap_raises():
    a = mk(5)
    a[3] = Candle(9 * H, 1, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        validate_context(a, '1h', 5)


def test_validate_context_duplicate_timestamps_raise():
    a = mk(5)
    a[4] = Candle(a[3].timestamp_ms, 1, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        validate_context(a, '1h', 5)


def test_validate_context_nan_raises():
    a = mk(5)
    a[2] = Candle(a[2].timestamp_ms, float('nan'), 1, 1, 1, 1)
    with pytest.raises(ValueError):
        validate_context(a, '1h', 5)


def test_validate_context_inf_raises():
    a = mk(5)
    a[2] = Candle(a[2].timestamp_ms, 1, float('inf'), 1, 1, 1)
    with pytest.raises(ValueError):
        validate_context(a, '1h', 5)


def test_validate_context_invalid_ohlc_raises():
    a = mk(5)
    a[2] = Candle(a[2].timestamp_ms, 100, 90, 99, 100, 1)  # high < low
    with pytest.raises(ValueError):
        validate_context(a, '1h', 5)


def test_validate_context_negative_volume_raises():
    a = mk(5)
    a[2] = Candle(a[2].timestamp_ms, 100, 101, 99, 100, -1)
    with pytest.raises(ValueError):
        validate_context(a, '1h', 5)


def test_validate_context_unsupported_timeframe_raises():
    with pytest.raises(ValueError):
        validate_context(mk(2), '5m', 1)


def test_validate_context_insufficient_context_raises():
    with pytest.raises(ValueError):
        validate_context(mk(3), '1h', 10)


def test_real_model_load_fails_clearly_without_weights():
    """Missing weights must produce an explicit error, never a fake success."""
    manager = ModelManager(local_files_only=True).load()
    assert not manager.available
    assert manager.error
    assert manager.report()['available'] is False


def test_real_predictor_requires_available_manager():
    manager = ModelManager(local_files_only=True)  # not loaded
    with pytest.raises(ModelUnavailableError):
        KronosRealPredictor(manager)


def test_cli_real_path_never_falls_back_to_mock(monkeypatch):
    import kronos_trading.cli as cli
    monkeypatch.setattr(cli.ModelManager, 'load', lambda self: self)

    class Args:
        mock = False
        model = 'NeoQuasar/Kronos-small'
        tokenizer = 'NeoQuasar/Kronos-Tokenizer-base'
        model_revision = None
        tokenizer_revision = None
        device = None
        max_context = 512
        cache_dir = None

    with pytest.raises(ModelUnavailableError):
        cli._build_pipeline(Args())


# --------------------------------------------------------------------------- #
# 9. Device resolution & CPU fallback
# --------------------------------------------------------------------------- #
def test_resolve_device_cpu_fallback_when_cuda_unavailable(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert resolve_device(None) == 'cpu'
    assert resolve_device('cpu') == 'cpu'


def test_resolve_device_explicit_cuda_raises_when_unavailable(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ModelUnavailableError):
        resolve_device('cuda:0')


def test_resolve_device_invalid_device_raises():
    with pytest.raises(ValueError):
        resolve_device('tpu')


# --------------------------------------------------------------------------- #
# 8. Determinism / repeatability
# --------------------------------------------------------------------------- #
def test_set_seed_reproducible_numpy():
    set_seed(7)
    a = np.random.rand(4)
    set_seed(7)
    b = np.random.rand(4)
    assert np.array_equal(a, b)


def test_mock_predictor_is_deterministic():
    pipe = PredictionPipeline(DeterministicMockPredictor())
    candles = mk(100)
    now = candles[-1].timestamp_ms + H
    p1 = pipe.predict('BTC/USDT', '1h', candles, 64, 1, now)
    p2 = pipe.predict('BTC/USDT', '1h', candles, 64, 1, now)
    assert p1.predicted_close == p2.predicted_close
    assert p1.expected_return == p2.expected_return


# --------------------------------------------------------------------------- #
# 4 / 6. Preprocessing contract & structured output
# --------------------------------------------------------------------------- #
def test_to_kronos_frame_columns_and_length():
    candles = mk(8)
    df, xt = to_kronos_frame(candles)
    assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert len(df) == len(xt) == len(candles)


def test_future_timestamps_spacing():
    candles = mk(4)
    ts = future_timestamps(candles[-1].timestamp_ms, '1h', 3)
    assert len(ts) == 3
    assert int((ts.iloc[1] - ts.iloc[0]).total_seconds()) == 3600
    assert int((ts.iloc[2] - ts.iloc[0]).total_seconds()) == 7200


def test_prediction_has_required_structured_keys():
    pipe = PredictionPipeline(DeterministicMockPredictor())
    candles = mk(100)
    now = candles[-1].timestamp_ms + H
    d = pipe.predict('BTC/USDT', '1h', candles, 64, 1, now).asdict()
    for k in ('symbol', 'timeframe', 'prediction_timestamp_ms', 'horizon',
              'predicted_close', 'expected_return', 'input_start_ms',
              'input_end_ms', 'generated_at_ms', 'model_version', 'device',
              'inference_ms'):
        assert k in d, k


try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False

@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_real_adapter_wiring_with_stub_upstream():
    """Exercise the real-predictor adapter against the upstream call contract.

    Uses a stub for the upstream ``KronosPredictor.predict`` so the adapter's
    frame construction, timestamp handling, result parsing and latency capture
    are verified without requiring the model weights. This is adapter coverage
    only - it is never presented as real model output.
    """
    import pandas as pd
    manager = ModelManager()  # not loaded yet

    class StubUpstream:
        def __init__(self):
            self.calls = []

        def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0,
                    top_k=0, top_p=0.9, sample_count=1, verbose=True):
            self.calls.append((df, x_timestamp, y_timestamp, pred_len, T,
                               top_k, top_p, sample_count))
            return pd.DataFrame({
                'open': [1.0] * pred_len, 'high': [2.0] * pred_len,
                'low': [0.5] * pred_len, 'close': [1.5] * pred_len,
                'volume': [10.0] * pred_len, 'amount': [15.0] * pred_len,
            }, index=y_timestamp)

    manager.predictor = StubUpstream()
    manager.error = None
    manager.device = 'cpu'
    manager.dtype = 'torch.float32'
    manager.model_name = 'NeoQuasar/Kronos-small'
    manager.resolved_model_revision = 'rev'
    manager.resolved_tokenizer_revision = 'rev'
    manager.max_context = 512

    predictor = KronosRealPredictor(manager)
    candles = mk(8)
    result = predictor.predict(candles, '1h', horizon=2, deterministic=True)

    df, xt, yt, plen, T, top_k, top_p, sample_count = manager.predictor.calls[0]
    assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert len(df) == len(xt) == 8
    assert len(yt) == 2 == plen
    # deterministic mode forces the upstream argmax recipe
    assert (top_k, top_p, sample_count) == (1, 1.0, 1)

    assert len(result.steps) == 2
    assert result.steps[0]['close'] == 1.5
    assert set(result.steps[0].keys()) == \
        {'open', 'high', 'low', 'close', 'volume', 'amount'}
    assert result.peak_vram_bytes is None  # CPU path
    assert result.latency_ms >= 0


# --------------------------------------------------------------------------- #
# Real-weight tests - run only when the real Kronos model is present.
# --------------------------------------------------------------------------- #
def _loaded_manager():
    manager = ModelManager(local_files_only=True).load()
    return manager if manager.available else None


def test_real_model_deterministic_inference_and_structured_output():
    pytest.importorskip('torch')
    manager = _loaded_manager()
    if manager is None:
        pytest.skip('real Kronos weights not available in this environment')

    candles = mk(600)  # more than max_context (512) to exercise the window
    now = candles[-1].timestamp_ms + H
    pipe = PredictionPipeline(KronosRealPredictor(manager))

    p1 = pipe.predict('BTC/USDT', '1h', candles, 512, 1, now,
                      seed=0, deterministic=True)
    p2 = pipe.predict('BTC/USDT', '1h', candles, 512, 1, now,
                      seed=0, deterministic=True)

    assert p1.predicted_close == p2.predicted_close
    assert p1.predicted_ohlcv is not None and len(p1.predicted_ohlcv) == 1
    assert set(p1.predicted_ohlcv[0].keys()) == \
        {'open', 'high', 'low', 'close', 'volume', 'amount'}
    assert p1.model_name == 'NeoQuasar/Kronos-small'
    assert p1.dtype is not None
    assert p1.device in ('cpu', 'cuda:0')
    assert p1.context_length == 512
    assert p1.prediction_timestamps_ms == [now]
    assert all(math.isfinite(v) for v in p1.predicted_ohlcv[0].values())


def test_real_model_reports_device_params_dtype():
    pytest.importorskip('torch')
    manager = _loaded_manager()
    if manager is None:
        pytest.skip('real Kronos weights not available in this environment')
    report = manager.report()
    assert report['available'] is True
    assert report['total_params'] and report['total_params'] > 0
    assert report['dtype'] in ('torch.float32', 'torch.float16', 'torch.bfloat16')
    assert report['device'] in ('cpu', 'cuda:0')
