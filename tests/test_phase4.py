"""Phase 4 - chronological no-lookahead evaluation tests.

The evaluator is exercised with a deterministic *fake* predictor that
implements the exact KronosRealPredictor interface (``predict(candles,
timeframe, horizon, ...) -> PredictorResult``). This is a test double used to
prove chronology/no-lookahead/metric correctness; it is never presented as
real Kronos output.

Real-weight tests are present but ``pytest.skip`` when the weights are not
available, so the suite stays green offline and runs fully on a machine with
the model present.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from kronos_trading.evaluation import (
    EvaluationConfig,
    EvaluationRow,
    PredictionEvaluator,
    compute_metrics,
    parse_timestamp,
)
from kronos_trading.model import ModelManager, KronosRealPredictor, PredictorResult
from kronos_trading.types import Candle

H = 3_600_000
BASE = 1_700_000_000_000


def mk(n, base=BASE, step=H):
    """Deterministically increasing, OHLC-valid candles."""
    return [Candle(base + i * step, 100.0 + i * 0.1, 101.0 + i * 0.1,
                   99.0 + i * 0.1, 100.05 + i * 0.1, 10.0) for i in range(n)]


class FakePredictor:
    """Deterministic test double with the real predictor's interface."""

    def __init__(self, close_factor=1.01):
        self.calls = []
        self.device = 'fake'
        self.dtype = 'torch.float32'
        self.manager = SimpleNamespace(
            model_name='FakeKronos-small',
            resolved_model_revision='fake-model-rev',
            resolved_tokenizer_revision='fake-tokenizer-rev',
            max_context=512,
        )
        self.close_factor = close_factor

    def predict(self, candles, timeframe, horizon=1, temperature=1.0,
                top_k=0, top_p=0.9, sample_count=1, seed=None,
                deterministic=False):
        self.calls.append({
            'timestamps': [c.timestamp_ms for c in candles],
            'horizon': horizon, 'seed': seed, 'deterministic': deterministic,
            'top_k': top_k, 'top_p': top_p,
        })
        last = candles[-1].close
        steps = []
        for k in range(horizon):
            close = last * (self.close_factor ** (k + 1))
            steps.append({'open': close * 0.995, 'high': close * 1.005,
                          'low': close * 0.99, 'close': close,
                          'volume': 1.0, 'amount': close})
        return PredictorResult(steps=steps, latency_ms=0.5, peak_vram_bytes=None)


def cfg(**overrides):
    defaults = dict(context_length=10, horizon=1, max_predictions=1000)
    defaults.update(overrides)
    return EvaluationConfig(**defaults)


# --------------------------------------------------------------------------- #
# 8.A - future candle never part of model input
# --------------------------------------------------------------------------- #
def test_future_candle_never_part_of_model_input():
    candles = mk(40)
    pred = FakePredictor()
    res = PredictionEvaluator(pred, cfg(), 'BTC/USDT', '1h').evaluate(candles)
    assert len(res.rows) == len(pred.calls) > 0
    for call, row in zip(pred.calls, res.rows):
        ts = call['timestamps']
        assert row.prediction_timestamp not in ts
        assert max(ts) < row.prediction_timestamp
        assert max(ts) == row.context_end_timestamp
        assert len(ts) == 10


# --------------------------------------------------------------------------- #
# 8.B - removing future candles leaves the context identical
# --------------------------------------------------------------------------- #
def test_removing_future_candles_keeps_context_identical():
    candles = mk(50)
    p1 = FakePredictor()
    r1 = PredictionEvaluator(p1, cfg(), 'BTC/USDT', '1h').evaluate(candles)
    p2 = FakePredictor()
    r2 = PredictionEvaluator(p2, cfg(), 'BTC/USDT', '1h').evaluate(candles[:30])
    assert 0 < len(r2.rows) <= len(r1.rows)
    assert p2.calls == p1.calls[:len(p2.calls)]
    assert [x.asdict() for x in r2.rows] == \
        [x.asdict() for x in r1.rows[:len(r2.rows)]]


# --------------------------------------------------------------------------- #
# 8.C / 8.F - strictly forward through time
# --------------------------------------------------------------------------- #
def test_evaluator_moves_strictly_forward():
    res = PredictionEvaluator(FakePredictor(), cfg(), 'BTC/USDT', '1h').evaluate(mk(40))
    pred_ts = [r.prediction_timestamp for r in res.rows]
    ctx_ts = [r.context_end_timestamp for r in res.rows]
    act_ts = [r.actual_timestamp for r in res.rows]
    assert pred_ts == sorted(pred_ts) and len(set(pred_ts)) == len(pred_ts)
    assert ctx_ts == sorted(ctx_ts) and len(set(ctx_ts)) == len(ctx_ts)
    assert act_ts == sorted(act_ts) and len(set(act_ts)) == len(act_ts)


# --------------------------------------------------------------------------- #
# 8.D - gap causes a skip, never fabricated data
# --------------------------------------------------------------------------- #
def test_gap_causes_skip_not_fabrication():
    full = mk(22)
    candles = full[:15] + full[16:]  # one missing candle at original index 15
    pred = FakePredictor()
    res = PredictionEvaluator(pred, cfg(context_length=5), 'BTC/USDT', '1h').evaluate(candles)
    assert res.skipped == 5
    assert res.skip_reasons.get('target_gap') == 1
    assert res.skip_reasons.get('context_invalid') == 4
    assert len(res.rows) == 10
    # every model input is contiguous - no fabricated candles
    for call in pred.calls:
        ts = call['timestamps']
        assert all(b - a == H for a, b in zip(ts, ts[1:]))
    # every produced row's target follows its context by exactly one step
    for row in res.rows:
        assert row.prediction_timestamp - row.context_end_timestamp == H


# --------------------------------------------------------------------------- #
# 8.E - the currently-forming candle is excluded
# --------------------------------------------------------------------------- #
def test_forming_candle_excluded():
    candles = mk(30)
    pred = FakePredictor()
    res = PredictionEvaluator(pred, cfg(), 'BTC/USDT', '1h').evaluate(candles)
    last_ts = candles[-1].timestamp_ms
    assert all(r.actual_timestamp < last_ts for r in res.rows)
    assert max(r.actual_timestamp for r in res.rows) == candles[-2].timestamp_ms
    assert all(last_ts not in call['timestamps'] for call in pred.calls)


# --------------------------------------------------------------------------- #
# 8.G - the actual target candle is accessed only after inference
# --------------------------------------------------------------------------- #
def test_target_candle_read_only_after_inference():
    pred = FakePredictor()
    res = PredictionEvaluator(pred, cfg(horizon=2), 'BTC/USDT', '1h').evaluate(mk(40))
    assert len(res.rows) == len(pred.calls) > 0
    for call, row in zip(pred.calls, res.rows):
        ts = call['timestamps']
        # neither the first predicted candle nor the final target was in input
        assert row.prediction_timestamp not in ts
        assert row.actual_timestamp not in ts
        assert max(ts) < row.prediction_timestamp
        # horizon=2: the final scored candle is one step after the first predicted
        assert row.actual_timestamp == row.prediction_timestamp + H


# --------------------------------------------------------------------------- #
# 9. Determinism / repeatability
# --------------------------------------------------------------------------- #
def test_deterministic_repeatability():
    candles = mk(40)
    r1 = PredictionEvaluator(FakePredictor(), cfg(), 'BTC/USDT', '1h').evaluate(candles)
    r2 = PredictionEvaluator(FakePredictor(), cfg(), 'BTC/USDT', '1h').evaluate(candles)
    assert [x.asdict() for x in r1.rows] == [x.asdict() for x in r2.rows]
    assert r1.report['metrics'] == r2.report['metrics']
    assert r1.report['deterministic'] is True
    assert r1.report['seed'] == 0
    assert r1.report['top_k'] == 1 and r1.report['top_p'] == 1.0


# --------------------------------------------------------------------------- #
# 6. Metrics - exact values
# --------------------------------------------------------------------------- #
def _row(pc, ac, pr, ar):
    return EvaluationRow(
        symbol='X', timeframe='1h', context_end_timestamp=0,
        prediction_timestamp=H, actual_timestamp=H, context_length=5,
        predicted_open=pc, predicted_high=pc, predicted_low=pc,
        predicted_close=pc, predicted_volume=1.0,
        actual_open=ac, actual_high=ac, actual_low=ac, actual_close=ac,
        actual_volume=1.0, predicted_return=pr, actual_return=ar,
        absolute_close_error=abs(pc - ac), squared_close_error=(pc - ac) ** 2,
        directional_correct=False, model_revision=None, tokenizer_revision=None,
        inference_latency_ms=0.0, device='fake', deterministic=True, seed=0,
        top_k=1, top_p=1.0, sample_count=1, horizon=1, direction_threshold=0.005,
    )


def test_compute_metrics_exact():
    rows = [
        _row(110, 105, 0.10, 0.05),
        _row(100, 102, 0.00, 0.02),
        _row(90, 95, -0.10, -0.05),
        _row(100, 100, 0.00, 0.00),
    ]
    m = compute_metrics(rows, 0.005)
    assert m['predictions'] == 4
    assert m['mae_close'] == pytest.approx(3.0)
    assert m['rmse_close'] == pytest.approx(math.sqrt(13.5))
    assert m['directional_accuracy'] == pytest.approx(2 / 3)
    assert m['bullish_accuracy'] == pytest.approx(0.5)
    assert m['bearish_accuracy'] == pytest.approx(1.0)
    assert m['n_positive_actual'] == 2
    assert m['n_negative_actual'] == 1
    assert m['n_near_zero_actual'] == 1
    assert m['n_directional_correct'] == 2
    assert m['n_directional_incorrect'] == 1
    assert m['mean_actual_return'] == pytest.approx(0.005)
    assert m['mean_predicted_return'] == pytest.approx(0.0)
    assert m['return_mae'] == pytest.approx(0.03)
    expected_corr = np.corrcoef([0.10, 0.0, -0.10, 0.0], [0.05, 0.02, -0.05, 0.0])[0, 1]
    assert m['return_correlation'] == pytest.approx(expected_corr)
    assert m['mape_close_valid_count'] == 4
    assert m['mape_close'] == pytest.approx(
        (5 / 105 + 2 / 102 + 5 / 95 + 0.0) / 4)


def test_metrics_safe_on_empty_and_zero_variance():
    empty = compute_metrics([], 0.0005)
    assert empty['predictions'] == 0
    assert empty['mae_close'] is None
    assert empty['directional_accuracy'] is None
    assert empty['return_correlation'] is None

    rows = [_row(100, 100, 0.01, 0.01), _row(100, 100, 0.01, 0.01)]
    m = compute_metrics(rows, 0.0005)
    assert m['rmse_close'] == 0.0
    assert m['return_rmse'] == 0.0
    assert m['return_correlation'] is None  # zero variance -> undefined
    assert m['directional_accuracy'] == 1.0  # both up / both up


# --------------------------------------------------------------------------- #
# 4. Window selection + documented timestamps
# --------------------------------------------------------------------------- #
def test_explicit_window_and_documented_timestamps():
    candles = mk(30)
    c = cfg(start_ms=candles[10].timestamp_ms,
            end_ms=candles[20].timestamp_ms + H)
    res = PredictionEvaluator(FakePredictor(), c, 'BTC/USDT', '1h').evaluate(candles)
    assert res.report['evaluation_start_ms'] == candles[10].timestamp_ms
    assert res.report['evaluation_end_ms'] == candles[20].timestamp_ms + H
    assert res.report['warmup_end_ms'] == candles[9].timestamp_ms
    assert len(res.rows) == 11
    assert all(r.prediction_timestamp >= candles[10].timestamp_ms for r in res.rows)
    assert all(r.prediction_timestamp + H <= candles[20].timestamp_ms + H for r in res.rows)


def test_end_only_window_is_explicit_and_uncapped():
    candles = mk(40)
    c = cfg(end_ms=candles[20].timestamp_ms + H, max_predictions=3)
    res = PredictionEvaluator(FakePredictor(), c, 'BTC/USDT', '1h').evaluate(candles)
    # explicit window: warm-up (10) through target close <= end -> 11 rows, uncapped
    assert len(res.rows) == 11


def test_default_window_is_capped_by_max_predictions():
    candles = mk(40)
    res = PredictionEvaluator(FakePredictor(), cfg(max_predictions=5),
                              'BTC/USDT', '1h').evaluate(candles)
    assert len(res.rows) == 5
    # recent window: last target = last closed candle
    assert max(r.actual_timestamp for r in res.rows) == candles[-2].timestamp_ms


# --------------------------------------------------------------------------- #
# 11. skip reasons for invalid targets
# --------------------------------------------------------------------------- #
def test_invalid_target_is_skipped_with_reason():
    candles = mk(20)
    bad = list(candles)
    bad[-2] = Candle(bad[-2].timestamp_ms, bad[-2].open, bad[-2].high,
                     bad[-2].low, float('nan'), bad[-2].volume)
    res = PredictionEvaluator(FakePredictor(), cfg(), 'BTC/USDT', '1h').evaluate(bad)
    assert res.skip_reasons.get('invalid_target') == 1
    assert res.skipped == 1


def test_unsorted_input_is_sorted_defensively():
    shuffled = mk(30)[::-1]
    pred = FakePredictor()
    res = PredictionEvaluator(pred, cfg(), 'BTC/USDT', '1h').evaluate(shuffled)
    assert res.report['predictions'] == 19
    for call in pred.calls:
        ts = call['timestamps']
        assert ts == sorted(ts)


# --------------------------------------------------------------------------- #
# 12. CLI: evaluate requires the real model (never mock)
# --------------------------------------------------------------------------- #
def test_cli_evaluate_requires_real_model(monkeypatch):
    import kronos_trading.cli as cli

    class Args:
        db = '/tmp/unused.db'
        symbol = 'BTC/USDT'
        timeframe = '1h'
        context = 10
        horizon = 1
        start = None
        end = None
        max_predictions = 10
        direction_threshold = 0.0005
        seed = 0
        no_deterministic = False
        model = 'NeoQuasar/Kronos-small'
        tokenizer = 'NeoQuasar/Kronos-Tokenizer-base'
        model_revision = None
        tokenizer_revision = None
        device = None
        max_context = 512
        cache_dir = None
        output = None
        include_rows = False

    monkeypatch.setattr(cli, 'load_candles', lambda *a, **k: mk(30))
    # Make the manager report unavailable - the real path must fail loudly.
    monkeypatch.setattr(cli.ModelManager, 'load', lambda self: self)
    with pytest.raises(cli.ModelUnavailableError):
        cli._evaluate(Args())


def test_parse_timestamp_forms():
    assert parse_timestamp('2026-08-01') == parse_timestamp('2026-08-01T00:00:00')
    assert parse_timestamp('2026-08-01T00:00:00Z') == parse_timestamp('2026-08-01')
    assert parse_timestamp('1785571200000') == 1785571200000
    assert parse_timestamp(1785571200000) == 1785571200000
    assert parse_timestamp(None) is None
    import datetime
    dt = datetime.datetime(2026, 8, 1, tzinfo=datetime.timezone.utc)
    assert parse_timestamp('2026-08-01') == int(dt.timestamp() * 1000)


# --------------------------------------------------------------------------- #
# Real-weight evaluation (skips without the model)
# --------------------------------------------------------------------------- #
def test_real_model_evaluation_deterministic():
    pytest.importorskip('torch')
    manager = ModelManager(local_files_only=True).load()
    if not manager.available:
        pytest.skip('real Kronos weights not available in this environment')

    candles = mk(600)
    c = EvaluationConfig(context_length=512, horizon=1, max_predictions=5)
    r1 = PredictionEvaluator(KronosRealPredictor(manager), c,
                             'BTC/USDT', '1h').evaluate(candles)
    r2 = PredictionEvaluator(KronosRealPredictor(manager), c,
                             'BTC/USDT', '1h').evaluate(candles)
    assert len(r1.rows) == 5
    assert [x.asdict() for x in r1.rows] == [x.asdict() for x in r2.rows]
    m = r1.report['metrics']
    assert m['mae_close'] is not None and math.isfinite(m['mae_close'])
    assert r1.report['model_revision'] is not None
    assert r1.report['device'] in ('cpu', 'cuda:0')
