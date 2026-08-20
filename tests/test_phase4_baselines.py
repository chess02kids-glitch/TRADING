"""Phase 4 - naive baseline tests (persistence + previous-direction).

Baselines must never inspect future candles, must use the same prediction
timestamps as Kronos, and must apply the same flat-direction threshold.
"""
import math
from types import SimpleNamespace

import pytest

from kronos_trading.baselines import (
    persistence_prediction,
    previous_direction_prediction,
    baseline_rows_for,
    build_model_comparison,
)
from kronos_trading.evaluation import (
    EvaluationConfig,
    PredictionEvaluator,
    compute_metrics,
)
from kronos_trading.model import PredictorResult
from kronos_trading.types import Candle

H = 3_600_000
BASE = 1_700_000_000_000
THRESHOLD = 0.0005


def mk(n, base=BASE, step=H):
    return [Candle(base + i * step, 100.0 + i * 0.1, 101.0 + i * 0.1,
                   99.0 + i * 0.1, 100.05 + i * 0.1, 10.0) for i in range(n)]


class FakePredictor:
    """Deterministic test double (never presented as real Kronos output)."""

    def __init__(self, close_factor=1.01):
        self.device = 'fake'
        self.dtype = 'torch.float32'
        self.manager = SimpleNamespace(
            model_name='FakeKronos-small',
            resolved_model_revision='fake-model-rev',
            resolved_tokenizer_revision='fake-tokenizer-rev',
            max_context=512)
        self.close_factor = close_factor

    def predict(self, candles, timeframe, horizon=1, temperature=1.0,
                top_k=0, top_p=0.9, sample_count=1, seed=None,
                deterministic=False):
        last = candles[-1].close
        steps = [{'open': last, 'high': last, 'low': last,
                  'close': last * (self.close_factor ** (k + 1)),
                  'volume': 1.0, 'amount': last} for k in range(horizon)]
        return PredictorResult(steps=steps, latency_ms=0.5, peak_vram_bytes=None)


def cfg(**overrides):
    defaults = dict(context_length=10, horizon=1, max_predictions=1000)
    defaults.update(overrides)
    return EvaluationConfig(**defaults)


def _evaluate(n=40, **overrides):
    return PredictionEvaluator(FakePredictor(), cfg(**overrides),
                               'BTC/USDT', '1h').evaluate(mk(n))


# --------------------------------------------------------------------------- #
# 1. Persistence uses only the last available close
# --------------------------------------------------------------------------- #
def test_persistence_uses_only_last_close():
    assert persistence_prediction(123.45) == (123.45, 0.0)
    assert persistence_prediction(0.0) == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# 2. Previous-direction uses only the previous/current closed candles
# --------------------------------------------------------------------------- #
def test_previous_direction_uses_only_last_two_candles():
    full = mk(10)
    ctx_a = full[:8]
    # same last two candles, different earlier history
    ctx_b = [Candle(0, 1, 1, 1, 1, 1),
             Candle(H, 1, 1, 1, 1, 1),
             ctx_a[-2], ctx_a[-1]]
    assert previous_direction_prediction(ctx_a, THRESHOLD) == \
        previous_direction_prediction(ctx_b, THRESHOLD)


def test_previous_direction_formula():
    c0 = Candle(0, 100, 100, 100, 100, 1)
    c1 = Candle(H, 100, 100, 100, 105, 1)  # 5% up
    pred_close, prev_return, d = previous_direction_prediction([c0, c1], THRESHOLD)
    assert prev_return == pytest.approx(0.05)
    assert d == 1
    assert pred_close == pytest.approx(105 * 1.05)


# --------------------------------------------------------------------------- #
# 3. Flat-direction threshold applied identically (0.0005)
# --------------------------------------------------------------------------- #
def test_flat_threshold_applied_identically():
    for ret in (THRESHOLD, -THRESHOLD, 0.0):
        c0 = Candle(0, 100, 100, 100, 100, 1)
        c1 = Candle(H, 100, 100, 100, 100 * (1 + ret), 1)
        _, _, d = previous_direction_prediction([c0, c1], THRESHOLD)
        assert d == 0
    for ret in (0.0006, -0.0006):
        c0 = Candle(0, 100, 100, 100, 100, 1)
        c1 = Candle(H, 100, 100, 100, 100 * (1 + ret), 1)
        _, _, d = previous_direction_prediction([c0, c1], THRESHOLD)
        assert d == (1 if ret > 0 else -1)


def test_baseline_rows_carry_same_threshold_as_kronos():
    res = _evaluate()
    kronos = res.rows[0]
    for kind in ('persistence', 'previous_direction'):
        row = res.baseline_rows[kind][0]
        assert row.direction_threshold == kronos.direction_threshold == THRESHOLD


# --------------------------------------------------------------------------- #
# 4. Baselines never inspect future candles (integration)
# --------------------------------------------------------------------------- #
def test_baselines_never_inspect_future():
    candles = mk(40)
    close_by_ts = {c.timestamp_ms: c.close for c in candles}
    res = PredictionEvaluator(FakePredictor(), cfg(), 'BTC/USDT', '1h').evaluate(candles)

    for krow, prow in zip(res.rows, res.baseline_rows['persistence']):
        # persistence uses the close of the candle ending at context_end (strictly
        # before the prediction timestamp).
        assert prow.context_end_timestamp < prow.prediction_timestamp
        assert prow.predicted_close == close_by_ts[prow.context_end_timestamp]
        assert prow.predicted_return == 0.0
        assert prow.predicted_open is None and prow.predicted_volume is None
        # same actual as Kronos row -> same timestamp / same actual close
        assert prow.prediction_timestamp == krow.prediction_timestamp
        assert prow.actual_close == krow.actual_close

    for krow, drow in zip(res.rows, res.baseline_rows['previous_direction']):
        ctx_end = drow.context_end_timestamp
        prev_ts = ctx_end - H
        prev_return = close_by_ts[ctx_end] / close_by_ts[prev_ts] - 1.0
        assert drow.predicted_return == pytest.approx(prev_return)
        assert drow.predicted_close == pytest.approx(close_by_ts[ctx_end] * (1 + prev_return))
        assert drow.prediction_timestamp == krow.prediction_timestamp
        # context end is strictly before the prediction timestamp
        assert ctx_end < drow.prediction_timestamp


# --------------------------------------------------------------------------- #
# 5. Same timestamps for all comparisons
# --------------------------------------------------------------------------- #
def test_same_timestamps_for_all_comparisons():
    res = _evaluate()
    kronos = res.rows
    pers = res.baseline_rows['persistence']
    prev = res.baseline_rows['previous_direction']
    assert len(kronos) == len(pers) == len(prev) > 0
    assert ([r.prediction_timestamp for r in kronos]
            == [r.prediction_timestamp for r in pers]
            == [r.prediction_timestamp for r in prev])
    assert res.report['baseline_results']['persistence']['predictions'] == len(kronos)
    assert res.report['baseline_results']['previous_direction']['predictions'] == len(kronos)
    assert res.report['model_comparison']['prediction_count']['same_timestamps'] is True


# --------------------------------------------------------------------------- #
# 6. Persistence direction policy (flat -> never scores direction)
# --------------------------------------------------------------------------- #
def test_persistence_never_scores_direction():
    res = _evaluate()
    pers_metrics = res.report['baseline_results']['persistence']
    assert all(r.predicted_return == 0.0 for r in res.baseline_rows['persistence'])
    # mk() candles have positive actual returns, so directional accuracy = 0/positive
    assert pers_metrics['n_positive_actual'] > 0
    assert pers_metrics['directional_accuracy'] == 0.0
    assert pers_metrics['n_directional_correct'] == 0


# --------------------------------------------------------------------------- #
# 7. Empty / zero-variance handled safely
# --------------------------------------------------------------------------- #
def test_empty_and_zero_variance_safe():
    empty = compute_metrics([], THRESHOLD)
    assert empty['mae_close'] is None
    assert empty['directional_accuracy'] is None
    assert empty['return_correlation'] is None

    cmp = build_model_comparison(empty, empty, empty)
    assert cmp['prediction_count']['same_timestamps'] is True  # 0 == 0 == 0
    for block in ('kronos_vs_persistence', 'kronos_vs_previous_direction'):
        for k, v in cmp[block].items():
            if k.endswith('_delta'):
                assert v is None
            if k.endswith('_winner'):
                assert v is None

    # zero-variance returns -> correlation undefined, RMSE defined (0)
    kronos = {'mae_close': 1.0, 'rmse_close': 0.0, 'mape_close': None,
              'directional_accuracy': 1.0, 'return_mae': 0.0, 'return_rmse': 0.0,
              'return_correlation': None, 'predictions': 2}
    cmp2 = build_model_comparison(kronos, kronos, kronos)
    assert cmp2['kronos_vs_persistence']['return_correlation_delta'] is None
    assert cmp2['kronos_vs_persistence']['return_correlation_winner'] is None


# --------------------------------------------------------------------------- #
# 8. Winner labels / deltas are explicit
# --------------------------------------------------------------------------- #
def test_model_comparison_deltas_and_winners():
    kronos = {'mae_close': 100.0, 'rmse_close': 120.0, 'mape_close': 0.01,
              'directional_accuracy': 0.40, 'return_correlation': 0.10,
              'return_mae': 0.01, 'return_rmse': 0.02, 'predictions': 5}
    persistence = {'mae_close': 120.0, 'rmse_close': 150.0, 'mape_close': 0.02,
                   'directional_accuracy': 0.0, 'return_correlation': None,
                   'return_mae': 0.03, 'return_rmse': 0.04, 'predictions': 5}
    prev_dir = {'mae_close': 90.0, 'rmse_close': 100.0, 'mape_close': 0.008,
                'directional_accuracy': 0.50, 'return_correlation': 0.02,
                'return_mae': 0.02, 'return_rmse': 0.03, 'predictions': 5}
    cmp = build_model_comparison(kronos, persistence, prev_dir)

    p = cmp['kronos_vs_persistence']
    assert p['mae_close_delta'] == pytest.approx(-20.0)
    assert p['mae_close_winner'] == 'kronos'
    assert p['rmse_close_delta'] == pytest.approx(-30.0)
    assert p['rmse_close_winner'] == 'kronos'
    assert p['directional_accuracy_delta'] == pytest.approx(0.40)
    assert p['directional_accuracy_winner'] == 'kronos'
    assert p['return_correlation_delta'] is None  # baseline correlation undefined
    assert p['return_correlation_winner'] is None

    d = cmp['kronos_vs_previous_direction']
    assert d['directional_accuracy_delta'] == pytest.approx(-0.10)
    assert d['directional_accuracy_winner'] == 'baseline'
    assert d['return_correlation_delta'] == pytest.approx(0.08)
    assert d['return_correlation_winner'] == 'kronos'
    assert d['mae_close_delta'] == pytest.approx(10.0)
    assert d['mae_close_winner'] == 'baseline'

    assert cmp['prediction_count']['same_timestamps'] is True


# --------------------------------------------------------------------------- #
# 9. Baseline metrics can be computed (integration: close + returns)
# --------------------------------------------------------------------------- #
def test_baseline_metrics_are_meaningful():
    res = _evaluate()
    p = res.report['baseline_results']['persistence']
    d = res.report['baseline_results']['previous_direction']
    for m in (p, d):
        assert m['predictions'] == len(res.rows)
        assert m['mae_close'] is not None and math.isfinite(m['mae_close'])
        assert m['rmse_close'] is not None and math.isfinite(m['rmse_close'])
        assert m['mape_close'] is not None and math.isfinite(m['mape_close'])
        assert m['return_mae'] is not None
        assert m['return_rmse'] is not None
        # open/high/low/volume are undefined for baselines
        assert m['mae_open'] is None
        assert m['mae_high'] is None
        assert m['mae_low'] is None
        assert m['mae_volume'] is None
    # persistence MAE close should equal mean |last_close - actual_close|
    expected = sum(abs(r.predicted_close - r.actual_close)
                   for r in res.baseline_rows['persistence']) / len(res.rows)
    assert p['mae_close'] == pytest.approx(expected)
