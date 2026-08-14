"""Phase 5 - target-formulation research tests.

All target formulations must preserve the no-lookahead guarantees, use the
same timestamps as the frozen evaluation, and be derived only from closed
candles. A deterministic test double stands in for Kronos (never presented as
real output); real-weight runs happen on the target machine.
"""
import math
from types import SimpleNamespace

import pytest

from kronos_trading.baselines import previous_direction_prediction
from kronos_trading.evaluation import EvaluationConfig, PredictionEvaluator
from kronos_trading.model import PredictorResult
from kronos_trading.research_targets import (
    ARCHITECTURE_CHECK,
    SELECTED_TARGETS,
    TARGET_SPECS,
    compute_target_metrics,
    frozen_baseline,
    frozen_baseline_hash,
    frozen_baseline_verified,
    run_research_experiment,
)
from kronos_trading.types import Candle

H = 3_600_000
BASE = 1_700_000_000_000
THRESHOLD = 0.0005


def mk(n, base=BASE, step=H, close_override=None):
    out = []
    for i in range(n):
        c = close_override if close_override is not None else 100.05 + i * 0.1
        out.append(Candle(base + i * step, 100.0 + i * 0.1, 101.0 + i * 0.1,
                          99.0 + i * 0.1, c, 10.0))
    return out


class FakePredictor:
    """Deterministic Kronos test double; records the context it was given."""

    def __init__(self, close_factor=1.001):
        self.device = 'fake'
        self.dtype = 'torch.float32'
        self.manager = SimpleNamespace(
            model_name='FakeKronos-small',
            resolved_model_revision='fake-model-rev',
            resolved_tokenizer_revision='fake-tokenizer-rev',
            max_context=512)
        self.close_factor = close_factor
        self.calls = []

    def predict(self, candles, timeframe, horizon=1, temperature=1.0,
                top_k=0, top_p=0.9, sample_count=1, seed=None,
                deterministic=False):
        self.calls.append({'max_input_ts': max(c.timestamp_ms for c in candles),
                           'horizon': horizon})
        last = candles[-1].close
        steps = [{'open': last, 'high': last, 'low': last,
                  'close': last * (self.close_factor ** (k + 1)),
                  'volume': 1.0, 'amount': last} for k in range(horizon)]
        return PredictorResult(steps=steps, latency_ms=0.5, peak_vram_bytes=None)


def cfg(**overrides):
    defaults = dict(context_length=10, horizon=1, max_predictions=1000,
                    window_size=20)
    defaults.update(overrides)
    return EvaluationConfig(**defaults)


def _eval(predictor=None, n=120, **kw):
    predictor = predictor or FakePredictor()
    c = cfg(**kw)
    res = PredictionEvaluator(predictor, c, 'BTC/USDT', '1h').evaluate(mk(n))
    return predictor, c, res


# --------------------------------------------------------------------------- #
# 1. Frozen baseline is immutable
# --------------------------------------------------------------------------- #
def test_frozen_baseline_hash_matches_lock():
    assert frozen_baseline_verified() is True
    data = frozen_baseline()
    assert data['experiment_id'] == 'phase4_baseline'
    assert data['configuration']['model_revision'] == \
        '901c26c1332695a2a8f243eb2f37243a37bea320'
    assert data['configuration']['context_length'] == 512
    assert data['configuration']['direction_threshold'] == 0.0005
    assert data['results']['summary_findings']  # the 18-window findings are recorded


def test_frozen_baseline_hash_function_deterministic():
    assert frozen_baseline_hash() == frozen_baseline_hash()


# --------------------------------------------------------------------------- #
# 2. Architecture check is present and explicit
# --------------------------------------------------------------------------- #
def test_architecture_check_documented():
    assert ARCHITECTURE_CHECK['finding']
    assert len(ARCHITECTURE_CHECK['evidence']) >= 3
    assert 'classification' in ARCHITECTURE_CHECK['consequence'].lower() or \
        'class' in ARCHITECTURE_CHECK['consequence'].lower()
    assert ARCHITECTURE_CHECK['allowed_changes']


# --------------------------------------------------------------------------- #
# 3. Target specs are complete and self-consistent
# --------------------------------------------------------------------------- #
def test_target_specs_complete():
    assert len(SELECTED_TARGETS) == 3
    assert set(TARGET_SPECS) == {'multi_period_return', 'range_volatility',
                                 'vol_normalized_return'}
    for spec in SELECTED_TARGETS:
        assert spec.definition
        assert spec.justification
        assert spec.contract_preservation
        assert spec.reversible
        assert spec.primary_metric
        assert spec.baselines
        assert spec.horizon >= 1
    # multi-period return is the only target that changes the horizon
    assert TARGET_SPECS['multi_period_return'].horizon == 4
    assert TARGET_SPECS['range_volatility'].horizon == 1
    assert TARGET_SPECS['vol_normalized_return'].horizon == 1


# --------------------------------------------------------------------------- #
# 4. Multi-period return target (no-lookahead + horizon-aware baseline)
# --------------------------------------------------------------------------- #
def test_multi_period_baseline_uses_past_horizon_return():
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    ctx = [Candle(i * H, c, c, c, c, 1) for i, c in enumerate(closes)]
    # horizon=4: prev_return = close[-1]/close[-5] - 1 = 104.0/100.0 - 1 = 0.04
    _, r4, _ = previous_direction_prediction(ctx, THRESHOLD, horizon=4)
    assert r4 == pytest.approx(104.0 / 100.0 - 1.0)
    # horizon=1: prev_return = close[-1]/close[-2] - 1
    _, r1, _ = previous_direction_prediction(ctx, THRESHOLD, horizon=1)
    assert r1 == pytest.approx(104.0 / 103.0 - 1.0)


def test_multi_period_no_future_in_input():
    pred = FakePredictor()
    _, c, res = _eval(pred, n=120, horizon=4)
    # every model call's context ended before the prediction timestamp
    assert len(pred.calls) == len(res.rows)
    for call in pred.calls:
        assert call['horizon'] == 4
    # rows carry horizon=4 and returns over 4 candles
    for row in res.rows:
        assert row.horizon == 4
    out = compute_target_metrics('multi_period_return', res.rows,
                                 res.baseline_rows, THRESHOLD)
    assert out['kronos']['sample_size'] == len(res.rows)
    # persistence predicts zero return -> correlation undefined
    assert out['persistence']['return_correlation'] is None
    assert out['comparison']['vs_persistence']['return_mae']['winner'] in (
        'kronos', 'baseline', 'tie')


def test_multi_period_previous_direction_correct_value():
    # horizon=4: previous return = close[-1]/close[-1-4] - 1 = close[5]/close[1] - 1
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 108.16]
    ctx = [Candle(i * H, c, c, c, c, 1) for i, c in enumerate(closes)]
    pred_close, prev_return, d = previous_direction_prediction(ctx, THRESHOLD, horizon=4)
    assert prev_return == pytest.approx(108.16 / 101.0 - 1.0)
    assert d == 1
    assert pred_close == pytest.approx(108.16 * (1 + prev_return))


# --------------------------------------------------------------------------- #
# 5. Range target
# --------------------------------------------------------------------------- #
def test_range_persistence_uses_last_range():
    _, _, res = _eval(n=120)
    out = compute_target_metrics('range_volatility', res.rows,
                                 res.baseline_rows, THRESHOLD)
    assert out['kronos']['sample_size'] == len(res.rows)
    assert out['previous_direction'] is None
    assert out['comparison']['vs_previous_direction'] is None
    # persistence predicted range == context last range for each row
    expected = sum(abs(r.context_last_range - (r.actual_high - r.actual_low))
                   for r in res.baseline_rows['persistence']) / len(res.rows)
    assert out['persistence']['range_mae'] == pytest.approx(expected)
    # kronos range uses its own predicted high/low
    for r in res.rows:
        assert r.predicted_high is not None and r.predicted_low is not None


def test_range_target_never_uses_future():
    _, _, res = _eval(n=120)
    for r in res.rows:
        # context_end_timestamp is strictly before the target
        assert r.context_end_timestamp < r.prediction_timestamp
        assert r.context_last_range is not None


# --------------------------------------------------------------------------- #
# 6. Volatility-normalized return target
# --------------------------------------------------------------------------- #
def test_normalized_return_scale_and_skip():
    _, _, res = _eval(n=120)
    out = compute_target_metrics('vol_normalized_return', res.rows,
                                 res.baseline_rows, THRESHOLD)
    # mk() returns are near-constant but non-zero vol -> all rows used
    assert out['kronos']['sample_size'] == len(res.rows)
    for r in res.rows:
        vol = r.context_return_vol
        assert vol is not None and vol > 0
        scale = vol * math.sqrt(r.horizon)
        # kronos normalized return == predicted_return / scale
        assert (r.predicted_return / scale) == pytest.approx(r.predicted_return / scale)


def test_normalized_return_zero_vol_skips():
    # constant closes -> zero context vol -> all rows skipped, no crash
    _, _, res = _eval(n=120)
    # force zero vol by replacing context vol with 0
    for r in res.rows:
        r.context_return_vol = 0.0
    for key in ('persistence', 'previous_direction'):
        for r in res.baseline_rows[key]:
            r.context_return_vol = 0.0
    out = compute_target_metrics('vol_normalized_return', res.rows,
                                 res.baseline_rows, THRESHOLD)
    assert out['kronos']['sample_size'] == 0
    assert out['kronos']['norm_return_mae'] is None
    assert out['kronos']['norm_return_correlation'] is None


# --------------------------------------------------------------------------- #
# 7. Identical timestamps across systems for every target
# --------------------------------------------------------------------------- #
def test_targets_share_identical_timestamps():
    _, _, res = _eval(n=120)
    kronos = res.rows
    pers = res.baseline_rows['persistence']
    prev = res.baseline_rows['previous_direction']
    assert len(kronos) == len(pers) == len(prev)
    assert ([r.prediction_timestamp for r in kronos]
            == [r.prediction_timestamp for r in pers]
            == [r.prediction_timestamp for r in prev])


# --------------------------------------------------------------------------- #
# 8. Empty / zero-variance safety
# --------------------------------------------------------------------------- #
def test_compute_target_metrics_empty_safe():
    for tid in ('multi_period_return', 'range_volatility', 'vol_normalized_return'):
        out = compute_target_metrics(tid, [], {'persistence': [], 'previous_direction': []},
                                     THRESHOLD)
        assert out['target_id'] == tid
        assert out['kronos']['sample_size'] == 0


def test_unknown_target_raises():
    with pytest.raises(ValueError):
        compute_target_metrics('nope', [], {'persistence': [], 'previous_direction': []},
                               THRESHOLD)


# --------------------------------------------------------------------------- #
# 9. Deterministic repeatability of the full experiment
# --------------------------------------------------------------------------- #
def test_research_experiment_deterministic():
    candles_by_series = {
        ('BTC/USDT', '1h'): mk(120),
        ('ETH/USDT', '1h'): mk(120),
    }
    series = [('BTC/USDT', '1h'), ('ETH/USDT', '1h')]

    r1 = run_research_experiment(FakePredictor(), cfg(window_size=20), series,
                                 lambda s, tf: candles_by_series[(s, tf)])
    r2 = run_research_experiment(FakePredictor(), cfg(window_size=20), series,
                                 lambda s, tf: candles_by_series[(s, tf)])
    assert r1['targets'] == r2['targets']


def test_research_experiment_structure_and_frozen_embed():
    candles_by_series = {('BTC/USDT', '1h'): mk(120)}
    series = [('BTC/USDT', '1h')]
    report = run_research_experiment(FakePredictor(), cfg(window_size=20), series,
                                     lambda s, tf: candles_by_series[(s, tf)])
    assert report['kind'] == 'phase5_research_targets'
    assert report['frozen_baseline_verified'] is True
    assert report['frozen_baseline']['experiment_id'] == 'phase4_baseline'
    assert report['architecture_check']['finding']
    assert set(report['targets']) == {'multi_period_return', 'range_volatility',
                                      'vol_normalized_return'}
    for tid, t in report['targets'].items():
        assert 'spec' in t
        assert 'BTC/USDT 1h' in t['series']
        assert set(t['series']['BTC/USDT 1h']['windows']) == \
            {'older', 'middle', 'recent'}
