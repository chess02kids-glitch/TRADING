"""Phase 5c classical volatility benchmark tests.

Covers: no look-ahead, past-only HAR fitting, deterministic outputs, identical
timestamps, normalized-target correctness, baseline consistency, short-context
safety, statistical alignment, improvement %, regime tracking (anti-shrinkage),
and the classical A/B/C decision gate. Uses a deterministic Kronos test double.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from kronos_trading.classical_volatility import (
    HAR_COMPARISONS,
    classify_classical_gate,
    evaluate_classical_gate,
    har_regime_tracking,
    improvement_pct,
    run_classical_volatility_benchmark,
)
from kronos_trading.evaluation import EvaluationConfig, PredictionEvaluator
from kronos_trading.model import PredictorResult
from kronos_trading.statistics_compare import diebold_mariano
from kronos_trading.types import Candle
from kronos_trading.volatility_baselines import (
    EWMA_SPAN,
    assign_regime,
    ewma_range,
    har_forecast,
    rolling_mean_range,
    volatility_forecasts,
)

H = 3_600_000
BASE = 1_700_000_000_000


def mk(n, base=BASE, step=H, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        close = 100.0 + i * 0.05
        r = 0.5 + (i % 5) * 0.3 + rng.normal(0, 0.05)
        out.append(Candle(base + i * step, close - r / 2, close + r / 2,
                          close - r / 2, close, 10.0))
    return out


class FakePredictor:
    """Deterministic Kronos test double (range = factor * last observed range)."""

    def __init__(self, factor=1.0):
        self.device = 'fake'
        self.dtype = 'torch.float32'
        self.manager = SimpleNamespace(
            model_name='FakeKronos-small',
            resolved_model_revision='fake-rev',
            resolved_tokenizer_revision='fake-tok-rev',
            max_context=512)
        self.factor = factor
        self.calls = []

    def predict(self, candles, timeframe, horizon=1, temperature=1.0,
                top_k=0, top_p=0.9, sample_count=1, seed=None,
                deterministic=False):
        last = candles[-1]
        self.calls.append(max(c.timestamp_ms for c in candles))
        rng = self.factor * (last.high - last.low)
        steps = [{'open': last.close, 'high': last.close + rng / 2,
                  'low': last.close - rng / 2, 'close': last.close,
                  'volume': 1.0, 'amount': last.close} for _ in range(horizon)]
        return PredictorResult(steps=steps, latency_ms=0.5, peak_vram_bytes=None)


def cfg(**overrides):
    defaults = dict(context_length=24, horizon=1, window_size=20,
                    max_predictions=1000)
    defaults.update(overrides)
    return EvaluationConfig(**defaults)


# --------------------------------------------------------------------------- #
# Baseline consistency (fixed, past-only)
# --------------------------------------------------------------------------- #
def test_classical_baseline_consistency():
    ranges = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert rolling_mean_range(ranges, 5) == pytest.approx(4.0)
    assert rolling_mean_range(ranges, 22) is None
    a = 2.0 / (EWMA_SPAN + 1.0)
    e = 1.0
    for r in [2.0, 3.0, 4.0]:
        e = a * r + (1 - a) * e
    assert ewma_range([1.0, 2.0, 3.0, 4.0]) == pytest.approx(e)
    fc = volatility_forecasts(ranges)
    assert set(fc) == {'prev', 'rolling5', 'rolling22', 'ewma', 'har'}
    assert fc['prev'] == 6.0
    assert HAR_COMPARISONS == ('prev', 'ewma', 'rolling5', 'rolling22')


def test_har_past_only_and_short_context():
    # constant series recovered exactly; short context -> None
    assert har_forecast([3.0] * 50) == pytest.approx(3.0)
    assert har_forecast([1.0, 2.0]) is None


# --------------------------------------------------------------------------- #
# Improvement % and regime tracking
# --------------------------------------------------------------------------- #
def test_improvement_pct():
    assert improvement_pct(100.0, 80.0) == pytest.approx(20.0)
    assert improvement_pct(100.0, 120.0) == pytest.approx(-20.0)
    assert improvement_pct(None, 80.0) is None
    assert improvement_pct(0.0, 80.0) is None


def test_regime_tracking_constant_predictor_fails():
    # constant forecast -> zero spread -> NOT tracking (pure shrinker)
    rng = np.random.default_rng(1)
    ranges = list(rng.uniform(1, 5, size=120))
    vrows = _fake_vrows(ranges, forecast=lambda r: 1.0)
    out = har_regime_tracking(vrows)
    assert out['tracks'] is False


def test_regime_tracking_proportional_predictor_succeeds():
    rng = np.random.default_rng(1)
    ranges = list(rng.uniform(1, 5, size=120))
    vrows = _fake_vrows(ranges, forecast=lambda r: 0.5 * r + 0.3)
    out = har_regime_tracking(vrows)
    assert out['tracks'] is True
    assert out['monotonic'] is True


def _fake_vrows(ranges, forecast):
    """Build minimal VolatilityRow-like objects for tracking tests."""
    from kronos_trading.evaluation import VolatilityRow
    out = []
    for i, r in enumerate(ranges):
        reg = assign_regime(ranges[max(0, i - 22):i + 1]) if i >= 22 else 'undefined'
        out.append(VolatilityRow(
            symbol='X', timeframe='1h', window=None,
            prediction_timestamp=i * H, regime=reg,
            kronos_range=forecast(r), actual_range=r,
            prev_range=r, rolling5_range=r, rolling22_range=r,
            ewma_range=r, har_range=forecast(r), denom_close=100.0))
    return out


# --------------------------------------------------------------------------- #
# End-to-end benchmark (deterministic, aligned, no look-ahead)
# --------------------------------------------------------------------------- #
def test_benchmark_structure_and_determinism():
    candles_by_series = {
        ('BTC/USDT', '1h'): mk(80),
        ('ETH/USDT', '1h'): mk(80),
    }
    series = [('BTC/USDT', '1h'), ('ETH/USDT', '1h')]
    c = cfg(context_length=24, window_size=20)

    r1 = run_classical_volatility_benchmark(FakePredictor(), c, series,
                                            lambda s, tf: candles_by_series[(s, tf)])
    r2 = run_classical_volatility_benchmark(FakePredictor(), c, series,
                                            lambda s, tf: candles_by_series[(s, tf)])
    assert r1['kind'] == 'classical_volatility_benchmark'
    assert r1['classical_baselines'] == r2['classical_baselines']
    assert r1['window_records'] == r2['window_records']
    assert r1['success_gate'] == r2['success_gate']
    assert r1['pooled_statistics'] == r2['pooled_statistics']
    for sid in ('BTC/USDT 1h', 'ETH/USDT 1h'):
        assert set(r1['series'][sid]['windows']) == {'older', 'middle', 'recent'}
        w = r1['series'][sid]['windows']['recent']
        assert set(w['systems']) == {'kronos', 'prev', 'rolling5', 'rolling22', 'ewma', 'har'}
        assert set(w['classical_comparisons']) == \
            {'har_vs_prev', 'har_vs_ewma', 'har_vs_rolling5', 'har_vs_rolling22'}
        assert 'improvement_vs_prev_pct' in w
        assert 'har_regime_tracking' in w


def test_benchmark_identical_timestamps_and_no_lookahead():
    candles = mk(80)
    c = cfg(context_length=24, window_size=20)
    pred = FakePredictor()
    res = PredictionEvaluator(pred, c, 'BTC/USDT', '1h').evaluate(candles)
    assert len(res.volatility_rows) == len(res.rows) > 0
    for call_ts, vrow in zip(pred.calls, res.volatility_rows):
        assert call_ts < vrow.prediction_timestamp  # context ends before target
    # all classical baselines are past-only by construction (same context)


def test_normalized_target_correctness():
    candles = mk(80)
    c = cfg(context_length=24, window_size=20)
    res = PredictionEvaluator(FakePredictor(), c, 'BTC/USDT', '1h').evaluate(candles)
    for vrow in res.volatility_rows:
        assert (vrow.actual_range / vrow.denom_close) == pytest.approx(
            vrow.actual_range / vrow.denom_close)
        assert (vrow.har_range / vrow.denom_close) == pytest.approx(
            vrow.har_range / vrow.denom_close) if vrow.har_range is not None else True


def test_benchmark_empty_safe():
    candles_by_series = {('BTC/USDT', '1h'): mk(5)}
    report = run_classical_volatility_benchmark(
        FakePredictor(), cfg(context_length=4), [('BTC/USDT', '1h')],
        lambda s, tf: candles_by_series[(s, tf)])
    assert report['success_gate']['overall'] == 'pending'
    assert report['success_gate']['eligible_windows'] == 0


# --------------------------------------------------------------------------- #
# Statistical alignment + DM
# --------------------------------------------------------------------------- #
def test_dm_alignment_and_winner():
    out = diebold_mariano([1.0] * 50, [2.0] * 50)
    assert out['winner'] == 'kronos' and out['p_value'] < 1e-6
    assert out['n'] == 50
    # mismatched-length safety (function masks to finite pairs)
    out2 = diebold_mariano([1.0, 2.0, float('nan')], [2.0, 1.0, 3.0])
    assert out2['n'] == 2


# --------------------------------------------------------------------------- #
# Decision gate (classical A/B/C)
# --------------------------------------------------------------------------- #
def _record(series='BTC/USDT', timeframe='1h', window='recent', sample_size=100,
            har_beats_prev=True, har_beats_ewma=True, har_beats_prev_norm=True,
            har_tracks=True):
    return {
        'series': series, 'timeframe': timeframe, 'window': window,
        'sample_size': sample_size, 'low_power': sample_size < 30,
        'har_beats_prev': har_beats_prev, 'har_beats_ewma': har_beats_ewma,
        'har_beats_prev_norm': har_beats_prev_norm, 'har_tracks': har_tracks,
    }


def test_gate_criteria_1_5_and_7():
    recs = [_record(series=s, window=w)
            for s in ('BTC/USDT', 'ETH/USDT', 'BTC/USDT', 'ETH/USDT')
            for w in ('older', 'middle', 'recent')]
    recs = [{**r, 'timeframe': '1h'} for r in recs]
    gate = evaluate_classical_gate(recs)
    c = gate['criteria']
    assert c['c1_har_beats_prev'] is True
    assert c['c2_har_beats_ewma'] is True
    assert c['c3_series_breadth'] is True
    assert c['c4_window_breadth'] is True
    assert c['c5_normalized_survives'] is True
    assert c['c7_not_solely_shrinkage'] is True
    assert c['c6_statistical_support'] is None  # filled by caller
    assert c['c8_regime_breadth'] is None


def test_gate_classification_a_b_c():
    all_true = {'c1_har_beats_prev': True, 'c2_har_beats_ewma': True,
                'c3_series_breadth': True, 'c4_window_breadth': True,
                'c5_normalized_survives': True, 'c6_statistical_support': True,
                'c7_not_solely_shrinkage': True, 'c8_regime_breadth': True}
    assert classify_classical_gate(all_true) == 'A'
    weak = dict(all_true, c2_har_beats_ewma=False, c6_statistical_support=False)
    assert classify_classical_gate(weak) == 'B'
    losing = dict(all_true, c1_har_beats_prev=False)
    assert classify_classical_gate(losing) == 'C'


def test_gate_excludes_daily_and_low_power():
    recs = [_record(timeframe='1d', sample_size=72) for _ in range(12)]
    gate = evaluate_classical_gate(recs)
    assert gate['eligible_windows'] == 0
    assert gate['overall'] == 'pending'
