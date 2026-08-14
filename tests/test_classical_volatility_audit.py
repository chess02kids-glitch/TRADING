"""Regression tests for the Phase 5c audit fixes.

These pin down two things discovered in the audit:

1. (genuine bug) the c6 gate criterion was computed with
   ``dm['winner'] == 'har'``, but ``diebold_mariano`` returned the hardcoded
   labels ``'kronos'/'baseline'``, so c6 was ALWAYS False regardless of the
   p-value. c6 is now computed label-independently from p-value + sign of the
   mean loss difference.

2. (naming bug) the DM winner labels for the classical HAR-vs-* comparisons
   reported ``'kronos'`` instead of ``'har'``. ``diebold_mariano`` now takes
   ``a_name``/``b_name``, and the classical path passes ``'har'``/``'baseline'``.

The DM statistic, p-value, and mean loss difference were always correct (system
A = HAR, negative diff = HAR better); these tests lock in that convention.
"""
import json
from types import SimpleNamespace

import pytest

from kronos_trading.classical_volatility import (
    _c6_from_dm,
    _c8_from_regime_pool,
    classical_pairwise,
    evaluate_classical_gate,
    recompute_classical_gate,
    recompute_classical_summary,
    run_classical_volatility_benchmark,
)
from kronos_trading.evaluation import EvaluationConfig, VolatilityRow
from kronos_trading.model import PredictorResult
from kronos_trading.statistics_compare import diebold_mariano
from kronos_trading.types import Candle

H = 3_600_000
BASE = 1_700_000_000_000


def mk(n, seed=7):
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        close = 100.0 + i * 0.05
        r = 0.5 + (i % 5) * 0.3 + rng.normal(0, 0.05)
        out.append(Candle(BASE + i * H, close - r / 2, close + r / 2,
                          close - r / 2, close, 10.0))
    return out


class FakePredictor:
    def __init__(self, factor=1.0):
        self.device = 'fake'
        self.dtype = 'torch.float32'
        self.manager = SimpleNamespace(
            model_name='FakeKronos-small',
            resolved_model_revision='fake-rev',
            resolved_tokenizer_revision='fake-tok-rev',
            max_context=512)
        self.factor = factor

    def predict(self, candles, timeframe, horizon=1, temperature=1.0,
                top_k=0, top_p=0.9, sample_count=1, seed=None,
                deterministic=False):
        last = candles[-1]
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
# DM sign convention + labels
# --------------------------------------------------------------------------- #
def test_dm_sign_convention_documented():
    # system A better -> negative mean loss diff -> winner == a_name
    out = diebold_mariano([1.0] * 50, [2.0] * 50, a_name='har', b_name='baseline')
    assert out['mean_loss_diff'] == -1.0
    assert out['winner'] == 'har'
    assert out['dm_statistic'] < 0
    # system B better -> positive diff -> winner == b_name
    out2 = diebold_mariano([2.0] * 50, [1.0] * 50, a_name='har', b_name='baseline')
    assert out2['mean_loss_diff'] == 1.0
    assert out2['winner'] == 'baseline'
    # tie
    out3 = diebold_mariano([1.0] * 20, [1.0] * 20, a_name='har', b_name='baseline')
    assert out3['winner'] == 'tie'


def test_dm_legacy_default_labels_unchanged():
    # legacy Kronos path keeps its default labels
    out = diebold_mariano([1.0] * 30, [2.0] * 30)
    assert out['winner'] == 'kronos'


def test_classical_pairwise_reports_har_not_kronos():
    import numpy as np
    rng = np.random.default_rng(4)
    vrows = []
    for i in range(60):
        reg = 'medium'
        vrows.append(VolatilityRow(
            symbol='X', timeframe='1h', window=None, prediction_timestamp=i * H,
            regime=reg, kronos_range=1.0, actual_range=float(rng.uniform(1, 3)),
            prev_range=2.0, rolling5_range=1.5, rolling22_range=1.5,
            ewma_range=1.5, har_range=1.0, denom_close=100.0))
    out = classical_pairwise(vrows, 'har', 'prev')
    assert out['dm']['winner'] in ('har', 'baseline', 'tie')
    assert out['dm']['winner'] != 'kronos'
    assert 'har' in out['dm']['note']


# --------------------------------------------------------------------------- #
# c6 criterion (label-independent)
# --------------------------------------------------------------------------- #
def test_c6_true_when_significant_and_har_better():
    assert _c6_from_dm({'p_value': 2.1494e-26, 'mean_loss_diff': -27.5322}) is True
    assert _c6_from_dm({'p_value': 0.001, 'mean_loss_diff': -0.1}) is True


def test_c6_false_when_not_significant_or_wrong_sign():
    assert _c6_from_dm({'p_value': 0.5, 'mean_loss_diff': -27.53}) is False
    assert _c6_from_dm({'p_value': 1e-20, 'mean_loss_diff': 27.53}) is False
    assert _c6_from_dm({'p_value': None, 'mean_loss_diff': -27.53}) is False
    assert _c6_from_dm({}) is False


def test_c8_from_regime_pool():
    pool = {'low': {'har_beats_prev': True}, 'medium': {'har_beats_prev': True},
            'high': {'har_beats_prev': False}}
    assert _c8_from_regime_pool(pool) is True
    pool2 = {'low': {'har_beats_prev': True}, 'medium': {'har_beats_prev': False},
             'high': {'har_beats_prev': False}}
    assert _c8_from_regime_pool(pool2) is False


# --------------------------------------------------------------------------- #
# Recompute gate from a saved report (corrected c6/c8)
# --------------------------------------------------------------------------- #
def _records():
    return [
        {'series': s, 'timeframe': '1h', 'window': w,
         'sample_size': 100, 'low_power': False,
         'har_beats_prev': True, 'har_beats_ewma': True,
         'har_beats_prev_norm': True, 'har_tracks': True}
        for s in ('BTC/USDT', 'ETH/USDT')
        for w in ('older', 'middle', 'recent')
    ]


def _report(p_value=2.1494e-26, mean_loss_diff=-27.5322,
            regime_beats=(True, True, True)):
    return {
        'window_records': _records(),
        'pooled_statistics': {
            'prev': {'raw': {'p_value': p_value, 'mean_loss_diff': mean_loss_diff,
                             'winner': 'kronos',  # stale label from old code
                             'dm_statistic': -10.63}},
            'ewma': {'raw': {'p_value': 0.5, 'mean_loss_diff': 0.0, 'winner': 'tie'}},
            'rolling5': {'raw': {'p_value': 0.5, 'mean_loss_diff': 0.0}},
            'rolling22': {'raw': {'p_value': 0.5, 'mean_loss_diff': 0.0}},
        },
        'regime_pooled': {
            'low': {'har_beats_prev': regime_beats[0]},
            'medium': {'har_beats_prev': regime_beats[1]},
            'high': {'har_beats_prev': regime_beats[2]},
        },
        'kronos_vs_best_classical': {},
    }


def test_recompute_gate_c6_becomes_true():
    report = _report()  # p = 2.15e-26 < 0.0125, mean loss diff < 0
    gate = recompute_classical_gate(report)
    c = gate['criteria']
    assert c['c6_statistical_support'] is True
    assert all(c.values())
    assert gate['verdict'] == 'A'
    assert gate['verdict_meaning'] == 'classical volatility predictability established'


def test_recompute_gate_c6_stays_false_when_not_significant():
    gate = recompute_classical_gate(_report(p_value=0.5))
    assert gate['criteria']['c6_statistical_support'] is False
    assert gate['verdict'] == 'B'  # beats prev (c1) but statistical support fails


def test_recompute_gate_c8_from_regime_pool():
    gate = recompute_classical_gate(_report(regime_beats=(True, False, False)))
    assert gate['criteria']['c8_regime_breadth'] is False
    assert gate['verdict'] == 'B'


def test_recompute_summary_shape():
    summ = recompute_classical_summary(_report())
    assert set(summ) == {'corrected_gate', 'pooled_primary_dm', 'kronos_vs_best_classical',
                         'verdict', 'verdict_meaning'}
    assert summ['verdict'] == 'A'
    assert summ['pooled_primary_dm']['har_vs_prev']['dm_statistic'] == -10.63


def test_live_benchmark_produces_correct_labels_and_gate():
    """End-to-end: the live run now emits 'har'/'baseline' winners and a
    consistent gate (regression guard for the fix)."""
    import numpy as np
    rng = np.random.default_rng(9)
    candles_by_series = {
        ('BTC/USDT', '1h'): mk(200),
        ('ETH/USDT', '1h'): mk(200),
    }
    series = [('BTC/USDT', '1h'), ('ETH/USDT', '1h')]
    # context 64 (>= HAR's 22+24 minimum) so HAR forecasts are defined
    report = run_classical_volatility_benchmark(
        FakePredictor(), cfg(context_length=64, window_size=40), series,
        lambda s, tf: candles_by_series[(s, tf)])

    # every HAR-vs-* DM winner is 'har'/'baseline'/'tie', never 'kronos'
    for sid in ('BTC/USDT 1h', 'ETH/USDT 1h'):
        for w, analysis in report['series'][sid]['windows'].items():
            for key, comp in analysis['classical_comparisons'].items():
                assert comp['raw']['dm']['winner'] in ('har', 'baseline', 'tie')
                assert comp['normalized']['dm']['winner'] in ('har', 'baseline', 'tie')
    # pooled labels too
    for key, d in report['pooled_statistics'].items():
        assert d['raw']['winner'] in ('har', 'baseline', 'tie')
        assert d['normalized']['winner'] in ('har', 'baseline', 'tie')
    # gate's c6 was computed label-independently
    assert 'c6_statistical_support' in report['success_gate']['criteria']
