"""Phase 4 robustness tests: multi-window generalization + paired statistics.

Uses a deterministic test double (never presented as real Kronos output) plus
self-contained statistical functions. Real-weight tests skip without the model.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from kronos_trading.evaluation import (
    EvaluationConfig,
    PredictionEvaluator,
    define_windows,
)
from kronos_trading.model import PredictorResult
from kronos_trading.robustness import run_robustness, summarize_windows
from kronos_trading.statistics_compare import (
    bootstrap_mean_ci,
    build_statistical_comparison,
    mcnemar,
    paired_direction_comparison,
    paired_error_comparison,
    wilcoxon_signed_rank,
)
from kronos_trading.types import Candle

H = 3_600_000
BASE = 1_700_000_000_000
THRESHOLD = 0.0005


def mk(n, base=BASE, step=H):
    return [Candle(base + i * step, 100.0 + i * 0.1, 101.0 + i * 0.1,
                   99.0 + i * 0.1, 100.05 + i * 0.1, 10.0) for i in range(n)]


class FakePredictor:
    def __init__(self, close_factor=1.001):
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
    defaults = dict(context_length=10, horizon=1, max_predictions=1000,
                    window_size=20)
    defaults.update(overrides)
    return EvaluationConfig(**defaults)


# --------------------------------------------------------------------------- #
# 1. Window definition
# --------------------------------------------------------------------------- #
def test_windows_abundant_data_three_nonoverlapping():
    specs, info = define_windows(n_targets=89, window_size=20)
    names = [s.name for s in specs]
    assert names == ['older', 'middle', 'recent']
    assert info['window_size_effective'] == 20
    assert info['reduced_due_to_volume'] is False
    assert info['windows_omitted'] == 0
    # older = first 20, recent = last 20
    assert specs[0].start == 0 and specs[0].end == 19
    assert specs[-1].end == 88 and specs[-1].start == 69
    ranges = [(s.start, s.end) for s in specs]
    for i in range(len(ranges) - 1):
        assert ranges[i][1] < ranges[i + 1][0]  # strictly non-overlapping


def test_windows_reduced_when_volume_insufficient():
    specs, info = define_windows(n_targets=20, window_size=100)
    assert info['reduced_due_to_volume'] is True
    assert info['window_size_effective'] == 6  # 20 // 3
    assert info['windows_produced'] == 3
    ranges = [(s.start, s.end) for s in specs]
    for i in range(len(ranges) - 1):
        assert ranges[i][1] < ranges[i + 1][0]
    # all windows fit within [0, 19]
    assert all(s.start >= 0 and s.end <= 19 for s in specs)


def test_windows_very_small_data_fewer_windows():
    specs, info = define_windows(n_targets=2, window_size=100)
    assert info['windows_produced'] == 2
    assert info['windows_omitted'] == 1
    assert [s.name for s in specs] == ['older', 'recent']
    ranges = [(s.start, s.end) for s in specs]
    assert ranges[0][1] < ranges[1][0]

    specs1, info1 = define_windows(n_targets=1, window_size=100)
    assert info1['windows_produced'] == 1
    assert info1['windows_omitted'] == 2
    assert [s.name for s in specs1] == ['recent']


def test_windows_empty():
    specs, info = define_windows(n_targets=0, window_size=20)
    assert specs == []
    assert info['windows_produced'] == 0


# --------------------------------------------------------------------------- #
# 2. evaluate_windows vs single evaluate (recent window consistency)
# --------------------------------------------------------------------------- #
def test_recent_window_matches_single_recent_evaluation():
    candles = mk(100)
    c = cfg(context_length=10, max_predictions=20, window_size=20)
    single = PredictionEvaluator(FakePredictor(), c, 'BTC/USDT', '1h').evaluate(candles)
    windows, info = PredictionEvaluator(FakePredictor(), c, 'BTC/USDT', '1h').evaluate_windows(candles)
    recent = windows['recent']
    assert len(single.rows) == len(recent.rows) == 20
    assert [r.prediction_timestamp for r in single.rows] == \
        [r.prediction_timestamp for r in recent.rows]
    assert single.report['metrics'] == recent.report['metrics']
    assert single.report['model_comparison'] == recent.report['model_comparison']


# --------------------------------------------------------------------------- #
# 3. No overlap / leakage between windows
# --------------------------------------------------------------------------- #
def test_windows_are_chronologically_disjoint():
    candles = mk(200)
    c = cfg(context_length=10, window_size=30)
    windows, info = PredictionEvaluator(FakePredictor(), c, 'BTC/USDT', '1h').evaluate_windows(candles)
    names = ['older', 'middle', 'recent']
    ts_sets = [set(r.prediction_timestamp for r in windows[n].rows) for n in names]
    # pairwise disjoint
    for i in range(3):
        for j in range(i + 1, 3):
            assert ts_sets[i].isdisjoint(ts_sets[j])
    # strictly increasing across windows
    assert max(ts_sets[0]) < min(ts_sets[1]) < min(ts_sets[2])


def test_identical_timestamps_across_systems_per_window():
    candles = mk(120)
    c = cfg(context_length=10, window_size=20)
    windows, _ = PredictionEvaluator(FakePredictor(), c, 'BTC/USDT', '1h').evaluate_windows(candles)
    for name, res in windows.items():
        kronos = res.rows
        pers = res.baseline_rows['persistence']
        prev = res.baseline_rows['previous_direction']
        assert len(kronos) == len(pers) == len(prev)
        assert ([r.prediction_timestamp for r in kronos]
                == [r.prediction_timestamp for r in pers]
                == [r.prediction_timestamp for r in prev])


# --------------------------------------------------------------------------- #
# 4. Statistical comparison correctness
# --------------------------------------------------------------------------- #
def test_bootstrap_mean_ci_constant():
    out = bootstrap_mean_ci([0.0] * 10)
    assert out['mean'] == 0.0
    assert out['ci_low'] == out['ci_high'] == 0.0


def test_bootstrap_mean_ci_empty():
    out = bootstrap_mean_ci([])
    assert out['n'] == 0 and out['mean'] is None and out['ci_low'] is None


def test_wilcoxon_all_negative_small_p():
    out = wilcoxon_signed_rank([-i for i in range(1, 21)])
    assert out['n_nonzero'] == 20
    assert out['statistic'] == 0.0
    assert out['p_value'] < 0.001


def test_wilcoxon_symmetric_p_one():
    out = wilcoxon_signed_rank([1, -1, 2, -2, 3, -3, 4, -4, 5, -5])
    assert out['p_value'] == pytest.approx(1.0, abs=1e-6)


def test_wilcoxon_no_nonzero():
    out = wilcoxon_signed_rank([0.0, 0.0])
    assert out['n_nonzero'] == 0 and out['p_value'] is None


def test_mcnemar_cases():
    assert mcnemar(0, 0)['p_value'] is None
    assert mcnemar(2, 0)['p_value'] == pytest.approx(0.5)
    assert mcnemar(10, 0)['p_value'] == pytest.approx(2 * 0.5 ** 10)
    out = mcnemar(30, 10)
    assert out['method'] == 'mcnemar_chi2_continuity_corrected'
    assert out['p_value'] < 0.01


def test_paired_error_comparison_winner():
    out = paired_error_comparison([0.0] * 10, [1.0] * 10)
    assert out['sample_size'] == 10
    assert out['mean_diff'] == -1.0
    assert out['winner_by_mean'] == 'kronos'
    assert out['bootstrap_ci_95'] == [-1.0, -1.0]
    assert out['wilcoxon_p_value'] < 0.01


def test_paired_direction_comparison_winner():
    out = paired_direction_comparison([True] * 10, [False] * 10)
    assert out['sample_size'] == 10
    assert out['kronos_accuracy'] == 1.0
    assert out['baseline_accuracy'] == 0.0
    assert out['accuracy_delta'] == 1.0
    assert out['winner_by_accuracy'] == 'kronos'
    assert out['mcnemar_b'] == 10 and out['mcnemar_c'] == 0
    assert out['mcnemar_p_value'] == pytest.approx(2 * 0.5 ** 10)


def test_paired_comparisons_use_identical_observations():
    candles = mk(120)
    c = cfg(context_length=10, window_size=20)
    res = PredictionEvaluator(FakePredictor(), c, 'BTC/USDT', '1h').evaluate(candles)
    sc = res.report['statistical_comparison']
    n_rows = len(res.rows)
    for key in ('kronos_vs_persistence', 'kronos_vs_previous_direction'):
        err = sc['paired_close_error'][key]
        assert err['sample_size'] == n_rows
    # mk() candles have non-flat actual returns (> threshold), so direction
    # sample sizes equal the row count too.
    for key in ('kronos_vs_persistence', 'kronos_vs_previous_direction'):
        d = sc['directional'][key]
        assert d['sample_size'] == n_rows
    # identical timestamps are the alignment key: verify via report counts
    assert res.report['model_comparison']['prediction_count']['same_timestamps'] is True


def test_statistical_comparison_empty_safe():
    sc = build_statistical_comparison([], [], [], THRESHOLD)
    for key in ('kronos_vs_persistence', 'kronos_vs_previous_direction'):
        assert sc['paired_close_error'][key]['sample_size'] == 0
        assert sc['paired_close_error'][key]['mean_diff'] is None
        assert sc['directional'][key]['sample_size'] == 0
        assert sc['directional'][key]['kronos_accuracy'] is None


# --------------------------------------------------------------------------- #
# 5. Deterministic repeatability
# --------------------------------------------------------------------------- #
def test_windows_and_stats_deterministic():
    candles = mk(120)
    c = cfg(context_length=10, window_size=20)
    w1, _ = PredictionEvaluator(FakePredictor(), c, 'BTC/USDT', '1h').evaluate_windows(candles)
    w2, _ = PredictionEvaluator(FakePredictor(), c, 'BTC/USDT', '1h').evaluate_windows(candles)
    assert set(w1.keys()) == set(w2.keys())
    for name in w1:
        assert w1[name].report['metrics'] == w2[name].report['metrics']
        assert w1[name].report['model_comparison'] == w2[name].report['model_comparison']
        assert w1[name].report['statistical_comparison'] == \
            w2[name].report['statistical_comparison']


# --------------------------------------------------------------------------- #
# 6. Consolidated robustness report
# --------------------------------------------------------------------------- #
def test_run_robustness_consolidated_structure():
    candles_by_series = {
        ('BTC/USDT', '1h'): mk(120),
        ('ETH/USDT', '1h'): mk(120),
        ('BTC/USDT', '4h'): mk(120, step=14_400_000),
    }
    series = [('BTC/USDT', '1h'), ('ETH/USDT', '1h'), ('BTC/USDT', '4h')]
    report = run_robustness(FakePredictor(), cfg(window_size=20), series,
                            lambda s, tf: candles_by_series[(s, tf)])
    assert report['kind'] == 'phase4_robustness'
    assert len(report['series']) == 3
    for s in report['series']:
        assert set(s['windows'].keys()) == {'older', 'middle', 'recent'}
        assert s['window_info']['windows_produced'] == 3
        for name, wrep in s['windows'].items():
            assert wrep['window'] == name
            assert 'baseline_results' in wrep
            assert 'model_comparison' in wrep
            assert 'statistical_comparison' in wrep
        # summary counts sum to the number of windows
        for metric, summ in s['summary'].items():
            assert (summ['vs_persistence']['kronos']
                    + summ['vs_persistence']['baseline']
                    + summ['vs_persistence']['tie']
                    + summ['vs_persistence']['undefined']) == 3
    assert report['across_all_series']['total_series'] == 3
    assert report['across_all_series']['total_window_evaluations'] == 9


def test_summarize_windows_consistency():
    candles = mk(200)
    c = cfg(context_length=10, window_size=30)
    windows, _ = PredictionEvaluator(FakePredictor(), c, 'BTC/USDT', '1h').evaluate_windows(candles)
    summ = summarize_windows(windows)
    assert 'mae_close' in summ
    # FakePredictor beats persistence (lower error) but loses to prev-direction
    # (a near-perfect extrapolator on the rising mk() series).
    assert summ['mae_close']['vs_persistence']['kronos'] == 3
    assert summ['mae_close']['vs_previous_direction']['baseline'] == 3
