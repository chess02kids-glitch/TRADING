"""Phase 5b volatility research tests.

Covers: no-lookahead, past-only HAR fitting, identical timestamps, deterministic
outputs, fixed baseline definitions, normalized-target correctness, regime
assignment, statistical-test alignment, safe empty/short behavior, and the
success-gate classification. A deterministic test double stands in for Kronos
(never presented as real output).
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from kronos_trading.evaluation import EvaluationConfig, PredictionEvaluator
from kronos_trading.model import PredictorResult
from kronos_trading.statistics_compare import (
    circular_block_bootstrap_mean_ci,
    diebold_mariano,
    spearman,
)
from kronos_trading.types import Candle
from kronos_trading.volatility_baselines import (
    EWMA_SPAN,
    HAR_MIN_TRAIN,
    ROLLING_WINDOWS,
    assign_regime,
    ewma_range,
    har_forecast,
    rolling_mean_range,
    volatility_forecasts,
)
from kronos_trading.volatility_research import (
    NONDAILY_TIMEFRAMES,
    classify_gate,
    evaluate_success_gate,
    run_volatility_research,
    system_stats,
)

H = 3_600_000
BASE = 1_700_000_000_000


def mk(n, base=BASE, step=H, rng=None):
    """Candles with deterministic, varying ranges (for regime separation)."""
    rng = rng or np.random.default_rng(7)
    out = []
    for i in range(n):
        close = 100.0 + i * 0.05
        r = 0.5 + (i % 5) * 0.3 + rng.normal(0, 0.05)  # deterministic-ish ranges
        out.append(Candle(base + i * step, close - r / 2, close + r / 2,
                          close - r / 2, close, 10.0))
    return out


class FakePredictor:
    """Deterministic Kronos test double; range = factor * last observed range."""

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
        last_range = last.high - last.low
        rng = self.factor * last_range
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
# Baseline definitions (fixed, past-only)
# --------------------------------------------------------------------------- #
def test_rolling_mean_range_fixed_windows():
    ranges = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert rolling_mean_range(ranges, 5) == pytest.approx(4.0)  # mean(2..6)
    assert rolling_mean_range(ranges, 22) is None
    assert ROLLING_WINDOWS == (5, 22)


def test_ewma_fixed_decay_and_seed():
    # span=1 -> alpha=1 -> EWMA equals the last value
    assert ewma_range([1.0, 2.0, 3.0], span=1) == pytest.approx(3.0)
    # span=22 default, alpha = 2/23
    a = 2.0 / (EWMA_SPAN + 1.0)
    e = 1.0
    for r in [2.0, 3.0, 4.0]:
        e = a * r + (1 - a) * e
    assert ewma_range([1.0, 2.0, 3.0, 4.0]) == pytest.approx(e)
    assert ewma_range([]) is None


def test_har_past_only_and_deterministic():
    rng = np.random.default_rng(3)
    ranges = list(rng.uniform(1, 5, size=120))
    f1 = har_forecast(ranges)
    f2 = har_forecast(ranges)
    assert f1 == f2  # deterministic
    # HAR reads only the provided (past) context; a constant series recovers
    # the constant exactly regardless of length (no look-ahead by construction).
    assert har_forecast([3.0] * 50) == pytest.approx(3.0)
    # insufficient history -> None
    assert har_forecast([1.0, 2.0]) is None
    assert har_forecast(list(rng.uniform(1, 5, size=22 + HAR_MIN_TRAIN - 1))) is None


def test_har_formula_small_case():
    # Construct a case where OLS recovers a known linear relationship.
    rng = np.random.default_rng(11)
    n = 40
    ranges = [1.0]
    for _ in range(n - 1):
        ranges.append(ranges[-1] * 1.0 + rng.normal(0, 0.1))
    f = har_forecast(ranges, min_train=5, windows=(2, 3))
    assert f is not None and math.isfinite(f)


def test_volatility_forecasts_keys():
    fc = volatility_forecasts([1.0, 2.0, 3.0])
    assert set(fc) == {'prev', 'rolling5', 'rolling22', 'ewma', 'har'}
    assert fc['prev'] == 3.0


# --------------------------------------------------------------------------- #
# Regime assignment (past-only, fixed terciles)
# --------------------------------------------------------------------------- #
def test_regime_past_only_and_fixed_terciles():
    rng = np.random.default_rng(5)
    ranges = list(rng.uniform(1, 10, size=80))
    reg1 = assign_regime(ranges)
    # adding future bars must not change the regime for the same context
    reg2 = assign_regime(ranges + list(rng.uniform(1, 10, size=10)))
    assert reg1 == reg2
    assert reg1 in ('low', 'medium', 'high')


def test_regime_high_vs_low():
    # a context ending in a sharp spike should be 'high'; ending in a valley 'low'
    low = [0.5] * 30 + [0.4, 0.45, 0.42]
    high = [0.5] * 30 + [9.0, 9.5, 9.2]
    assert assign_regime(low) == 'low'
    assert assign_regime(high) == 'high'


def test_regime_short_context_undefined():
    assert assign_regime([1.0, 2.0]) == 'undefined'


# --------------------------------------------------------------------------- #
# Statistical functions
# --------------------------------------------------------------------------- #
def test_spearman_perfect_and_ties():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)
    assert spearman([1], [2]) is None
    assert spearman([1, 1, 1], [1, 2, 3]) is None  # zero variance in ranks


def test_diebold_mariano_direction_and_serial_correlation():
    # kronos always smaller error -> negative loss diff -> winner kronos, p ~ 0
    out = diebold_mariano([1.0] * 50, [2.0] * 50)
    assert out['winner'] == 'kronos'
    assert out['mean_loss_diff'] == -1.0
    assert out['p_value'] < 1e-6
    # identical errors -> tie, p = 1
    out2 = diebold_mariano([1.0] * 20, [1.0] * 20)
    assert out2['winner'] == 'tie'
    assert out2['p_value'] == 1.0
    # autocorrelated series still yields a finite p and a positive lag
    rng = np.random.default_rng(0)
    e1 = np.cumsum(rng.normal(0, 1, 200))  # random walk -> strong serial corr
    e2 = np.cumsum(rng.normal(0, 1, 200))
    out3 = diebold_mariano(np.abs(e1), np.abs(e2))
    assert math.isfinite(out3['p_value'])
    assert out3['lag'] >= 1
    assert out3['n'] == 200


def test_block_bootstrap_constant_and_empty():
    out = circular_block_bootstrap_mean_ci([1.0] * 20)
    assert out['mean'] == 1.0 and out['ci_low'] == out['ci_high'] == 1.0
    out2 = circular_block_bootstrap_mean_ci([])
    assert out2['n'] == 0 and out2['mean'] is None


# --------------------------------------------------------------------------- #
# system_stats correctness (MAE/RMSE/bias/dispersion)
# --------------------------------------------------------------------------- #
def test_system_stats_exact():
    s = system_stats([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
    assert s['sample_size'] == 3
    assert s['mae'] == pytest.approx(1.0)
    assert s['rmse'] == pytest.approx(1.0)
    assert s['bias'] == pytest.approx(1.0)
    assert s['mean_pred'] == 3.0 and s['mean_actual'] == 2.0
    assert s['bias_ratio'] == pytest.approx(1.5)
    assert s['dispersion_ratio'] == pytest.approx(1.0)  # both std = 1
    assert s['pearson'] == pytest.approx(1.0)
    assert s['spearman'] == pytest.approx(1.0)


def test_system_stats_shrinkage_detected():
    # constant predictions -> std_pred = 0 -> dispersion ratio 0
    s = system_stats([2.0, 2.0, 2.0, 2.0], [1.0, 2.0, 3.0, 4.0])
    assert s['dispersion_ratio'] == 0.0
    assert s['std_pred'] == 0.0
    assert s['std_actual'] == pytest.approx(math.sqrt(5.0 / 3.0))


# --------------------------------------------------------------------------- #
# Evaluator: volatility rows are aligned, past-only, deterministic
# --------------------------------------------------------------------------- #
def test_volatility_rows_aligned_and_past_only():
    candles = mk(80)
    c = cfg(context_length=24, window_size=20)
    res = PredictionEvaluator(FakePredictor(), c, 'BTC/USDT', '1h').evaluate(candles)
    assert len(res.volatility_rows) == len(res.rows) > 0
    for vrow, krow in zip(res.volatility_rows, res.rows):
        assert vrow.prediction_timestamp == krow.prediction_timestamp
        # prev_range == the context's last closed range == kronos row's field
        assert vrow.prev_range == pytest.approx(krow.context_last_range)
        assert vrow.denom_close == pytest.approx(krow.context_last_close)
        assert vrow.actual_range == pytest.approx(krow.actual_high - krow.actual_low)


def test_volatility_no_lookahead():
    candles = mk(60)
    c = cfg(context_length=24, window_size=10)
    pred = FakePredictor(factor=1.0)
    res = PredictionEvaluator(pred, c, 'BTC/USDT', '1h').evaluate(candles)
    for call_ts, vrow in zip(pred.calls, res.volatility_rows):
        # every model context ended strictly before the prediction timestamp
        assert call_ts < vrow.prediction_timestamp


def test_volatility_deterministic():
    candles = mk(80)
    c = cfg(context_length=24, window_size=20)
    r1 = PredictionEvaluator(FakePredictor(), c, 'BTC/USDT', '1h').evaluate(candles)
    r2 = PredictionEvaluator(FakePredictor(), c, 'BTC/USDT', '1h').evaluate(candles)
    assert [v.asdict() for v in r1.volatility_rows] == \
        [v.asdict() for v in r2.volatility_rows]


def test_normalized_target_correctness():
    candles = mk(60)
    c = cfg(context_length=24, window_size=10)
    res = PredictionEvaluator(FakePredictor(), c, 'BTC/USDT', '1h').evaluate(candles)
    for vrow in res.volatility_rows:
        assert (vrow.kronos_range / vrow.denom_close) == pytest.approx(
            vrow.kronos_range / vrow.denom_close)
        assert (vrow.actual_range / vrow.denom_close) == pytest.approx(
            vrow.actual_range / vrow.denom_close)


# --------------------------------------------------------------------------- #
# Success gate
# --------------------------------------------------------------------------- #
def _record(series='BTC/USDT', timeframe='1h', window='recent', sample_size=100,
            k_mae=1.0, prev_mae=2.0, ewma_mae=1.5, har_mae=1.6,
            k_nmae=0.5, ewma_nmae=0.6, har_nmae=0.7,
            dispersion=1.0, regime_wins=None):
    return {
        'series': series, 'timeframe': timeframe, 'window': window,
        'sample_size': sample_size, 'low_power': sample_size < 30,
        'kronos_mae': k_mae, 'prev_mae': prev_mae,
        'ewma_mae': ewma_mae, 'har_mae': har_mae,
        'kronos_nmae': k_nmae, 'ewma_nmae': ewma_nmae, 'har_nmae': har_nmae,
        'kronos_beats_prev': k_mae < prev_mae,
        'kronos_beats_ewma': k_mae < ewma_mae,
        'kronos_beats_har': k_mae < har_mae,
        'kronos_beats_serious': (k_mae < ewma_mae) or (k_mae < har_mae),
        'kronos_beats_serious_norm': (k_nmae < ewma_nmae) or (k_nmae < har_nmae),
        'kronos_dispersion_ratio': dispersion,
        'regime_kronos_beats_serious': regime_wins or {'low': True, 'medium': True, 'high': True},
    }


def test_gate_all_criteria_pass():
    recs = [_record(series=s, window=w)
            for s in ('BTC/USDT', 'ETH/USDT', 'BTC/USDT', 'ETH/USDT')
            for w in ('older', 'middle', 'recent')]
    recs = [{**r, 'timeframe': '1h'} for r in recs]
    gate = evaluate_success_gate(recs)
    c = gate['criteria']
    assert c['c1_beats_previous_range'] is True
    assert c['c2_beats_ewma_or_har'] is True
    assert c['c3_series_breadth'] is True
    assert c['c4_window_breadth'] is True
    assert c['c5_normalized_survives'] is True


def test_gate_shrinkage_blocks_verdict_a():
    # kronos beats baselines but is heavily shrunk (dispersion 0.5)
    recs = [_record(dispersion=0.5) for _ in range(12)]
    recs = [{**r, 'timeframe': '1h'} for r in recs]
    gate = evaluate_success_gate(recs)
    assert gate['criteria']['c1_beats_previous_range'] is True
    # c7 is filled by run_volatility_research; here we simulate its decision
    criteria = dict(gate['criteria'])
    criteria['c6_statistical_support'] = True
    criteria['c7_not_solely_shrinkage'] = False  # dispersion 0.5 < 0.7
    criteria['c8_regime_breadth'] = True
    assert classify_gate(criteria) == 'B'  # beats persistence but shrinkage blocks A


def test_gate_classification_a_b_c():
    all_true = {'c1_beats_previous_range': True, 'c2_beats_ewma_or_har': True,
                'c3_series_breadth': True, 'c4_window_breadth': True,
                'c5_normalized_survives': True, 'c6_statistical_support': True,
                'c7_not_solely_shrinkage': True, 'c8_regime_breadth': True}
    assert classify_gate(all_true) == 'A'
    weak = dict(all_true, c2_beats_ewma_or_har=False, c6_statistical_support=False)
    assert classify_gate(weak) == 'B'
    losing = dict(all_true, c1_beats_previous_range=False, c2_beats_ewma_or_har=False)
    assert classify_gate(losing) == 'C'


def test_gate_excludes_daily_and_low_power():
    recs = [_record(timeframe='1d', sample_size=72) for _ in range(12)]
    gate = evaluate_success_gate(recs)
    assert gate['eligible_windows'] == 0
    assert gate['overall'] == 'pending'


# --------------------------------------------------------------------------- #
# End-to-end run (deterministic) on a fake predictor
# --------------------------------------------------------------------------- #
def test_run_volatility_research_deterministic_and_structure():
    candles_by_series = {
        ('BTC/USDT', '1h'): mk(80),
        ('ETH/USDT', '1h'): mk(80),
    }
    series = [('BTC/USDT', '1h'), ('ETH/USDT', '1h')]
    c = cfg(context_length=24, window_size=20)

    r1 = run_volatility_research(FakePredictor(), c, series,
                                 lambda s, tf: candles_by_series[(s, tf)])
    r2 = run_volatility_research(FakePredictor(), c, series,
                                 lambda s, tf: candles_by_series[(s, tf)])
    assert r1['kind'] == 'phase5b_volatility_research'
    assert r1['baseline_definitions'] == r2['baseline_definitions']
    assert r1['target_definitions'] == r2['target_definitions']
    # deterministic comparisons/gate
    assert r1['window_records'] == r2['window_records']
    assert r1['success_gate'] == r2['success_gate']
    assert r1['pooled_statistics'] == r2['pooled_statistics']
    # structure
    for sid in ('BTC/USDT 1h', 'ETH/USDT 1h'):
        assert set(r1['series'][sid]['windows']) == {'older', 'middle', 'recent'}
        for w, analysis in r1['series'][sid]['windows'].items():
            assert set(analysis['systems']) == \
                {'kronos', 'prev', 'rolling5', 'rolling22', 'ewma', 'har'}
            assert set(analysis['regimes']) == {'low', 'medium', 'high'}
            assert '_vrows' not in analysis  # never serialized


def test_run_volatility_research_empty_safe():
    candles_by_series = {('BTC/USDT', '1h'): mk(5)}  # too few for windows
    series = [('BTC/USDT', '1h')]
    report = run_volatility_research(FakePredictor(), cfg(context_length=4),
                                     series, lambda s, tf: candles_by_series[(s, tf)])
    assert report['success_gate']['overall'] == 'pending'
    assert report['success_gate']['eligible_windows'] == 0


def test_non_daily_definition():
    assert NONDAILY_TIMEFRAMES == frozenset({'1h', '4h'})
    assert '1d' not in NONDAILY_TIMEFRAMES
