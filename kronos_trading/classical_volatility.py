"""Phase 5c - classical volatility benchmark (HAR as primary model).

Phase 5b concluded B (weak/ambiguous): Kronos beats the weak previous-range
baseline but not the serious volatility baselines, and its MAE advantage is
partly forecast shrinkage. The highest-value question now is NOT "does Kronos
win" but:

    Is the crypto candle-range forecasting problem itself predictable, and does
    the classical HAR model already capture that structure?

This module reuses the existing per-step ``VolatilityRow`` machinery and the
Phase 5b ``_analyze_window`` statistics (no duplication). It adds:

* the classical benchmark matrix: previous-range / rolling-5 / rolling-22 /
  EWMA(span=22) / HAR, evaluated among themselves (HAR as primary);
* improvement percentages relative to previous-range;
* HAR shrinkage/adequacy diagnostics (dispersion ratio, bias ratio, and a
  direct "does HAR distinguish low/med/high regimes" test);
* a pre-registered 8-criterion A/B/C decision for *classical* predictability;
* a Kronos-vs-HAR incremental-value comparison on the same windows.

All constants are fixed a priori. Nothing is tuned, and Kronos is treated as a
challenger, not the foundation of the design.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .evaluation import EvaluationConfig, PredictionEvaluator, VolatilityRow
from .statistics_compare import (circular_block_bootstrap_mean_ci,
                                 diebold_mariano, wilcoxon_signed_rank)
from .volatility_baselines import (EWMA_SPAN, HAR_MIN_TRAIN, ROLLING_WINDOWS,
                                   assign_regime, ewma_range, har_forecast,
                                   rolling_mean_range)
from .volatility_research import (_analyze_window, _fmean, _mae, _std,
                                  _winner_lower, NONDAILY_TIMEFRAMES,
                                  MIN_SAMPLES)

# --- Fixed a priori constants (do NOT tune against results) -----------------
CLASSICAL_KEYS = ('prev', 'rolling5', 'rolling22', 'ewma', 'har')
PRIMARY_MODEL = 'har'
# HAR is compared against these (4 primary comparisons -> Bonferroni alpha/4).
HAR_COMPARISONS = ('prev', 'ewma', 'rolling5', 'rolling22')
GATE_MAJORITY = 0.5
BONFERRONI_N_TESTS = len(HAR_COMPARISONS)
GATE_ALPHA = 0.05 / BONFERRONI_N_TESTS  # 0.0125
GATE_P_VALUE = GATE_ALPHA
# HAR regime tracking: HAR's forecast spread across regimes must reach at least
# this fraction of the actual regime spread (a pure shrinker fails this).
TRACKING_MIN_SPREAD_RATIO = 0.1
TRACKING_MIN_PER_REGIME = 5

VERDICT_MEANING = {
    'A': 'classical volatility predictability established',
    'B': 'weak / ambiguous classical volatility predictability',
    'C': 'no robust classical volatility predictability',
    'pending': 'not enough eligible evidence to classify',
}


def classify_classical_gate(criteria: Dict[str, Any]) -> str:
    """Map the pre-registered criteria to the scientific verdict (A/B/C)."""
    if all(v is True for v in criteria.values()):
        return 'A'
    if criteria.get('c1_har_beats_prev') is True:
        return 'B'
    return 'C'


# --------------------------------------------------------------------------- #
# Improvement %, classical pairwise tests, regime tracking
# --------------------------------------------------------------------------- #
def _classical_winner(a: Optional[float], b: Optional[float]) -> Optional[str]:
    """Winner for a metric where lower is better, between two classical models.

    ``a`` is the primary model (HAR) and ``b`` the comparison model; returns
    ``'har'`` / ``'baseline'`` / ``'tie'`` / ``None``.
    """
    if a is None or b is None:
        return None
    if a < b:
        return 'har'
    if a > b:
        return 'baseline'
    return 'tie'


def improvement_pct(base_mae: Optional[float], model_mae: Optional[float]) -> Optional[float]:
    """Improvement of ``model`` over the base model in percent (positive=better)."""
    if base_mae is None or model_mae is None or base_mae <= 1e-12:
        return None
    return (base_mae - model_mae) / base_mae * 100.0


def classical_pairwise(vrows: List[VolatilityRow], key_a: str, key_b: str,
                       normalized: bool = False) -> Dict[str, Any]:
    """Paired forecast-error comparison between two classical baselines.

    Uses identical prediction timestamps (same ``VolatilityRow`` list); errors
    are absolute range errors (raw) or normalized range errors.
    """
    def err(r, key):
        f = getattr(r, key + '_range')
        if f is None:
            return math.nan
        if normalized:
            return abs(f / r.denom_close - r.actual_range / r.denom_close)
        return abs(f - r.actual_range)

    ea = [err(r, key_a) for r in vrows]
    eb = [err(r, key_b) for r in vrows]
    diff = [a - b for a, b in zip(ea, eb)]
    return {
        'dm': diebold_mariano(ea, eb),
        'bootstrap_mean_diff_ci': circular_block_bootstrap_mean_ci(diff),
        'wilcoxon_p_value': wilcoxon_signed_rank(diff)['p_value'],
        'wilcoxon_n_nonzero': wilcoxon_signed_rank(diff)['n_nonzero'],
    }


def har_regime_tracking(vrows: List[VolatilityRow],
                        min_per_regime: int = TRACKING_MIN_PER_REGIME,
                        min_spread_ratio: float = TRACKING_MIN_SPREAD_RATIO) -> Dict[str, Any]:
    """Does HAR distinguish volatility regimes (vs shrinking toward the mean)?

    Groups rows by their past-only regime label, computes the mean HAR forecast
    and mean actual range per regime, and checks that (a) the forecast means
    are monotone in the actual means and (b) the forecast spread is at least
    ``min_spread_ratio`` of the actual spread. A pure shrinker has ~zero spread
    and fails (b).
    """
    groups: Dict[str, List[Tuple[float, float]]] = {'low': [], 'medium': [], 'high': []}
    for r in vrows:
        if r.regime in groups and r.har_range is not None:
            groups[r.regime].append((r.har_range, r.actual_range))

    means: Dict[str, Dict[str, float]] = {}
    for reg, pairs in groups.items():
        if len(pairs) >= min_per_regime:
            means[reg] = {
                'har_mean': _fmean([f for f, _ in pairs]),
                'actual_mean': _fmean([a for _, a in pairs]),
                'n': len(pairs),
            }
    if len(means) < 2:
        return {'regime_means': means, 'n_regimes': len(means),
                'monotonic': None, 'spread_ratio': None, 'tracks': None}

    ordered = sorted(means.items(), key=lambda kv: kv[1]['actual_mean'])
    fs = [v['har_mean'] for _, v in ordered]
    actuals = [v['actual_mean'] for _, v in ordered]
    monotonic = all(fs[i] <= fs[i + 1] + 1e-12 for i in range(len(fs) - 1))
    actual_spread = actuals[-1] - actuals[0]
    har_spread = fs[-1] - fs[0]
    spread_ratio = (har_spread / actual_spread) if actual_spread > 1e-12 else None
    tracks = bool(monotonic and spread_ratio is not None
                  and spread_ratio >= min_spread_ratio)
    return {'regime_means': means, 'n_regimes': len(means),
            'monotonic': bool(monotonic), 'spread_ratio': spread_ratio,
            'tracks': tracks}


def _analyze_classical_window(vrows: List[VolatilityRow]) -> Dict[str, Any]:
    base = _analyze_window(vrows)
    systems = base['systems']
    systems_norm = base['systems_normalized']

    # Improvement % vs previous-range (raw + normalized MAE).
    improvement = {}
    prev_mae = systems['prev']['mae']
    prev_nmae = systems_norm['prev']['mae']
    for key in CLASSICAL_KEYS:
        improvement[key] = {
            'mae_improvement_pct_vs_prev': improvement_pct(prev_mae, systems[key]['mae']),
            'nmae_improvement_pct_vs_prev': improvement_pct(prev_nmae, systems_norm[key]['mae']),
        }

    # HAR vs the other classical baselines (primary comparisons).
    classical_comparisons = {}
    for key in HAR_COMPARISONS:
        classical_comparisons['har_vs_%s' % key] = {
            'mae_delta': systems['har']['mae'] - systems[key]['mae']
            if (systems['har']['mae'] is not None and systems[key]['mae'] is not None)
            else None,
            'mae_winner': _classical_winner(systems['har']['mae'], systems[key]['mae']),
            'norm_mae_delta': (systems_norm['har']['mae'] - systems_norm[key]['mae'])
            if (systems_norm['har']['mae'] is not None and systems_norm[key]['mae'] is not None)
            else None,
            'raw': classical_pairwise(vrows, 'har', key, normalized=False),
            'normalized': classical_pairwise(vrows, 'har', key, normalized=True),
        }

    tracking = har_regime_tracking(vrows)

    return {
        'sample_size': base['sample_size'],
        'low_power': base['sample_size'] < MIN_SAMPLES,
        'regime_counts': base['regime_counts'],
        'systems': systems,
        'systems_normalized': systems_norm,
        'improvement_vs_prev_pct': improvement,
        'classical_comparisons': classical_comparisons,
        'har_regime_tracking': tracking,
        'har_shrinkage': {
            'dispersion_ratio': systems['har']['dispersion_ratio'],
            'bias_ratio': systems['har']['bias_ratio'],
            'std_pred': systems['har']['std_pred'],
            'std_actual': systems['har']['std_actual'],
            'mean_pred': systems['har']['mean_pred'],
            'mean_actual': systems['har']['mean_actual'],
        },
        'regimes': base['regimes'],
        'kronos_vs_har': base['comparisons'].get('har'),
    }


# --------------------------------------------------------------------------- #
# Decision gate (classical predictability)
# --------------------------------------------------------------------------- #
def _majority(count: int, total: int) -> bool:
    return count > GATE_MAJORITY * total


def evaluate_classical_gate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pre-registered 8-criterion gate for *classical* volatility predictability."""
    eligible = [r for r in records
                if not r['low_power'] and r['timeframe'] in NONDAILY_TIMEFRAMES]
    n = len(eligible)
    gate: Dict[str, Any] = {
        'eligible_windows': n,
        'definition': {
            'c1': 'HAR beats previous-range (raw MAE) in > half of windows',
            'c2': 'HAR beats EWMA (raw MAE) in > half of windows',
            'c3': 'HAR beats previous-range on >=2 of 4 series',
            'c4': 'HAR beats previous-range in >=2 of 3 windows',
            'c5': 'HAR beats previous-range on normalized MAE in > half of windows',
            'c6': 'pooled DM (HAR vs previous-range) p < 0.0125 (Bonferroni)',
            'c7': 'HAR improvement not purely shrinkage (regime tracking in > half)',
            'c8': 'HAR beats previous-range across >1 regime',
        },
        'criteria': {},
    }
    if n == 0:
        gate['overall'] = 'pending'
        gate['verdict'] = 'pending'
        gate['verdict_meaning'] = VERDICT_MEANING['pending']
        gate['note'] = 'no eligible windows'
        return gate

    c1 = _majority(sum(r['har_beats_prev'] for r in eligible), n)
    c2 = _majority(sum(r['har_beats_ewma'] for r in eligible), n)
    c5 = _majority(sum(r['har_beats_prev_norm'] for r in eligible), n)
    c7 = _majority(sum(r['har_tracks'] for r in eligible), n)

    series_wins: Dict[str, List[bool]] = {}
    for r in eligible:
        series_wins.setdefault(r['series'], []).append(r['har_beats_prev'])
    c3 = sum(1 for wins in series_wins.values() if sum(wins) >= 2) >= 2

    window_wins: Dict[str, List[bool]] = {}
    for r in eligible:
        window_wins.setdefault(r['window'], []).append(r['har_beats_prev'])
    c4 = sum(1 for wins in window_wins.values() if sum(wins) >= 2) >= 2

    # c6 and c8 are filled by the caller (pooled DM / pooled regime results).
    gate['criteria'] = {
        'c1_har_beats_prev': bool(c1),
        'c2_har_beats_ewma': bool(c2),
        'c3_series_breadth': bool(c3),
        'c4_window_breadth': bool(c4),
        'c5_normalized_survives': bool(c5),
        'c6_statistical_support': None,
        'c7_not_solely_shrinkage': bool(c7),
        'c8_regime_breadth': None,
    }
    return gate


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def _window_record(series: str, timeframe: str, window: str,
                   analysis: Dict[str, Any]) -> Dict[str, Any]:
    s = analysis['systems']
    sn = analysis['systems_normalized']
    return {
        'series': series, 'timeframe': timeframe, 'window': window,
        'sample_size': analysis['sample_size'],
        'low_power': analysis['low_power'],
        'prev_mae': s['prev']['mae'], 'rolling5_mae': s['rolling5']['mae'],
        'rolling22_mae': s['rolling22']['mae'], 'ewma_mae': s['ewma']['mae'],
        'har_mae': s['har']['mae'], 'kronos_mae': s['kronos']['mae'],
        'har_beats_prev': _classical_winner(s['har']['mae'], s['prev']['mae']) == 'har',
        'har_beats_ewma': _classical_winner(s['har']['mae'], s['ewma']['mae']) == 'har',
        'har_beats_prev_norm': _classical_winner(sn['har']['mae'], sn['prev']['mae']) == 'har',
        'har_beats_kronos': _classical_winner(s['har']['mae'], s['kronos']['mae']) == 'har',
        'har_dispersion_ratio': s['har']['dispersion_ratio'],
        'har_bias_ratio': s['har']['bias_ratio'],
        'har_tracks': analysis['har_regime_tracking']['tracks'] is True,
        'har_vs_prev_mae_pct': analysis['improvement_vs_prev_pct']['har']['mae_improvement_pct_vs_prev'],
        'har_vs_prev_nmae_pct': analysis['improvement_vs_prev_pct']['har']['nmae_improvement_pct_vs_prev'],
        'kronos_vs_har_mae_winner': analysis['kronos_vs_har']['mae_winner'],
    }


def run_classical_volatility_benchmark(predictor, config: EvaluationConfig,
                                       series: List[Tuple[str, str]],
                                       load_candles) -> Dict[str, Any]:
    """Run the classical volatility benchmark and evaluate the decision gate."""
    config = EvaluationConfig(**{**config.asdict()})  # copy (horizon=1 for range)
    series_output: Dict[str, Any] = {}
    window_records: List[Dict[str, Any]] = []
    gate_analyses: List[Tuple[str, str, str, Dict[str, Any], List[VolatilityRow]]] = []

    for symbol, timeframe in series:
        evaluator = PredictionEvaluator(predictor, config, symbol, timeframe)
        windows, window_info = evaluator.evaluate_windows(
            load_candles(symbol, timeframe))
        per_window: Dict[str, Any] = {}
        for name, res in windows.items():
            vrows = res.volatility_rows
            analysis = _analyze_classical_window(vrows)
            per_window[name] = analysis
            rec = _window_record(symbol, timeframe, name, analysis)
            window_records.append(rec)
            if not rec['low_power'] and timeframe in NONDAILY_TIMEFRAMES:
                gate_analyses.append((symbol, timeframe, name, analysis, vrows))
        series_output['%s %s' % (symbol, timeframe)] = {
            'window_info': window_info,
            'windows': per_window,
        }

    # Pooled (supplementary) statistics: HAR vs each comparison baseline.
    pooled = {}
    for key in HAR_COMPARISONS:
        raw_a, raw_b, norm_a, norm_b = [], [], [], []
        for _, _, _, analysis, vrows in gate_analyses:
            for r in vrows:
                f_har = r.har_range
                f_key = getattr(r, key + '_range')
                if f_har is None or f_key is None:
                    continue
                raw_a.append(abs(f_har - r.actual_range))
                raw_b.append(abs(f_key - r.actual_range))
                norm_a.append(abs(f_har / r.denom_close - r.actual_range / r.denom_close))
                norm_b.append(abs(f_key / r.denom_close - r.actual_range / r.denom_close))
        pooled[key] = {
            'raw': diebold_mariano(raw_a, raw_b),
            'normalized': diebold_mariano(norm_a, norm_b),
            'raw_bootstrap': circular_block_bootstrap_mean_ci([a - b for a, b in zip(raw_a, raw_b)]),
            'raw_wilcoxon_p': wilcoxon_signed_rank([a - b for a, b in zip(raw_a, raw_b)])['p_value'],
        }

    # Pooled regime analysis (HAR vs previous-range per regime).
    regime_pool = {'low': {'har': [], 'actual': [], 'prev': []},
                   'medium': {'har': [], 'actual': [], 'prev': []},
                   'high': {'har': [], 'actual': [], 'prev': []}}
    for _, _, _, analysis, vrows in gate_analyses:
        for r in vrows:
            if r.regime not in regime_pool:
                continue
            d = regime_pool[r.regime]
            if r.har_range is not None:
                d['har'].append(r.har_range)
                d['actual'].append(r.actual_range)
            if r.prev_range is not None:
                d['prev'].append(r.prev_range)
    regime_result = {}
    regime_wins = 0
    for reg, d in regime_pool.items():
        har_mae = _mae(d['har'], d['actual'])
        prev_mae = _mae(d['prev'], d['actual'])
        har_beats_prev = (har_mae is not None and prev_mae is not None
                          and har_mae < prev_mae)
        regime_result[reg] = {'har_mae': har_mae, 'prev_mae': prev_mae,
                              'har_beats_prev': har_beats_prev,
                              'n': len(d['actual'])}
        regime_wins += int(har_beats_prev)
    c8 = regime_wins >= 2

    # Pooled HAR adequacy (regime tracking + dispersion).
    pooled_tracking = har_regime_tracking(
        [r for _, _, _, _, vrows in gate_analyses for r in vrows])
    all_har, all_actual = [], []
    for _, _, _, analysis, vrows in gate_analyses:
        for r in vrows:
            if r.har_range is not None:
                all_har.append(r.har_range)
                all_actual.append(r.actual_range)
    std_pred = _std(all_har) if len(all_har) > 1 else None
    std_act = _std(all_actual) if len(all_actual) > 1 else None
    pooled_dispersion = (std_pred / std_act) if (std_act is not None and std_act > 1e-12) else None
    pooled_bias = (_fmean(all_har) / _fmean(all_actual)) \
        if (_fmean(all_har) is not None and abs(_fmean(all_actual)) > 1e-12) else None

    # c6: pooled DM (HAR vs prev) significant under Bonferroni.
    dm_har_prev = pooled['prev']['raw']
    c6 = (dm_har_prev['p_value'] is not None
          and dm_har_prev['p_value'] < GATE_P_VALUE
          and dm_har_prev['winner'] == 'har')

    # Gate.
    gate = evaluate_classical_gate(window_records)
    if gate.get('overall') != 'pending':
        gate['criteria']['c6_statistical_support'] = bool(c6)
        gate['criteria']['c8_regime_breadth'] = bool(c8)
        gate['overall'] = 'pass' if all(v is True for v in gate['criteria'].values()) else 'fail'
        gate['verdict'] = classify_classical_gate(gate['criteria'])
        gate['verdict_meaning'] = VERDICT_MEANING[gate['verdict']]

    # Kronos incremental value vs HAR (challenger question).
    kronos_vs_har_wins = sum(1 for r in window_records
                             if r['kronos_vs_har_mae_winner'] == 'kronos')
    eligible_kh = [r for r in window_records
                   if not r['low_power'] and r['timeframe'] in NONDAILY_TIMEFRAMES]
    kh_norm_a, kh_norm_b = [], []
    for _, _, _, analysis, vrows in gate_analyses:
        for r in vrows:
            if r.har_range is None:
                continue
            kh_norm_a.append(abs(r.kronos_range / r.denom_close - r.actual_range / r.denom_close))
            kh_norm_b.append(abs(r.har_range / r.denom_close - r.actual_range / r.denom_close))
    kronos_dm_vs_har = diebold_mariano(kh_norm_a, kh_norm_b)
    kronos_wins_majority = (kronos_vs_har_wins > GATE_MAJORITY * len(eligible_kh)) \
        if eligible_kh else None
    kronos_dm_favors = (kronos_dm_vs_har['winner'] == 'kronos'
                        and kronos_dm_vs_har['p_value'] is not None
                        and kronos_dm_vs_har['p_value'] < 0.05)
    # Conservative: Kronos "adds incremental value" only if it wins a majority
    # of eligible windows AND the pooled DM is significant in its favour.
    kronos_adds = bool(kronos_wins_majority and kronos_dm_favors) \
        if eligible_kh else None

    return {
        'kind': 'classical_volatility_benchmark',
        'configuration': config.asdict(),
        'classical_baselines': {
            'previous_range': 'range_{t-1}',
            'rolling5': 'mean(range_{t-5:t})',
            'rolling22': 'mean(range_{t-22:t})',
            'ewma': 'EWMA of ranges, span=22 (alpha=2/23), seeded on first range',
            'har': 'beta0 + beta1*range_{t-1} + beta2*mean5 + beta3*mean22, '
                   'expanding past-only OLS (min %d rows, refit per step)' % HAR_MIN_TRAIN,
        },
        'targets': {
            'raw_range': 'high_future - low_future',
            'normalized_range': '(high_future - low_future) / close_current',
        },
        'oos_methodology': {
            'windows': 'older / middle / recent (fixed, chronological, non-overlapping)',
            'series': 'BTC/USDT & ETH/USDT at 1h/4h/1d (1d supplementary only)',
            'no_lookahead': 'all coefficients/features estimated strictly from '
                            'closed candles before each prediction',
            'no_test_fitting': 'no evaluation-window fitting; HAR refits on '
                               'expanding past-only data per step',
            'no_future_normalization': 'normalized target divides by close_current '
                                       '(past) only',
            'no_regime_future': 'regime thresholds are expanding past-only terciles',
        },
        'statistical_methodology': {
            'dm': 'two-sided Diebold-Mariano, Newey-West HAC (serial correlation aware)',
            'bootstrap': 'circular block bootstrap 95% CI on paired error differences',
            'wilcoxon': 'Wilcoxon signed-rank (nonparametric robustness)',
            'multiple_testing': '4 primary HAR comparisons -> Bonferroni alpha=0.0125',
            'pooling': 'pooled statistics are supplementary only (not primary evidence)',
        },
        'series': series_output,
        'window_records': window_records,
        'pooled_statistics': pooled,
        'regime_pooled': regime_result,
        'har_adequacy': {
            'pooled_regime_tracking': pooled_tracking,
            'pooled_dispersion_ratio': pooled_dispersion,
            'pooled_bias_ratio': pooled_bias,
            'interpretation': ('a tracking model shows a positive forecast spread '
                               'across regimes; a pure shrinker has ~zero spread and '
                               'fails the tracking test'),
        },
        'kronos_vs_best_classical': {
            'best_classical': 'har',
            'kronos_mae_wins_over_har': kronos_vs_har_wins,
            'eligible_windows': len(eligible_kh) if eligible_kh is not None else 0,
            'kronos_wins_majority_of_windows': kronos_wins_majority,
            'pooled_dm_kronos_vs_har_normalized': kronos_dm_vs_har,
            'kronos_dm_favors': kronos_dm_favors,
            'kronos_adds_incremental_value': kronos_adds,
            'criterion': 'kronos adds incremental value ONLY if it wins a '
                         'majority of eligible windows AND the pooled DM '
                         '(normalized errors) significantly favors kronos',
        },
        'success_gate': gate,
        'notes': [
            'statistical significance is NOT trading profitability',
            'no hyperparameter search, no window/regime cherry-picking',
            'Kronos is evaluated as a challenger against the best classical model',
            'the frozen Phase 4/5 and Phase 5b reports are unchanged',
        ],
    }
