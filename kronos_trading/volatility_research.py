"""Phase 5b - does Kronos have genuine volatility/range forecasting skill?

The Phase 5 raw-range target showed Kronos beating the weak "previous range"
persistence baseline on range MAE in all 12 non-daily windows. This module
tests whether that advantage survives when the baseline is replaced by serious
volatility baselines, whether it survives price-level normalization, whether it
is explained by forecast shrinkage, and whether it is regime-confined.

Everything is FIXED a priori (no tuning):

* baselines: previous range, rolling-mean range (5 & 22), EWMA range
  (span=22), HAR-style range (expanding past-only OLS);
* normalized target: ``range / close_current`` (scale-invariant);
* regimes: past-only terciles of a rolling-22 mean range;
* statistical tests: Diebold-Mariano (Newey-West HAC), circular block
  bootstrap, Wilcoxon signed-rank, Spearman/Pearson;
* success gate: 8 explicit, pre-registered criteria.

The experiment only reports; it never tunes, never cherry-picks, and never
claims profitability.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .evaluation import EvaluationConfig, PredictionEvaluator, VolatilityRow
from .statistics_compare import (circular_block_bootstrap_mean_ci,
                                 diebold_mariano, spearman, wilcoxon_signed_rank)

# --- Fixed a priori constants (do NOT tune against results) -----------------
NONDAILY_TIMEFRAMES = frozenset({'1h', '4h'})
BASELINE_KEYS = ('prev', 'rolling5', 'rolling22', 'ewma', 'har')
SERIOUS_KEYS = ('ewma', 'har')
MIN_SAMPLES = 30           # windows below this are 'low_power' (excluded from gate)
GATE_MAJORITY = 0.5        # "beats" = strictly more than half of windows
GATE_SHRINKAGE_MIN_DISPERSION = 0.7  # a priori shrinkage threshold
GATE_P_VALUE = 0.05

VERDICT_MEANING = {
    'A': 'genuine volatility signal',
    'B': 'weak / ambiguous volatility signal',
    'C': 'false positive / no useful volatility signal',
    'pending': 'not enough eligible evidence to classify',
}

EPS = 1e-12


def classify_gate(criteria: Dict[str, Any]) -> str:
    """Map gate criteria to the scientific verdict (A/B/C).

    A = all criteria hold; B = beats the weak persistence baseline but fails at
    least one stronger criterion; C = does not even beat persistence (or the
    advantage disappears under stronger analysis).
    """
    if all(v is True for v in criteria.values()):
        return 'A'
    if criteria.get('c1_beats_previous_range') is True:
        return 'B'
    return 'C'


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _fmean(xs: List[float]) -> Optional[float]:
    return statistics.fmean(xs) if xs else None


def _pairs(pred, actual):
    return [(p, a) for p, a in zip(pred, actual)
            if p is not None and a is not None
            and math.isfinite(p) and math.isfinite(a)]


def _mae(pred, actual) -> Optional[float]:
    pairs = _pairs(pred, actual)
    return statistics.fmean([abs(p - a) for p, a in pairs]) if pairs else None


def _rmse(pred, actual) -> Optional[float]:
    pairs = _pairs(pred, actual)
    if not pairs:
        return None
    return math.sqrt(statistics.fmean([(p - a) ** 2 for p, a in pairs]))


def _pearson(pred, actual) -> Optional[float]:
    pairs = _pairs(pred, actual)
    if len(pairs) < 2:
        return None
    ps = [p for p, _ in pairs]
    ac = [a for _, a in pairs]
    mp, ma = statistics.fmean(ps), statistics.fmean(ac)
    cov = sum((p - mp) * (a - ma) for p, a in pairs)
    vp = sum((p - mp) ** 2 for p in ps)
    va = sum((a - ma) ** 2 for a in ac)
    if vp <= EPS or va <= EPS:
        return None
    return cov / math.sqrt(vp * va)


def _std(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    return statistics.stdev(xs)


def system_stats(pred: List[Optional[float]], actual: List[float]) -> Dict[str, Any]:
    """Full error/statistics profile for one forecasting system."""
    pairs = _pairs(pred, actual)
    n = len(pairs)
    if n == 0:
        return {'sample_size': 0, 'mae': None, 'rmse': None, 'bias': None,
                'mean_pred': None, 'mean_actual': None, 'std_pred': None,
                'std_actual': None, 'dispersion_ratio': None, 'bias_ratio': None,
                'pearson': None, 'spearman': None}
    ps = [p for p, _ in pairs]
    ac = [a for _, a in pairs]
    std_p = _std(ps)
    std_a = _std(ac)
    mean_p = _fmean(ps)
    mean_a = _fmean(ac)
    disp = (std_p / std_a) if (std_a is not None and std_a > EPS) else None
    bias_r = (mean_p / mean_a) if (mean_a is not None and abs(mean_a) > EPS) else None
    return {
        'sample_size': n,
        'mae': _mae(pred, actual),
        'rmse': _rmse(pred, actual),
        'bias': _fmean([p - a for p, a in pairs]),
        'mean_pred': mean_p,
        'mean_actual': mean_a,
        'std_pred': std_p,
        'std_actual': std_a,
        'dispersion_ratio': disp,
        'bias_ratio': bias_r,
        'pearson': _pearson(ps, ac),
        'spearman': spearman(ps, ac),
    }


def _delta(a, b) -> Optional[float]:
    return (a - b) if (a is not None and b is not None) else None


def _winner_lower(a, b) -> Optional[str]:
    if a is None or b is None:
        return None
    return 'kronos' if a < b else ('baseline' if a > b else 'tie')


def _winner_higher(a, b) -> Optional[str]:
    if a is None or b is None:
        return None
    return 'kronos' if a > b else ('baseline' if a < b else 'tie')


# --------------------------------------------------------------------------- #
# Per-window analysis
# --------------------------------------------------------------------------- #
def _baseline_preds(vrows: List[VolatilityRow], key: str) -> List[Optional[float]]:
    return [getattr(r, key + '_range') for r in vrows]


def _analyze_window(vrows: List[VolatilityRow]) -> Dict[str, Any]:
    actual = [r.actual_range for r in vrows]
    actual_norm = [r.actual_range / r.denom_close for r in vrows]

    systems = {}
    systems_norm = {}
    for key in BASELINE_KEYS:
        pred = _baseline_preds(vrows, key)
        systems[key] = system_stats(pred, actual)
        systems_norm[key] = system_stats(
            [p / r.denom_close if p is not None else None
             for p, r in zip(pred, vrows)], actual_norm)
    kronos_pred = [r.kronos_range for r in vrows]
    systems['kronos'] = system_stats(kronos_pred, actual)
    systems_norm['kronos'] = system_stats(
        [r.kronos_range / r.denom_close for r in vrows], actual_norm)

    # Kronos vs each baseline (raw + normalized + paired tests).
    comparisons = {}
    for key in BASELINE_KEYS:
        b_pred = _baseline_preds(vrows, key)
        k_err = [abs(k - a) for k, a in zip(kronos_pred, actual)]
        b_err = [abs(b - a) if b is not None else math.nan
                 for b, a in zip(b_pred, actual)]
        diff = [ke - be for ke, be in zip(k_err, b_err)]
        k_norm_err = [abs(r.kronos_range / r.denom_close - r.actual_range / r.denom_close)
                      for r in vrows]
        b_norm_err = [abs(b / r.denom_close - r.actual_range / r.denom_close)
                      if b is not None else math.nan
                      for b, r in zip(b_pred, vrows)]
        comparisons[key] = {
            'mae_delta': _delta(systems['kronos']['mae'], systems[key]['mae']),
            'mae_winner': _winner_lower(systems['kronos']['mae'], systems[key]['mae']),
            'rmse_delta': _delta(systems['kronos']['rmse'], systems[key]['rmse']),
            'rmse_winner': _winner_lower(systems['kronos']['rmse'], systems[key]['rmse']),
            'norm_mae_delta': _delta(systems_norm['kronos']['mae'], systems_norm[key]['mae']),
            'norm_mae_winner': _winner_lower(systems_norm['kronos']['mae'], systems_norm[key]['mae']),
            'norm_rmse_delta': _delta(systems_norm['kronos']['rmse'], systems_norm[key]['rmse']),
            'norm_rmse_winner': _winner_lower(systems_norm['kronos']['rmse'], systems_norm[key]['rmse']),
            'spearman_delta': _delta(systems['kronos']['spearman'], systems[key]['spearman']),
            'spearman_winner': _winner_higher(systems['kronos']['spearman'], systems[key]['spearman']),
            'pearson_delta': _delta(systems['kronos']['pearson'], systems[key]['pearson']),
            'bias_delta': _delta(systems['kronos']['bias'], systems[key]['bias']),
            'dm': diebold_mariano(k_err, b_err),
            'bootstrap_mean_diff_ci': circular_block_bootstrap_mean_ci(diff),
            'wilcoxon_p_value': wilcoxon_signed_rank(diff)['p_value'],
            'wilcoxon_n_nonzero': wilcoxon_signed_rank(diff)['n_nonzero'],
        }

    # Regimes (raw range), past-only labels already on each row.
    regimes = {}
    regime_counts: Dict[str, int] = {}
    for regime in ('low', 'medium', 'high'):
        sub = [r for r in vrows if r.regime == regime]
        regime_counts[regime] = len(sub)
        regimes[regime] = {
            'sample_size': len(sub),
            'kronos': system_stats([r.kronos_range for r in sub],
                                   [r.actual_range for r in sub]),
            'prev': system_stats([r.prev_range for r in sub],
                                 [r.actual_range for r in sub]),
            'rolling5': system_stats([r.rolling5_range for r in sub],
                                     [r.actual_range for r in sub]),
            'rolling22': system_stats([r.rolling22_range for r in sub],
                                      [r.actual_range for r in sub]),
            'ewma': system_stats([r.ewma_range for r in sub],
                                 [r.actual_range for r in sub]),
            'har': system_stats([r.har_range for r in sub],
                                [r.actual_range for r in sub]),
        }
    regime_counts['undefined'] = sum(1 for r in vrows if r.regime == 'undefined')

    return {
        'sample_size': len(vrows),
        'regime_counts': regime_counts,
        'systems': systems,
        'systems_normalized': systems_norm,
        'comparisons': comparisons,
        'regimes': regimes,
        'shrinkage': {
            'kronos_dispersion_ratio': systems['kronos']['dispersion_ratio'],
            'kronos_bias_ratio': systems['kronos']['bias_ratio'],
            'kronos_std_pred': systems['kronos']['std_pred'],
            'kronos_std_actual': systems['kronos']['std_actual'],
            'kronos_mean_pred': systems['kronos']['mean_pred'],
            'kronos_mean_actual': systems['kronos']['mean_actual'],
        },
    }


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
def _window_record(series: str, timeframe: str, window: str,
                   analysis: Dict[str, Any]) -> Dict[str, Any]:
    s = analysis['systems']
    sn = analysis['systems_normalized']
    def mae(sy, k): return sy[k]['mae']
    beats_ewma = _winner_lower(mae(s, 'kronos'), mae(s, 'ewma')) == 'kronos'
    beats_har = _winner_lower(mae(s, 'kronos'), mae(s, 'har')) == 'kronos'
    beats_ewma_n = _winner_lower(mae(sn, 'kronos'), mae(sn, 'ewma')) == 'kronos'
    beats_har_n = _winner_lower(mae(sn, 'kronos'), mae(sn, 'har')) == 'kronos'
    regime_kronos_beats_serious = {}
    for regime in ('low', 'medium', 'high'):
        rg = analysis['regimes'][regime]
        kronos_mae = rg['kronos']['mae']
        ewma_mae = rg['ewma']['mae']
        har_mae = rg['har']['mae']
        beats = ((kronos_mae is not None and ewma_mae is not None and kronos_mae < ewma_mae)
                 or (kronos_mae is not None and har_mae is not None and kronos_mae < har_mae))
        regime_kronos_beats_serious[regime] = beats
    return {
        'series': series, 'timeframe': timeframe, 'window': window,
        'sample_size': analysis['sample_size'],
        'low_power': analysis['sample_size'] < MIN_SAMPLES,
        'kronos_mae': mae(s, 'kronos'), 'prev_mae': mae(s, 'prev'),
        'rolling5_mae': mae(s, 'rolling5'), 'rolling22_mae': mae(s, 'rolling22'),
        'ewma_mae': mae(s, 'ewma'), 'har_mae': mae(s, 'har'),
        'kronos_nmae': mae(sn, 'kronos'), 'ewma_nmae': mae(sn, 'ewma'),
        'har_nmae': mae(sn, 'har'),
        'kronos_beats_prev': _winner_lower(mae(s, 'kronos'), mae(s, 'prev')) == 'kronos',
        'kronos_beats_ewma': beats_ewma,
        'kronos_beats_har': beats_har,
        'kronos_beats_serious': beats_ewma or beats_har,
        'kronos_beats_serious_norm': beats_ewma_n or beats_har_n,
        'kronos_dispersion_ratio': analysis['shrinkage']['kronos_dispersion_ratio'],
        'kronos_bias_ratio': analysis['shrinkage']['kronos_bias_ratio'],
        'regime_kronos_beats_serious': regime_kronos_beats_serious,
    }


def _majority(count: int, total: int) -> bool:
    return count > GATE_MAJORITY * total


def evaluate_success_gate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pre-registered success gate (all 8 criteria must hold for verdict A)."""
    eligible = [r for r in records
                if not r['low_power'] and r['timeframe'] in NONDAILY_TIMEFRAMES]
    n = len(eligible)
    gate: Dict[str, Any] = {
        'eligible_windows': n,
        'definition': {
            'primary_metric': 'range MAE (raw)',
            'c1': 'kronos beats previous-range persistence in > half of windows',
            'c2': 'kronos beats EWMA or HAR in > half of windows',
            'c3': 'advantage vs serious baseline in >=2 of 4 series',
            'c4': 'effect present in >=2 of 3 chronological windows',
            'c5': 'normalized-range MAE advantage survives in > half of windows',
            'c6': 'pooled Diebold-Mariano (normalized errors) supports improvement',
            'c7': 'improvement not explained solely by shrinkage (dispersion >= 0.7)',
            'c8': 'advantage present in >=2 of 3 regimes (not single-regime)',
        },
        'criteria': {},
    }
    if n == 0:
        gate['overall'] = 'pending'
        gate['verdict'] = 'pending'
        gate['note'] = 'no eligible windows'
        return gate

    c1 = _majority(sum(r['kronos_beats_prev'] for r in eligible), n)
    c2 = _majority(sum(r['kronos_beats_serious'] for r in eligible), n)
    c5 = _majority(sum(r['kronos_beats_serious_norm'] for r in eligible), n)

    # c3: series-level breadth
    series_wins = {}
    for r in eligible:
        series_wins.setdefault(r['series'], []).append(r['kronos_beats_serious'])
    c3 = sum(1 for wins in series_wins.values() if sum(wins) >= 2) >= 2

    # c4: window-level breadth
    window_wins = {}
    for r in eligible:
        window_wins.setdefault(r['window'], []).append(r['kronos_beats_serious'])
    c4 = sum(1 for wins in window_wins.values() if sum(wins) >= 2) >= 2

    # c6 (pooled DM), c7 (pooled dispersion) and c8 (regime breadth) are filled
    # by the caller (run_volatility_research), which has access to the raw
    # per-row volatility data.

    gate['criteria'] = {
        'c1_beats_previous_range': c1,
        'c2_beats_ewma_or_har': c2,
        'c3_series_breadth': c3,
        'c4_window_breadth': c4,
        'c5_normalized_survives': c5,
        'c6_statistical_support': None,  # filled by caller with pooled errors
        'c7_not_solely_shrinkage': None,  # filled by caller
        'c8_regime_breadth': None,  # filled by caller
    }
    return gate


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def run_volatility_research(predictor, config: EvaluationConfig,
                            series: List[Tuple[str, str]],
                            load_candles) -> Dict[str, Any]:
    """Run the volatility research experiment and evaluate the success gate."""
    config = EvaluationConfig(**{**config.asdict()})  # copy (horizon=1 for range)
    series_output: Dict[str, Any] = {}
    window_records: List[Dict[str, Any]] = []
    gate_analyses: List[Tuple[str, str, str, Dict[str, Any]]] = []

    for symbol, timeframe in series:
        evaluator = PredictionEvaluator(predictor, config, symbol, timeframe)
        windows, window_info = evaluator.evaluate_windows(
            load_candles(symbol, timeframe))
        per_window: Dict[str, Any] = {}
        for name, res in windows.items():
            vrows = res.volatility_rows
            analysis = _analyze_window(vrows)
            analysis['_vrows'] = vrows  # retained for pooled stats (not serialized)
            per_window[name] = analysis
            rec = _window_record(symbol, timeframe, name, analysis)
            window_records.append(rec)
            if not rec['low_power'] and timeframe in NONDAILY_TIMEFRAMES:
                gate_analyses.append((symbol, timeframe, name, analysis))
        series_output['%s %s' % (symbol, timeframe)] = {
            'window_info': window_info,
            'windows': {k: {kk: vv for kk, vv in v.items() if kk != '_vrows'}
                        for k, v in per_window.items()},
        }

    # Pooled statistical tests (normalized errors) across eligible windows.
    kronos_pool, ewma_pool, har_pool = [], [], []
    for symbol, timeframe, name, analysis in gate_analyses:
        for r in analysis['_vrows']:
            kronos_pool.append(abs(r.kronos_range / r.denom_close - r.actual_range / r.denom_close))
            if r.ewma_range is not None:
                ewma_pool.append(abs(r.ewma_range / r.denom_close - r.actual_range / r.denom_close))
            if r.har_range is not None:
                har_pool.append(abs(r.har_range / r.denom_close - r.actual_range / r.denom_close))

    dm_ewma = diebold_mariano(kronos_pool, ewma_pool)
    dm_har = diebold_mariano(kronos_pool, har_pool)
    wil_ewma = wilcoxon_signed_rank([k - e for k, e in zip(kronos_pool, ewma_pool)])
    wil_har = wilcoxon_signed_rank([k - e for k, e in zip(kronos_pool, har_pool)])
    boot_ewma = circular_block_bootstrap_mean_ci([k - e for k, e in zip(kronos_pool, ewma_pool)])
    boot_har = circular_block_bootstrap_mean_ci([k - e for k, e in zip(kronos_pool, har_pool)])

    # Gate.
    gate = evaluate_success_gate(window_records)
    if gate.get('overall') != 'pending':
        c6 = ((dm_ewma['p_value'] is not None and dm_ewma['p_value'] < GATE_P_VALUE
               and dm_ewma['winner'] == 'kronos')
              or (dm_har['p_value'] is not None and dm_har['p_value'] < GATE_P_VALUE
                  and dm_har['winner'] == 'kronos'))
        # c7: pooled dispersion ratio across eligible windows
        all_kronos, all_actual = [], []
        for symbol, timeframe, name, analysis in gate_analyses:
            for r in analysis['_vrows']:
                all_kronos.append(r.kronos_range)
                all_actual.append(r.actual_range)
        std_pred = _std(all_kronos) if len(all_kronos) > 1 else None
        std_act = _std(all_actual) if len(all_actual) > 1 else None
        pooled_dispersion = (std_pred / std_act) if (std_act is not None and std_act > EPS) else None
        c7 = (pooled_dispersion is not None and pooled_dispersion >= GATE_SHRINKAGE_MIN_DISPERSION)

        # c8: regime breadth (pooled per-regime MAE vs ewma/har).
        regime_data = {'low': {'kronos': [], 'actual': [], 'ewma': [], 'har': []},
                       'medium': {'kronos': [], 'actual': [], 'ewma': [], 'har': []},
                       'high': {'kronos': [], 'actual': [], 'ewma': [], 'har': []}}
        for symbol, timeframe, name, analysis in gate_analyses:
            for r in analysis['_vrows']:
                if r.regime not in regime_data:
                    continue
                d = regime_data[r.regime]
                d['kronos'].append(r.kronos_range)
                d['actual'].append(r.actual_range)
                if r.ewma_range is not None:
                    d['ewma'].append(r.ewma_range)
                if r.har_range is not None:
                    d['har'].append(r.har_range)
        regime_win = {}
        for regime, d in regime_data.items():
            k_mae = _mae(d['kronos'], d['actual'])
            e_mae = _mae(d['ewma'], d['actual'])
            h_mae = _mae(d['har'], d['actual'])
            regime_win[regime] = (
                (k_mae is not None and e_mae is not None and k_mae < e_mae)
                or (k_mae is not None and h_mae is not None and k_mae < h_mae))
        c8 = sum(regime_win.values()) >= 2

        gate['criteria']['c6_statistical_support'] = bool(c6)
        gate['criteria']['c7_not_solely_shrinkage'] = bool(c7)
        gate['criteria']['c8_regime_breadth'] = bool(c8)
        gate['overall'] = 'pass' if all(v is True for v in gate['criteria'].values()) else 'fail'
        gate['verdict'] = classify_gate(gate['criteria'])
        gate['verdict_meaning'] = VERDICT_MEANING[gate['verdict']]
    else:
        pooled_dispersion = None
        regime_win = {}
        c6 = c7 = c8 = False

    return {
        'kind': 'phase5b_volatility_research',
        'configuration': config.asdict(),
        'baseline_definitions': {
            'A_previous_range': 'range_{t-1} (last closed bar high-low)',
            'B_rolling_mean_range': 'mean of last 5 and last 22 closed ranges '
                                    '(windows fixed a priori)',
            'C_ewma_range': 'EWMA of closed ranges, alpha=2/(span+1), span=22, '
                            'seeded on first closed range (fixed)',
            'D_har_range': 'beta0 + beta1*range_{t-1} + beta2*mean5 + beta3*mean22, '
                           'expanding past-only OLS (min 24 rows, refit per step)',
        },
        'target_definitions': {
            'raw_range': 'high_future - low_future',
            'normalized_range': '(high_future - low_future) / close_current '
                               '(scale-invariant percent range)',
        },
        'oos_methodology': {
            'windows': 'older / middle / recent (fixed, chronological, non-overlapping)',
            'series': 'BTC/USDT & ETH/USDT at 1h/4h/1d',
            'no_lookahead': 'all baselines/regimes use only closed candles strictly '
                            'before the prediction timestamp',
            'fitting': 'HAR refits on expanding past-only data; no future coefficients',
            'daily_note': 'daily windows are low-power supplementary evidence and '
                          'excluded from the success gate',
        },
        'statistical_tests': {
            'dm': 'Diebold-Mariano, two-sided, Newey-West HAC variance (appropriate '
                  'for serially correlated forecast errors)',
            'bootstrap': 'circular block bootstrap 95% CI on paired error differences',
            'wilcoxon': 'Wilcoxon signed-rank (nonparametric robustness; assumes '
                        'exchangeability, listed for completeness)',
            'spearman_pearson': 'rank and linear correlation of predicted vs actual range',
            'multiple_comparisons': 'raw p-values reported; Bonferroni note included',
        },
        'series': series_output,
        'window_records': window_records,
        'pooled_statistics': {
            'dm_vs_ewma': dm_ewma,
            'dm_vs_har': dm_har,
            'wilcoxon_vs_ewma': wil_ewma,
            'wilcoxon_vs_har': wil_har,
            'bootstrap_vs_ewma': boot_ewma,
            'bootstrap_vs_har': boot_har,
            'pooled_dispersion_ratio': pooled_dispersion,
            'regime_kronos_beats_serious': regime_win,
            'bonferroni_note': 'two primary DM tests -> Bonferroni alpha=0.025',
        },
        'success_gate': gate,
        'notes': [
            'statistical significance is NOT trading profitability',
            'no hyperparameter search, no window/regime cherry-picking',
            'the raw-range Phase 5 result is frozen; this experiment is a separate '
            'hypothesis on the same timestamps',
        ],
    }
