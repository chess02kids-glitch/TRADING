"""Phase 7 - cross-asset information experiment (HAR + linear cross-asset extension).

The OHLCV-only model-complexity branch is CLOSED (HAR is the frozen champion;
see ``docs/FINAL_OHLCV_RESEARCH_CONCLUSION.md``). This experiment tests the
single next information hypothesis:

    Does information from the OTHER asset improve volatility forecasting beyond
    single-asset HAR?

Design (pre-registered in ``docs/NEXT_RESEARCH_BRANCH.md``):

* Target: normalized next-candle range ``(high_{t+1} - low_{t+1}) / close_t``
  (raw range reported as secondary).
* Baseline: the FROZEN single-asset HAR (``volatility_baselines.har_forecast``)
  - never re-tuned, never re-fitted differently.
* Cross-asset features (OTHER asset only, exactly four):
  1. previous normalized range ``(high-low)/close``,
  2. trailing 22-bar realized volatility (std of close-to-close returns),
  3. 1-bar return,
  4. 22-bar return.
* Model: HAR + linear cross-asset extension - OLS on RAW range with features
  ``[1, range_{t-1}, mean5, mean22, x_nr_prev, x_rv_22, x_ret_1, x_ret_22]``,
  expanding past-only window, refit every step (same cadence as frozen HAR).
* Strict temporal alignment: for forecasting asset A's candle with open time T,
  every A-feature uses A's candles with open time < T and every cross feature
  uses B's candle with open time == T - step (the last B candle that closed at
  or before T). A missing cross candle is a SKIP, never forward-filled.

No Kronos, no LightGBM, no extra features, no tuning. The gate is pre-registered
in ``evaluate_cross_gate`` and applied before any results are inspected.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .evaluation import EvaluationConfig, PredictionEvaluator, VolatilityRow
from .ml_volatility import _StubPredictor
from .statistics_compare import (circular_block_bootstrap_mean_ci,
                                 diebold_mariano, wilcoxon_signed_rank)
from .volatility_baselines import EWMA_SPAN, HAR_MIN_TRAIN, ROLLING_WINDOWS
from .volatility_research import (MIN_SAMPLES, NONDAILY_TIMEFRAMES, _fmean,
                                  _mae, _std, system_stats)
from .types import Candle

# --------------------------------------------------------------------------- #
# Fixed a priori constants (do NOT tune against OOS results)
# --------------------------------------------------------------------------- #
MIN_HISTORY = 24       # target-feature warm-up (covers mean22 + 22-step returns)
MIN_TRAIN_ROWS = 24    # minimum expanding-OLS training rows (matches frozen HAR)
REFIT_EVERY = 1        # refit OLS every prediction step (matches frozen HAR)
PRIMARY_ALPHA = 0.05   # single primary comparison (cross vs HAR) -> no correction
EXTREME_QUANTILE = 0.99  # for the extreme-period sensitivity check (c7b)

FEATURE_NAMES = [
    'har_range_prev', 'har_mean5', 'har_mean22',
    'x_nr_prev', 'x_rv_22', 'x_ret_1', 'x_ret_22',
]

FEATURE_DESCRIPTIONS = {
    'har_range_prev': 'target asset previous raw range (high-low)',
    'har_mean5': 'target asset mean raw range over last 5',
    'har_mean22': 'target asset mean raw range over last 22',
    'x_nr_prev': 'OTHER asset previous normalized range (high-low)/close',
    'x_rv_22': 'OTHER asset trailing 22-bar realized volatility (std of close-to-close returns)',
    'x_ret_1': 'OTHER asset 1-bar return',
    'x_ret_22': 'OTHER asset 22-bar return',
}

VERDICT_MEANING = {
    'PASS': 'cross-asset information adds genuine incremental value over HAR',
    'B': 'weak / ambiguous cross-asset benefit',
    'C': 'cross-asset information does not robustly beat HAR',
    'pending': 'not enough eligible evidence to classify',
}


# --------------------------------------------------------------------------- #
# Feature construction (strictly past-only, no forward-fill)
# --------------------------------------------------------------------------- #
def build_aligned_features(target: List[Candle], other: List[Candle],
                           step_ms: int) -> Dict[str, np.ndarray]:
    """Build the target/cross feature matrix aligned by target-candle index.

    Row ``j`` of the returned matrix corresponds to the target candle with open
    time ``T_j``. Target HAR features use only target candles with index ``< j``.
    Cross features use ONLY the other asset's candle with open time exactly
    ``T_j - step_ms`` (the last other candle that closed at or before ``T_j``).
    If that other candle is missing, the cross features are NaN and the row is
    invalid (the prediction is skipped - never forward-filled).

    Returns ``{'X', 'y_raw', 'y_norm', 'ts', 'valid', 'cross_missing'}`` where
    ``cross_missing`` counts rows skipped due to a missing cross candle.
    """
    n = len(target)
    X = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=float)
    y_raw = np.full(n, np.nan, dtype=float)
    y_norm = np.full(n, np.nan, dtype=float)
    ts = np.array([c.timestamp_ms for c in target], dtype='int64')
    valid = np.zeros(n, dtype=bool)
    cross_missing = 0

    if n < 2:
        return {'X': X, 'y_raw': y_raw, 'y_norm': y_norm, 'ts': ts,
                'valid': valid, 'cross_missing': cross_missing}

    # target arrays
    t_close = np.array([c.close for c in target], dtype=float)
    t_high = np.array([c.high for c in target], dtype=float)
    t_low = np.array([c.low for c in target], dtype=float)
    t_range = t_high - t_low

    # other arrays + timestamp -> index map
    o_close = np.array([c.close for c in other], dtype=float)
    o_high = np.array([c.high for c in other], dtype=float)
    o_low = np.array([c.low for c in other], dtype=float)
    o_range = o_high - o_low
    o_ts = np.array([c.timestamp_ms for c in other], dtype='int64')
    o_ts_to_idx = {int(t): i for i, t in enumerate(o_ts)}
    o_ret = np.zeros(len(o_close), dtype=float)
    o_ret[1:] = o_close[1:] / o_close[:-1] - 1.0

    def rolling_mean(a, w, end):  # mean of a[end-w:end]
        if end < w:
            return np.nan
        return float(a[end - w:end].mean())

    for j in range(n):
        if j < MIN_HISTORY:
            continue
        # target HAR features (strictly before candle j)
        har_prev = t_range[j - 1]
        har_mean5 = rolling_mean(t_range, 5, j)
        har_mean22 = rolling_mean(t_range, 22, j)
        # target (y) for candle j
        y_raw[j] = t_range[j]
        y_norm[j] = t_range[j] / t_close[j - 1]

        # cross features: other candle with open time exactly T_j - step
        k = o_ts_to_idx.get(int(ts[j] - step_ms))
        if k is None or k < 0:
            cross_missing += 1
            continue
        if k < 22:  # need 22-bar realized vol + 22-step return
            continue
        x_nr_prev = o_range[k] / o_close[k - 1]
        rv_window = o_ret[k - 21:k + 1]  # 22 close-to-close returns ending at k
        x_rv_22 = float(rv_window.std(ddof=1)) if len(rv_window) >= 2 else np.nan
        x_ret_1 = o_close[k] / o_close[k - 1] - 1.0
        x_ret_22 = o_close[k] / o_close[k - 22] - 1.0

        row = [har_prev, har_mean5, har_mean22, x_nr_prev, x_rv_22,
               x_ret_1, x_ret_22]
        if not all(math.isfinite(v) for v in row):
            continue
        X[j] = row
        valid[j] = True

    return {'X': X, 'y_raw': y_raw, 'y_norm': y_norm, 'ts': ts,
            'valid': valid, 'cross_missing': cross_missing}


# --------------------------------------------------------------------------- #
# Expanding past-only OLS walk-forward
# --------------------------------------------------------------------------- #
def run_walk_ols(X: np.ndarray, y: np.ndarray, valid: np.ndarray,
                 target_indices: List[int], min_history: int = MIN_HISTORY,
                 min_train_rows: int = MIN_TRAIN_ROWS,
                 refit_every: int = REFIT_EVERY) -> Dict[str, Any]:
    """Expanding-window OLS walk-forward over ``target_indices``.

    At prediction index ``j`` the coefficients are fitted on rows in
    ``[min_history, j)`` (strictly before ``j``), refit every ``refit_every``
    predictions, and only when at least ``min_train_rows`` training rows exist.
    Returns index-keyed raw-range predictions, a ``leaks`` counter (must stay
    0), and the coefficient history for reproducibility/audit.
    """
    predictions: Dict[int, float] = {}
    beta = None
    last_refit: Optional[int] = None
    last_train_end: Optional[int] = None
    retrains: List[Dict[str, Any]] = []
    coefficients: List[np.ndarray] = []
    leaks = 0
    n = X.shape[0]
    for j in target_indices:
        if j < min_history or j >= n or not valid[j]:
            predictions[j] = math.nan
            continue
        train_mask = valid[min_history:j]
        if int(train_mask.sum()) < min_train_rows:
            predictions[j] = math.nan
            continue
        if beta is None or (last_refit is not None and (j - last_refit) >= refit_every):
            Xtr = X[min_history:j][train_mask]
            ytr = y[min_history:j][train_mask]
            Dtr = np.hstack([np.ones((Xtr.shape[0], 1)), Xtr])
            beta = np.linalg.lstsq(Dtr, ytr, rcond=None)[0]
            last_refit = j
            last_train_end = j  # exclusive: rows [min_history, j) only
            coefficients.append(beta)
            retrains.append({'train_start_idx': min_history, 'train_end_idx': j,
                             'n_train': int(Xtr.shape[0]), 'pred_idx': j})
        # Leak guard: the fitted model must never have seen a row with
        # index >= j (training window end is exclusive == j, i.e. max row j-1).
        if last_train_end is not None and last_train_end > j:
            leaks += 1
        row = X[j]
        if not np.all(np.isfinite(row)):
            predictions[j] = math.nan
            continue
        pred = float(beta[0] + float(beta[1:] @ row))
        predictions[j] = pred if math.isfinite(pred) else math.nan
    return {'predictions': predictions, 'leaks': leaks,
            'retrains': retrains, 'coefficients': coefficients}


# --------------------------------------------------------------------------- #
# Winner / improvement helpers
# --------------------------------------------------------------------------- #
def _winner_lower(a: Optional[float], b: Optional[float]) -> Optional[str]:
    if a is None or b is None:
        return None
    return 'cross' if a < b else ('har' if a > b else 'tie')


def _improvement_pct(base: Optional[float], model: Optional[float]) -> Optional[float]:
    if base is None or model is None or base <= 1e-12:
        return None
    return (base - model) / base * 100.0


# --------------------------------------------------------------------------- #
# Per-window analysis
# --------------------------------------------------------------------------- #
def _window_analysis(vrows: List[VolatilityRow],
                     pred_by_ts: Dict[int, float]) -> Dict[str, Any]:
    # pred_by_ts holds RAW range predictions (the cross model extends frozen
    # HAR, which is fitted on raw range). Normalize by denom_close for the
    # primary normalized-range metrics.
    cross_raw = [pred_by_ts.get(r.prediction_timestamp) for r in vrows]
    actual_norm = [r.actual_range / r.denom_close for r in vrows]
    cross_norm = [(p / r.denom_close) if (p is not None and math.isfinite(p)) else None
                  for p, r in zip(cross_raw, vrows)]
    actual_raw = [r.actual_range for r in vrows]

    systems_norm = {'cross': system_stats(cross_norm, actual_norm)}
    systems_raw = {'cross': system_stats(cross_raw, actual_raw)}
    for key in ('har', 'prev', 'rolling5', 'rolling22', 'ewma'):
        pred = [getattr(r, key + '_range') for r in vrows]
        systems_norm[key] = system_stats(
            [p / r.denom_close if p is not None else None for p, r in zip(pred, vrows)],
            actual_norm)
        systems_raw[key] = system_stats(pred, actual_raw)

    # cross vs each baseline (HAR is primary)
    comparisons: Dict[str, Any] = {}
    for key in ('har', 'prev', 'rolling5', 'rolling22', 'ewma'):
        b_norm = [getattr(r, key + '_range') / r.denom_close
                  if getattr(r, key + '_range') is not None else None for r in vrows]
        c_err = [abs(p - a) if (p is not None and math.isfinite(p)) else math.nan
                 for p, a in zip(cross_norm, actual_norm)]
        b_err = [abs(p - a) if p is not None else math.nan for p, a in zip(b_norm, actual_norm)]
        comparisons['cross_vs_%s' % key] = {
            'norm_mae_delta': (systems_norm['cross']['mae'] - systems_norm[key]['mae'])
            if (systems_norm['cross']['mae'] is not None and systems_norm[key]['mae'] is not None)
            else None,
            'norm_mae_winner': _winner_lower(systems_norm['cross']['mae'], systems_norm[key]['mae']),
            'norm_rmse_delta': (systems_norm['cross']['rmse'] - systems_norm[key]['rmse'])
            if (systems_norm['cross']['rmse'] is not None and systems_norm[key]['rmse'] is not None)
            else None,
            'raw_mae_delta': (systems_raw['cross']['mae'] - systems_raw[key]['mae'])
            if (systems_raw['cross']['mae'] is not None and systems_raw[key]['mae'] is not None)
            else None,
            'dm': diebold_mariano(c_err, b_err, a_name='cross', b_name=key),
            'bootstrap_mean_diff_ci': circular_block_bootstrap_mean_ci(
                [a - b for a, b in zip(c_err, b_err)]),
            'wilcoxon_p_value': wilcoxon_signed_rank(
                [a - b for a, b in zip(c_err, b_err)])['p_value'],
        }

    improvement_vs_har = {
        'norm_mae_pct': _improvement_pct(systems_norm['har']['mae'], systems_norm['cross']['mae']),
        'norm_rmse_pct': _improvement_pct(systems_norm['har']['rmse'], systems_norm['cross']['rmse']),
        'raw_mae_pct': _improvement_pct(systems_raw['har']['mae'], systems_raw['cross']['mae']),
        'raw_rmse_pct': _improvement_pct(systems_raw['har']['rmse'], systems_raw['cross']['rmse']),
    }

    # regimes (normalized target)
    regimes: Dict[str, Any] = {}
    for reg in ('low', 'medium', 'high'):
        sub = [r for r in vrows if r.regime == reg]
        p = [pred_by_ts.get(r.prediction_timestamp) for r in sub]
        a = [r.actual_range / r.denom_close for r in sub]
        h = [r.har_range / r.denom_close if r.har_range is not None else None for r in sub]
        c_mae = _mae(p, a)
        h_mae = _mae(h, a)
        regimes[reg] = {'n': len(sub), 'cross_norm_mae': c_mae, 'har_norm_mae': h_mae,
                        'cross_beats_har': (c_mae is not None and h_mae is not None
                                            and c_mae < h_mae)}

    return {
        'sample_size': len(vrows),
        'low_power': len(vrows) < MIN_SAMPLES,
        'systems_normalized': systems_norm,
        'systems_raw': systems_raw,
        'comparisons': comparisons,
        'improvement_vs_har_pct': improvement_vs_har,
        'regimes': regimes,
        'cross_shrinkage': {
            'dispersion_ratio': systems_norm['cross']['dispersion_ratio'],
            'bias_ratio': systems_norm['cross']['bias_ratio'],
            'std_pred': systems_norm['cross']['std_pred'],
            'std_actual': systems_norm['cross']['std_actual'],
        },
    }


# --------------------------------------------------------------------------- #
# Success gate (pre-registered)
# --------------------------------------------------------------------------- #
def classify_cross_gate(criteria: Dict[str, Any]) -> str:
    if all(v is True for v in criteria.values()):
        return 'PASS'
    if criteria.get('c1_window_breadth') is True:
        return 'B'
    return 'C'


def evaluate_cross_gate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    eligible = [r for r in records
                if not r['low_power'] and r['timeframe'] in NONDAILY_TIMEFRAMES]
    n = len(eligible)
    gate: Dict[str, Any] = {
        'eligible_windows': n,
        'definition': {
            'c1': 'cross beats HAR on normalized MAE in > half of primary windows',
            'c2': 'improvement appears across more than one target asset '
                  '(BTC and ETH each win >=2 of their 6 primary windows)',
            'c3': 'raw-range result is consistent (cross beats HAR raw MAE in '
                  '> half of windows)',
            'c4': 'pooled DM (cross vs HAR, normalized) p<0.05 and cross wins',
            'c5': 'benefit not isolated to one regime (cross wins >=2 of 3 regimes)',
            'c6': 'no look-ahead / leakage (leaks == 0 everywhere)',
            'c7': 'not caused by a single extreme period (>=2 of 3 window '
                  'positions AND DM sign survives removing the top-1% extremes)',
        },
        'criteria': {},
    }
    if n == 0:
        gate['overall'] = 'pending'
        gate['verdict'] = 'pending'
        gate['verdict_meaning'] = VERDICT_MEANING['pending']
        gate['note'] = 'no eligible windows'
        return gate

    c1 = sum(1 for r in eligible if r['cross_har_nmae_winner']) > 0.5 * n
    c3 = sum(1 for r in eligible if r['cross_har_mae_winner']) > 0.5 * n
    c6 = all(r['leaks'] == 0 for r in eligible)

    # c2: more than one asset - each target asset wins >=2 of its 6 windows
    asset_wins: Dict[str, List[bool]] = {}
    for r in eligible:
        asset_wins.setdefault(r['series'], []).append(r['cross_har_nmae_winner'])
    c2 = (len(asset_wins) >= 2
          and all(sum(w) >= 2 for w in asset_wins.values()))

    # c7a: >=2 of 3 window positions show cross winning (aggregated)
    window_wins: Dict[str, List[bool]] = {}
    for r in eligible:
        window_wins.setdefault(r['window'], []).append(r['cross_har_nmae_winner'])
    c7a = sum(1 for w in window_wins.values() if sum(w) > 0) >= 2

    gate['criteria'] = {
        'c1_window_breadth': bool(c1),
        'c2_asset_breadth': bool(c2),
        'c3_raw_consistent': bool(c3),
        'c4_statistical_support': None,  # filled by caller (pooled DM)
        'c5_regime_breadth': None,       # filled by caller (pooled regimes)
        'c6_no_leakage': bool(c6),
        'c7_not_single_period_window': bool(c7a),  # c7b filled by caller
        'c7b_not_single_period_extreme': None,
    }
    return gate


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def _window_record(series: str, timeframe: str, window: str,
                   analysis: Dict[str, Any], leaks: int) -> Dict[str, Any]:
    sn = analysis['systems_normalized']
    sr = analysis['systems_raw']
    return {
        'series': series, 'timeframe': timeframe, 'window': window,
        'sample_size': analysis['sample_size'], 'low_power': analysis['low_power'],
        'cross_norm_mae': sn['cross']['mae'], 'har_norm_mae': sn['har']['mae'],
        'cross_norm_rmse': sn['cross']['rmse'], 'har_norm_rmse': sn['har']['rmse'],
        'cross_raw_mae': sr['cross']['mae'], 'har_raw_mae': sr['har']['mae'],
        'cross_har_nmae_winner': _winner_lower(sn['cross']['mae'], sn['har']['mae']) == 'cross',
        'cross_har_nrmse_winner': _winner_lower(sn['cross']['rmse'], sn['har']['rmse']) == 'cross',
        'cross_har_mae_winner': _winner_lower(sr['cross']['mae'], sr['har']['mae']) == 'cross',
        'cross_dispersion_ratio': sn['cross']['dispersion_ratio'],
        'cross_bias_ratio': sn['cross']['bias_ratio'],
        'improvement_vs_har_nmae_pct': analysis['improvement_vs_har_pct']['norm_mae_pct'],
        'improvement_vs_har_raw_mae_pct': analysis['improvement_vs_har_pct']['raw_mae_pct'],
        'leaks': leaks,
        'cross_missing': analysis['cross_missing'],
        'regime_cross_beats_har': {reg: analysis['regimes'][reg]['cross_beats_har']
                                   for reg in ('low', 'medium', 'high')},
    }


def run_cross_asset(load_candles: Callable[[str, str], List[Candle]],
                    config: EvaluationConfig,
                    pairs: List[Tuple[str, str, str]]) -> Dict[str, Any]:
    """Run the cross-asset experiment over (target, other, timeframe) pairs."""
    series_output: Dict[str, Any] = {}
    window_records: List[Dict[str, Any]] = []
    gate_analyses: List[Tuple[str, str, str, Dict[str, Any], List[VolatilityRow],
                              Dict[int, float], int]] = []
    all_beta: List[np.ndarray] = []

    for target_sym, other_sym, timeframe in pairs:
        target = load_candles(target_sym, timeframe)
        other = load_candles(other_sym, timeframe)
        ev = PredictionEvaluator(_StubPredictor(), config, target_sym, timeframe)
        closed = ev._closed_data(target)
        step_ms = ev.tf_ms
        built = build_aligned_features(closed, sorted(other, key=lambda c: c.timestamp_ms),
                                       step_ms)
        ts_to_idx = {int(t): i for i, t in enumerate(built['ts'])}
        windows, window_info = ev.evaluate_windows(target)

        series_id = '%s<-other(%s)' % (target_sym, other_sym)
        per_window: Dict[str, Any] = {}
        for name, res in windows.items():
            vrows = res.volatility_rows
            idxs = [ts_to_idx[int(r.prediction_timestamp)] for r in vrows]
            walk = run_walk_ols(built['X'], built['y_raw'], built['valid'], idxs)
            all_beta.extend(walk['coefficients'])
            pred_by_ts = {int(r.prediction_timestamp): walk['predictions'][j]
                          for r, j in zip(vrows, idxs)}
            analysis = _window_analysis(vrows, pred_by_ts)
            analysis['cross_missing'] = built['cross_missing']
            per_window[name] = analysis
            rec = _window_record(series_id, timeframe, name, analysis, walk['leaks'])
            window_records.append(rec)
            if not rec['low_power'] and timeframe in NONDAILY_TIMEFRAMES:
                gate_analyses.append((series_id, timeframe, name, analysis, vrows,
                                      pred_by_ts, walk['leaks']))
        series_output[series_id] = {'timeframe': timeframe,
                                    'other_asset': other_sym,
                                    'window_info': window_info,
                                    'windows': per_window}

    # Pooled statistics (normalized errors) over primary windows.
    c_err_har, h_err = [], []
    c_err_prev, p_err = [], []
    c_err_ewma, e_err = [], []
    c_err_r5, r5_err = [], []
    c_err_r22, r22_err = [], []
    all_cross, all_actual, all_har = [], [], []
    for _, _, _, analysis, vrows, pred_by_ts, _ in gate_analyses:
        for r in vrows:
            p = pred_by_ts.get(r.prediction_timestamp)
            if p is None or not math.isfinite(p):
                continue
            a = r.actual_range / r.denom_close
            p_norm = p / r.denom_close  # raw prediction -> normalized
            c_err_har.append(abs(p_norm - a))
            if r.har_range is not None:
                h_err.append(abs(r.har_range / r.denom_close - a))
            if r.prev_range is not None:
                c_err_prev.append(abs(p_norm - a))
                p_err.append(abs(r.prev_range / r.denom_close - a))
            if r.ewma_range is not None:
                c_err_ewma.append(abs(p_norm - a))
                e_err.append(abs(r.ewma_range / r.denom_close - a))
            if r.rolling5_range is not None:
                c_err_r5.append(abs(p_norm - a))
                r5_err.append(abs(r.rolling5_range / r.denom_close - a))
            if r.rolling22_range is not None:
                c_err_r22.append(abs(p_norm - a))
                r22_err.append(abs(r.rolling22_range / r.denom_close - a))
            all_cross.append(p_norm)
            all_actual.append(a)
            if r.har_range is not None:
                all_har.append(r.har_range / r.denom_close)

    pooled = {
        'cross_vs_har': diebold_mariano(c_err_har, h_err, a_name='cross', b_name='har'),
        'cross_vs_prev': diebold_mariano(c_err_prev, p_err, a_name='cross', b_name='prev'),
        'cross_vs_ewma': diebold_mariano(c_err_ewma, e_err, a_name='cross', b_name='ewma'),
        'cross_vs_rolling5': diebold_mariano(c_err_r5, r5_err, a_name='cross', b_name='rolling5'),
        'cross_vs_rolling22': diebold_mariano(c_err_r22, r22_err, a_name='cross', b_name='rolling22'),
    }

    # Pooled regime analysis.
    regime_pool = {'low': {'cross': [], 'actual': [], 'har': []},
                   'medium': {'cross': [], 'actual': [], 'har': []},
                   'high': {'cross': [], 'actual': [], 'har': []}}
    for _, _, _, analysis, vrows, pred_by_ts, _ in gate_analyses:
        for r in vrows:
            if r.regime not in regime_pool:
                continue
            p = pred_by_ts.get(r.prediction_timestamp)
            if p is None or not math.isfinite(p):
                continue
            d = regime_pool[r.regime]
            a = r.actual_range / r.denom_close
            d['cross'].append(p / r.denom_close)
            d['actual'].append(a)
            if r.har_range is not None:
                d['har'].append(r.har_range / r.denom_close)
    regime_result = {}
    for reg, d in regime_pool.items():
        c_mae = _mae(d['cross'], d['actual'])
        h_mae = _mae(d['har'], d['actual'])
        regime_result[reg] = {'cross_norm_mae': c_mae, 'har_norm_mae': h_mae,
                              'cross_beats_har': (c_mae is not None and h_mae is not None
                                                  and c_mae < h_mae),
                              'n': len(d['actual'])}
    c5 = sum(1 for d in regime_result.values() if d['cross_beats_har'] is True) >= 2

    # c4: pooled DM (cross vs HAR, normalized).
    dm_cross_har = pooled['cross_vs_har']
    c4 = (dm_cross_har['p_value'] is not None
          and dm_cross_har['p_value'] < PRIMARY_ALPHA
          and dm_cross_har['mean_loss_diff'] is not None
          and dm_cross_har['mean_loss_diff'] < 0)

    # c7b: extreme-period sensitivity - remove top-1% largest |actual_norm|
    # and check the pooled DM sign for cross-vs-HAR does not flip.
    arr_a = np.array(all_actual, dtype=float)
    arr_c = np.array(all_cross, dtype=float)
    arr_h = np.array(all_har, dtype=float)
    if len(arr_a) > 50:
        thr = float(np.quantile(np.abs(arr_a), EXTREME_QUANTILE))
        keep = np.abs(arr_a) <= thr
        dm_trim = diebold_mariano(
            np.abs(arr_c[keep] - arr_a[keep]).tolist(),
            np.abs(arr_h[keep] - arr_a[keep]).tolist(),
            a_name='cross', b_name='har')
    else:
        dm_trim = {'mean_loss_diff': None, 'p_value': None, 'winner': None}
    c7b = (dm_trim.get('mean_loss_diff') is not None
           and dm_trim['mean_loss_diff'] < 0)

    # shrinkage (pooled dispersion of cross predictions).
    std_pred = _std(all_cross) if len(all_cross) > 1 else None
    std_act = _std(all_actual) if len(all_actual) > 1 else None
    pooled_dispersion = (std_pred / std_act) if (std_act is not None and std_act > 1e-12) else None
    pooled_bias = (_fmean(all_cross) / _fmean(all_actual)) \
        if (_fmean(all_cross) is not None and abs(_fmean(all_actual)) > 1e-12) else None

    # coefficient summary (mean + std across all retrains)
    coefficient_summary = None
    if all_beta:
        arr = np.vstack(all_beta)
        names = ['intercept'] + FEATURE_NAMES
        coefficient_summary = [
            {'coefficient': names[k], 'mean': float(arr[:, k].mean()),
             'std': float(arr[:, k].std())}
            for k in range(arr.shape[1])]

    gate = evaluate_cross_gate(window_records)
    if gate.get('overall') != 'pending':
        gate['criteria']['c4_statistical_support'] = bool(c4)
        gate['criteria']['c5_regime_breadth'] = bool(c5)
        gate['criteria']['c7b_not_single_period_extreme'] = bool(c7b)
        gate['overall'] = 'pass' if all(v is True for v in gate['criteria'].values()) else 'fail'
        gate['verdict'] = classify_cross_gate(gate['criteria'])
        gate['verdict_meaning'] = VERDICT_MEANING[gate['verdict']]

    return {
        'kind': 'cross_asset_volatility',
        'configuration': config.asdict(),
        'model': {
            'form': 'HAR + linear cross-asset extension (OLS on raw range)',
            'features': FEATURE_NAMES,
            'feature_descriptions': FEATURE_DESCRIPTIONS,
            'min_history': MIN_HISTORY,
            'refit_every': REFIT_EVERY,
            'alignment_rule': 'cross features use the OTHER asset candle with '
                              'open time == T - step; missing -> skip (no forward-fill)',
        },
        'target': {
            'primary': '(high_{t+1} - low_{t+1}) / close_t',
            'secondary': 'high_{t+1} - low_{t+1}',
        },
        'frozen_baseline': {
            'name': 'single-asset HAR (frozen)',
            'formula': 'beta0 + beta1*range_{t-1} + beta2*mean5 + beta3*mean22',
            'note': 'never re-tuned; from volatility_baselines.har_forecast',
        },
        'statistical_methodology': {
            'primary_comparison': 'cross-asset vs HAR (normalized range errors)',
            'dm': 'two-sided Diebold-Mariano, Newey-West HAC variance',
            'bootstrap': 'circular block bootstrap 95% CI on paired error differences',
            'wilcoxon': 'Wilcoxon signed-rank (nonparametric robustness)',
            'multiple_testing': 'single primary comparison -> alpha=0.05; the four '
                                'secondary comparisons reported without gate weight',
        },
        'series': series_output,
        'window_records': window_records,
        'pooled_statistics': pooled,
        'regime_pooled': regime_result,
        'cross_adequacy': {
            'pooled_dispersion_ratio': pooled_dispersion,
            'pooled_bias_ratio': pooled_bias,
            'extreme_sensitivity_dm_trimmed': dm_trim,
            'coefficient_summary': coefficient_summary,
        },
        'success_gate': gate,
        'notes': [
            'statistical significance is NOT trading profitability',
            'no hyperparameter search, no window/regime cherry-picking, no tuning',
            'frozen single-asset HAR methodology is unchanged',
            'daily windows are supplementary and excluded from the gate',
        ],
    }
