"""Phase 8 (F-01) - funding-only positioning vs frozen HAR.

The pre-registered Phase 8 experiment is FUNDING-ONLY. It tests whether two
derived funding covariates contain incremental information for next-candle
normalized range beyond the frozen single-asset HAR model:

    funding_mean_24h     = mean of settled funding rates in (t-24h, t]
    abs_funding_mean_24h = mean of |funding rate| in (t-24h, t]

H0: funding information provides no incremental information beyond HAR.
H1: funding information provides incremental information.

Design (pre-registered, see docs/DERIVATIVES_METHODOLOGY.md):

* Target: normalized next-candle range (raw range secondary).
* Frozen baseline: single-asset HAR, unchanged.
* Exactly 2 external features (derived from settled funding), point-in-time.
* Model: HAR + linear extension, expanding past-only OLS on raw range, refit
  every step, min 24 training rows. No ML, no feature selection, no tuning.
* Missing required funding observation => skip that row (no forward-fill,
  no interpolation).
* Pre-registered C1-C7 gate; PASS = all 7, B = C1 true only, C = C1 false.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .derivatives_data import funding_features_24h
from .evaluation import EvaluationConfig, PredictionEvaluator, VolatilityRow
from .ml_volatility import _StubPredictor
from .statistics_compare import (circular_block_bootstrap_mean_ci,
                                 diebold_mariano, wilcoxon_signed_rank)
from .volatility_research import (MIN_SAMPLES, NONDAILY_TIMEFRAMES, _fmean,
                                  _mae, _std, system_stats)
from .types import Candle

MIN_HISTORY = 24
MIN_TRAIN_ROWS = 24
REFIT_EVERY = 1
PRIMARY_ALPHA = 0.05

FEATURE_NAMES = ['har_range_prev', 'har_mean5', 'har_mean22',
                 'funding_mean_24h', 'abs_funding_mean_24h']

FEATURE_DESCRIPTIONS = {
    'har_range_prev': 'target asset previous raw range (high-low)',
    'har_mean5': 'target asset mean raw range over last 5',
    'har_mean22': 'target asset mean raw range over last 22',
    'funding_mean_24h': 'mean of settled funding rates in (t-24h, t]',
    'abs_funding_mean_24h': 'mean of |funding rate| in (t-24h, t]',
}

VERDICT_MEANING = {
    'PASS': 'funding positioning adds genuine incremental value over HAR',
    'B': 'weak / ambiguous funding benefit',
    'C': 'funding positioning does not robustly beat HAR',
    'pending': 'not enough eligible evidence to classify',
}


# --------------------------------------------------------------------------- #
# Feature construction (point-in-time, no forward-fill)
# --------------------------------------------------------------------------- #
def build_derivatives_features(target: List[Candle],
                               funding_rows: List[Dict[str, Any]],
                               step_ms: int) -> Dict[str, Any]:
    """Build (X, y_raw, y_norm, ts, valid) with target + funding features.

    Row ``j`` corresponds to the target candle with open time ``T_j``. Target
    HAR features use only target candles with index < j. Funding features use
    only settled funding rates with funding_time <= T_j (prediction timestamp),
    computed by ``funding_features_24h``; a missing/stale observation makes the
    row invalid (skip, never forward-filled).
    """
    n = len(target)
    X = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=float)
    y_raw = np.full(n, np.nan, dtype=float)
    y_norm = np.full(n, np.nan, dtype=float)
    ts = np.array([c.timestamp_ms for c in target], dtype='int64')
    valid = np.zeros(n, dtype=bool)
    missing = 0

    if n < 2:
        return {'X': X, 'y_raw': y_raw, 'y_norm': y_norm, 'ts': ts,
                'valid': valid, 'missing': missing}

    t_close = np.array([c.close for c in target], dtype=float)
    t_high = np.array([c.high for c in target], dtype=float)
    t_low = np.array([c.low for c in target], dtype=float)
    t_range = t_high - t_low

    # Funding features at each prediction timestamp (target candle open time).
    feat = funding_features_24h(funding_rows, ts.tolist())
    funding_mean = feat['funding_mean_24h']
    abs_funding_mean = feat['abs_funding_mean_24h']

    def rolling_mean(a, w, end):
        if end < w:
            return np.nan
        return float(a[end - w:end].mean())

    for j in range(n):
        if j < MIN_HISTORY:
            continue
        har_prev = t_range[j - 1]
        har_mean5 = rolling_mean(t_range, 5, j)
        har_mean22 = rolling_mean(t_range, 22, j)
        y_raw[j] = t_range[j]
        y_norm[j] = t_range[j] / t_close[j - 1]

        f_mean = funding_mean[j]
        f_abs = abs_funding_mean[j]
        if f_mean is None or f_abs is None:
            missing += 1
            continue
        row = [har_prev, har_mean5, har_mean22, f_mean, f_abs]
        if not all(math.isfinite(v) for v in row):
            missing += 1
            continue
        X[j] = row
        valid[j] = True

    return {'X': X, 'y_raw': y_raw, 'y_norm': y_norm, 'ts': ts,
            'valid': valid, 'missing': missing}


# --------------------------------------------------------------------------- #
# Expanding past-only OLS walk-forward
# --------------------------------------------------------------------------- #
def run_walk_ols(X: np.ndarray, y: np.ndarray, valid: np.ndarray,
                 target_indices: List[int], min_history: int = MIN_HISTORY,
                 min_train_rows: int = MIN_TRAIN_ROWS,
                 refit_every: int = REFIT_EVERY) -> Dict[str, Any]:
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
            last_train_end = j
            coefficients.append(beta)
            retrains.append({'train_start_idx': min_history, 'train_end_idx': j,
                             'n_train': int(Xtr.shape[0]), 'pred_idx': j})
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
# Window analysis
# --------------------------------------------------------------------------- #
def _winner_lower(a: Optional[float], b: Optional[float]) -> Optional[str]:
    if a is None or b is None:
        return None
    return 'ext' if a < b else ('har' if a > b else 'tie')


def _improvement_pct(base: Optional[float], model: Optional[float]) -> Optional[float]:
    if base is None or model is None or base <= 1e-12:
        return None
    return (base - model) / base * 100.0


def _window_analysis(vrows: List[VolatilityRow],
                     pred_by_ts: Dict[int, float]) -> Dict[str, Any]:
    ext_raw = [pred_by_ts.get(r.prediction_timestamp) for r in vrows]
    actual_norm = [r.actual_range / r.denom_close for r in vrows]
    ext_norm = [(p / r.denom_close) if (p is not None and math.isfinite(p)) else None
                for p, r in zip(ext_raw, vrows)]
    actual_raw = [r.actual_range for r in vrows]

    systems_norm = {'ext': system_stats(ext_norm, actual_norm)}
    systems_raw = {'ext': system_stats(ext_raw, actual_raw)}
    for key in ('har', 'prev', 'rolling5', 'rolling22', 'ewma'):
        pred = [getattr(r, key + '_range') for r in vrows]
        systems_norm[key] = system_stats(
            [p / r.denom_close if p is not None else None for p, r in zip(pred, vrows)],
            actual_norm)
        systems_raw[key] = system_stats(pred, actual_raw)

    comparisons: Dict[str, Any] = {}
    for key in ('har', 'prev', 'rolling5', 'rolling22', 'ewma'):
        b_norm = [getattr(r, key + '_range') / r.denom_close
                  if getattr(r, key + '_range') is not None else None for r in vrows]
        e_err = [abs(p - a) if (p is not None and math.isfinite(p)) else math.nan
                 for p, a in zip(ext_norm, actual_norm)]
        b_err = [abs(p - a) if p is not None else math.nan for p, a in zip(b_norm, actual_norm)]
        comparisons['ext_vs_%s' % key] = {
            'norm_mae_delta': (systems_norm['ext']['mae'] - systems_norm[key]['mae'])
            if (systems_norm['ext']['mae'] is not None and systems_norm[key]['mae'] is not None)
            else None,
            'norm_mae_winner': _winner_lower(systems_norm['ext']['mae'], systems_norm[key]['mae']),
            'norm_rmse_delta': (systems_norm['ext']['rmse'] - systems_norm[key]['rmse'])
            if (systems_norm['ext']['rmse'] is not None and systems_norm[key]['rmse'] is not None)
            else None,
            'raw_mae_delta': (systems_raw['ext']['mae'] - systems_raw[key]['mae'])
            if (systems_raw['ext']['mae'] is not None and systems_raw[key]['mae'] is not None)
            else None,
            'dm': diebold_mariano(e_err, b_err, a_name='ext', b_name=key),
            'bootstrap_mean_diff_ci': circular_block_bootstrap_mean_ci(
                [a - b for a, b in zip(e_err, b_err)]),
            'wilcoxon_p_value': wilcoxon_signed_rank(
                [a - b for a, b in zip(e_err, b_err)])['p_value'],
        }

    improvement_vs_har = {
        'norm_mae_pct': _improvement_pct(systems_norm['har']['mae'], systems_norm['ext']['mae']),
        'norm_rmse_pct': _improvement_pct(systems_norm['har']['rmse'], systems_norm['ext']['rmse']),
        'raw_mae_pct': _improvement_pct(systems_raw['har']['mae'], systems_raw['ext']['mae']),
        'raw_rmse_pct': _improvement_pct(systems_raw['har']['rmse'], systems_raw['ext']['rmse']),
    }

    regimes: Dict[str, Any] = {}
    for reg in ('low', 'medium', 'high'):
        sub = [r for r in vrows if r.regime == reg]
        p = [pred_by_ts.get(r.prediction_timestamp) for r in sub]
        a = [r.actual_range / r.denom_close for r in sub]
        h = [r.har_range / r.denom_close if r.har_range is not None else None for r in sub]
        e_mae = _mae(p, a)
        h_mae = _mae(h, a)
        regimes[reg] = {'n': len(sub), 'ext_norm_mae': e_mae, 'har_norm_mae': h_mae,
                        'ext_beats_har': (e_mae is not None and h_mae is not None
                                          and e_mae < h_mae)}

    return {
        'sample_size': len(vrows),
        'low_power': len(vrows) < MIN_SAMPLES,
        'systems_normalized': systems_norm,
        'systems_raw': systems_raw,
        'comparisons': comparisons,
        'improvement_vs_har_pct': improvement_vs_har,
        'regimes': regimes,
        'ext_shrinkage': {
            'dispersion_ratio': systems_norm['ext']['dispersion_ratio'],
            'bias_ratio': systems_norm['ext']['bias_ratio'],
            'std_pred': systems_norm['ext']['std_pred'],
            'std_actual': systems_norm['ext']['std_actual'],
        },
    }


# --------------------------------------------------------------------------- #
# Gate (pre-registered C1-C7, identical to the Phase 7 gate)
# --------------------------------------------------------------------------- #
def classify_derivatives_gate(criteria: Dict[str, Any]) -> str:
    if all(v is True for v in criteria.values()):
        return 'PASS'
    if criteria.get('c1_window_breadth_per_asset') is True:
        return 'B'
    return 'C'


def _asset_beats_2of3(records: List[Dict[str, Any]], field: str, asset: str) -> bool:
    wins = [r[field] for r in records
            if r['asset'] == asset and r['timeframe'] in NONDAILY_TIMEFRAMES]
    return len(wins) >= 3 and sum(wins) >= 2


def evaluate_derivatives_gate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    eligible = [r for r in records
                if not r['low_power'] and r['timeframe'] in NONDAILY_TIMEFRAMES]
    n = len(eligible)
    gate: Dict[str, Any] = {
        'eligible_windows': n,
        'definition': {
            'C1': 'ext beats HAR on normalized MAE in >=2/3 windows for >=1 asset',
            'C2': 'evidence exists in BOTH BTC and ETH (each >=2/3 nMAE wins)',
            'C3': 'normalized RMSE has the same broad pattern (both assets >=2/3)',
            'C4': 'improvement survives raw-range evaluation (> half windows raw MAE)',
            'C5': 'pooled DM p<0.05 AND mean loss difference favors ext',
            'C6': 'improvement exists in >=2/3 volatility regimes',
            'C7': 'no leakage (leaks == 0 everywhere)',
        },
        'criteria': {},
    }
    if n == 0:
        gate['overall'] = 'pending'
        gate['verdict'] = 'pending'
        gate['verdict_meaning'] = VERDICT_MEANING['pending']
        gate['note'] = 'no eligible windows'
        return gate

    btc_nmae = _asset_beats_2of3(eligible, 'ext_har_nmae_winner', 'BTC/USDT')
    eth_nmae = _asset_beats_2of3(eligible, 'ext_har_nmae_winner', 'ETH/USDT')
    btc_nrmse = _asset_beats_2of3(eligible, 'ext_har_nrmse_winner', 'BTC/USDT')
    eth_nrmse = _asset_beats_2of3(eligible, 'ext_har_nrmse_winner', 'ETH/USDT')

    c1 = btc_nmae or eth_nmae
    c2 = btc_nmae and eth_nmae
    c3 = btc_nrmse and eth_nrmse
    c4 = sum(1 for r in eligible if r['ext_har_mae_winner']) > 0.5 * n
    c7 = all(r['leaks'] == 0 for r in eligible)

    gate['criteria'] = {
        'c1_window_breadth_per_asset': bool(c1),
        'c2_both_assets': bool(c2),
        'c3_rmse_pattern': bool(c3),
        'c4_raw_survives': bool(c4),
        'c5_statistical_support': None,
        'c6_regime_breadth': None,
        'c7_no_leakage': bool(c7),
    }
    return gate


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def _window_record(asset: str, series: str, timeframe: str, window: str,
                   analysis: Dict[str, Any], leaks: int, missing: int) -> Dict[str, Any]:
    sn = analysis['systems_normalized']
    sr = analysis['systems_raw']
    return {
        'asset': asset, 'series': series, 'timeframe': timeframe, 'window': window,
        'sample_size': analysis['sample_size'], 'low_power': analysis['low_power'],
        'ext_norm_mae': sn['ext']['mae'], 'har_norm_mae': sn['har']['mae'],
        'ext_norm_rmse': sn['ext']['rmse'], 'har_norm_rmse': sn['har']['rmse'],
        'ext_raw_mae': sr['ext']['mae'], 'har_raw_mae': sr['har']['mae'],
        'ext_har_nmae_winner': _winner_lower(sn['ext']['mae'], sn['har']['mae']) == 'ext',
        'ext_har_nrmse_winner': _winner_lower(sn['ext']['rmse'], sn['har']['rmse']) == 'ext',
        'ext_har_mae_winner': _winner_lower(sr['ext']['mae'], sr['har']['mae']) == 'ext',
        'ext_dispersion_ratio': sn['ext']['dispersion_ratio'],
        'ext_bias_ratio': sn['ext']['bias_ratio'],
        'improvement_vs_har_nmae_pct': analysis['improvement_vs_har_pct']['norm_mae_pct'],
        'improvement_vs_har_raw_mae_pct': analysis['improvement_vs_har_pct']['raw_mae_pct'],
        'leaks': leaks, 'missing_funding': missing,
        'regime_ext_beats_har': {reg: analysis['regimes'][reg]['ext_beats_har']
                                 for reg in ('low', 'medium', 'high')},
    }


def run_derivatives_volatility(load_candles: Callable[[str, str], List[Candle]],
                               load_funding: Callable[[str], Dict[str, List[Dict[str, Any]]]],
                               config: EvaluationConfig,
                               series: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Run the funding-only (F-01) derivatives-vs-HAR experiment.

    ``load_funding(symbol)`` returns ``{'funding': [{timestamp_ms, funding_rate}]}``
    (point-in-time settled funding history; see ``derivatives_data``). No open
    interest or basis is required or read.
    """
    series_output: Dict[str, Any] = {}
    window_records: List[Dict[str, Any]] = []
    gate_analyses: List[Tuple[str, str, str, Dict[str, Any], List[VolatilityRow],
                              Dict[int, float], int, int]] = []
    all_beta: List[np.ndarray] = []

    def symbol_of(asset: str) -> str:
        return asset.replace('/', '')

    for symbol, timeframe in series:
        candles = load_candles(symbol, timeframe)
        funding_data = load_funding(symbol_of(symbol))
        funding_rows = funding_data.get('funding', [])
        ev = PredictionEvaluator(_StubPredictor(), config, symbol, timeframe)
        closed = ev._closed_data(candles)
        step_ms = ev.tf_ms
        built = build_derivatives_features(closed, funding_rows, step_ms)
        ts_to_idx = {int(t): i for i, t in enumerate(built['ts'])}
        windows, window_info = ev.evaluate_windows(candles)

        per_window: Dict[str, Any] = {}
        for name, res in windows.items():
            vrows = res.volatility_rows
            idxs = [ts_to_idx[int(r.prediction_timestamp)] for r in vrows]
            walk = run_walk_ols(built['X'], built['y_raw'], built['valid'], idxs)
            all_beta.extend(walk['coefficients'])
            pred_by_ts = {int(r.prediction_timestamp): walk['predictions'][j]
                          for r, j in zip(vrows, idxs)}
            analysis = _window_analysis(vrows, pred_by_ts)
            per_window[name] = analysis
            rec = _window_record(symbol, symbol, timeframe, name, analysis,
                                 walk['leaks'], built['missing'])
            window_records.append(rec)
            if not rec['low_power'] and timeframe in NONDAILY_TIMEFRAMES:
                gate_analyses.append((symbol, timeframe, name, analysis, vrows,
                                      pred_by_ts, walk['leaks'], built['missing']))
        series_output[symbol] = {'timeframe': timeframe,
                                 'window_info': window_info,
                                 'windows': per_window,
                                 'missing_funding': built['missing']}

    # Pooled statistics (normalized errors).
    e_err_har, h_err = [], []
    all_ext, all_actual, all_har = [], [], []
    for _, _, _, analysis, vrows, pred_by_ts, _, _ in gate_analyses:
        for r in vrows:
            p = pred_by_ts.get(r.prediction_timestamp)
            if p is None or not math.isfinite(p):
                continue
            a = r.actual_range / r.denom_close
            pn = p / r.denom_close
            e_err_har.append(abs(pn - a))
            if r.har_range is not None:
                h_err.append(abs(r.har_range / r.denom_close - a))
            all_ext.append(pn)
            all_actual.append(a)
            if r.har_range is not None:
                all_har.append(r.har_range / r.denom_close)

    dm_ext_har = diebold_mariano(e_err_har, h_err, a_name='ext', b_name='har')
    pooled_primary = {
        'dm': dm_ext_har,
        'bootstrap_mean_diff_ci': circular_block_bootstrap_mean_ci(
            [a - b for a, b in zip(e_err_har, h_err)]),
        'wilcoxon_p_value': wilcoxon_signed_rank(
            [a - b for a, b in zip(e_err_har, h_err)])['p_value'],
    }

    # Extreme-trimmed DM (diagnostic).
    arr_a = np.array(all_actual, dtype=float)
    arr_e = np.array(all_ext, dtype=float)
    arr_h = np.array(all_har, dtype=float)
    if len(arr_a) > 50:
        thr = float(np.quantile(np.abs(arr_a), 0.99))
        keep = np.abs(arr_a) <= thr
        dm_trim = diebold_mariano(
            np.abs(arr_e[keep] - arr_a[keep]).tolist(),
            np.abs(arr_h[keep] - arr_a[keep]).tolist(),
            a_name='ext', b_name='har')
    else:
        dm_trim = {'mean_loss_diff': None, 'p_value': None, 'winner': None}

    # Regime pooling.
    regime_pool = {'low': {'ext': [], 'actual': [], 'har': []},
                   'medium': {'ext': [], 'actual': [], 'har': []},
                   'high': {'ext': [], 'actual': [], 'har': []}}
    for _, _, _, analysis, vrows, pred_by_ts, _, _ in gate_analyses:
        for r in vrows:
            if r.regime not in regime_pool:
                continue
            p = pred_by_ts.get(r.prediction_timestamp)
            if p is None or not math.isfinite(p):
                continue
            d = regime_pool[r.regime]
            a = r.actual_range / r.denom_close
            d['ext'].append(p / r.denom_close)
            d['actual'].append(a)
            if r.har_range is not None:
                d['har'].append(r.har_range / r.denom_close)
    regime_result = {}
    for reg, d in regime_pool.items():
        e_mae = _mae(d['ext'], d['actual'])
        h_mae = _mae(d['har'], d['actual'])
        regime_result[reg] = {'ext_norm_mae': e_mae, 'har_norm_mae': h_mae,
                              'ext_beats_har': (e_mae is not None and h_mae is not None
                                                and e_mae < h_mae),
                              'n': len(d['actual'])}
    c6 = sum(1 for d in regime_result.values() if d['ext_beats_har'] is True) >= 2
    c5 = (dm_ext_har['p_value'] is not None
          and dm_ext_har['p_value'] < PRIMARY_ALPHA
          and dm_ext_har['mean_loss_diff'] is not None
          and dm_ext_har['mean_loss_diff'] < 0)

    std_pred = _std(all_ext) if len(all_ext) > 1 else None
    std_act = _std(all_actual) if len(all_actual) > 1 else None
    pooled_dispersion = (std_pred / std_act) if (std_act is not None and std_act > 1e-12) else None
    pooled_bias = (_fmean(all_ext) / _fmean(all_actual)) \
        if (_fmean(all_ext) is not None and abs(_fmean(all_actual)) > 1e-12) else None

    coefficient_summary = None
    if all_beta:
        arr = np.vstack(all_beta)
        names = ['intercept'] + FEATURE_NAMES
        coefficient_summary = [
            {'coefficient': names[k], 'mean': float(arr[:, k].mean()),
             'std': float(arr[:, k].std())}
            for k in range(arr.shape[1])]

    per_asset_summary = {}
    for asset in sorted({r['asset'] for r in window_records}):
        recs = [r for r in window_records
                if r['asset'] == asset and not r['low_power']
                and r['timeframe'] in NONDAILY_TIMEFRAMES]
        per_asset_summary[asset] = {
            'n_primary_windows': len(recs),
            'nmae_wins': sum(1 for r in recs if r['ext_har_nmae_winner']),
            'nrmse_wins': sum(1 for r in recs if r['ext_har_nrmse_winner']),
            'raw_mae_wins': sum(1 for r in recs if r['ext_har_mae_winner']),
            'mean_nmae_improvement_pct': (
                _fmean([r['improvement_vs_har_nmae_pct'] for r in recs
                        if r['improvement_vs_har_nmae_pct'] is not None])),
        }

    gate = evaluate_derivatives_gate(window_records)
    if gate.get('overall') != 'pending':
        gate['criteria']['c5_statistical_support'] = bool(c5)
        gate['criteria']['c6_regime_breadth'] = bool(c6)
        gate['overall'] = 'pass' if all(v is True for v in gate['criteria'].values()) else 'fail'
        gate['verdict'] = classify_derivatives_gate(gate['criteria'])
        gate['verdict_meaning'] = VERDICT_MEANING[gate['verdict']]

    return {
        'kind': 'derivatives_volatility_f01',
        'configuration': config.asdict(),
        'model': {
            'form': 'HAR + linear funding extension (OLS on raw range)',
            'features': FEATURE_NAMES,
            'feature_descriptions': FEATURE_DESCRIPTIONS,
            'min_history': MIN_HISTORY,
            'refit_every': REFIT_EVERY,
            'alignment_rule': 'funding features use settled rates with '
                              'funding_time <= prediction timestamp over a 24h '
                              'window; missing/stale -> skip (no forward-fill)',
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
            'primary_comparison': 'funding extension vs HAR (normalized errors)',
            'dm': 'two-sided Diebold-Mariano, Newey-West HAC variance',
            'bootstrap': 'circular block bootstrap 95% CI',
            'wilcoxon': 'Wilcoxon signed-rank (robustness)',
            'multiple_testing': 'single primary comparison -> alpha=0.05',
        },
        'series': series_output,
        'window_records': window_records,
        'pooled_primary': pooled_primary,
        'extreme_sensitivity_dm_trimmed': dm_trim,
        'regime_pooled': regime_result,
        'per_asset_summary': per_asset_summary,
        'ext_adequacy': {
            'pooled_dispersion_ratio': pooled_dispersion,
            'pooled_bias_ratio': pooled_bias,
            'coefficient_summary': coefficient_summary,
        },
        'success_gate': gate,
        'notes': [
            'statistical significance is NOT trading profitability',
            'funding-only F-01: no OI, no basis, no liquidations',
            'no hyperparameter search, no feature selection, no tuning',
            'frozen single-asset HAR methodology is unchanged',
            'daily windows are supplementary and excluded from the gate',
        ],
    }
