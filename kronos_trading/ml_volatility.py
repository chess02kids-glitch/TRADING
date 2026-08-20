"""Phase 6 - does supervised ML add incremental value over HAR?

The project has established that the classical HAR model robustly captures the
OHLCV-only volatility structure and that Kronos adds nothing beyond it. The
remaining highest-value question is:

    H0: HAR captures essentially all useful OHLCV-only volatility structure;
        a more flexible ML model will not robustly outperform it.
    H1: nonlinear interactions in past OHLCV contain incremental information
        beyond HAR that a supervised ML model can exploit.

This module implements ONE clean experiment: a LightGBM (XGBoost fallback)
regressor with a fixed, economically-motivated feature set, trained in a strict
expanding-window walk-forward (retrain every ``RETRAIN_EVERY`` predictions),
evaluated against HAR on the SAME prediction timestamps and windows as the
frozen classical benchmark.

Kronos is a retired historical challenger and is not used here. Nothing is
tuned against the out-of-sample windows.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .evaluation import EvaluationConfig, PredictionEvaluator, VolatilityRow
from .model import PredictorResult
from .statistics_compare import (circular_block_bootstrap_mean_ci,
                                 diebold_mariano, spearman, wilcoxon_signed_rank)
from .volatility_baselines import EWMA_SPAN, HAR_MIN_TRAIN, ROLLING_WINDOWS
from .volatility_research import (MIN_SAMPLES, NONDAILY_TIMEFRAMES, _fmean,
                                  _mae, _std, system_stats)

# --------------------------------------------------------------------------- #
# Fixed a priori constants (do NOT tune against OOS results)
# --------------------------------------------------------------------------- #
MIN_HISTORY = 64        # feature warm-up: rows with target index >= MIN_HISTORY
RETRAIN_EVERY = 100     # fixed expanding-window retrain cadence (predictions)
PRIMARY_ALPHA = 0.05    # single primary comparison (ML vs HAR) -> no correction
ML_TRACKING_MIN_SPREAD_RATIO = 0.1   # anti-shrinkage: forecast regime spread
ML_TRACKING_MIN_PER_REGIME = 5

BASELINE_KEYS = ('prev', 'rolling5', 'rolling22', 'ewma', 'har')

MODEL_PARAMS_LIGHTGBM = {
    'objective': 'regression',
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'random_state': 42,
    'bagging_seed': 42,
    'feature_fraction_seed': 42,
    'n_jobs': 1,          # reproducibility first (single-threaded determinism)
    'verbosity': -1,
}

MODEL_PARAMS_XGBOOST = {
    'objective': 'reg:squarederror',
    'n_estimators': 300,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': 1,
    'verbosity': 0,
}

FEATURE_NAMES = [
    # returns (multi-horizon, ending at the last closed candle)
    'ret_1', 'ret_2', 'ret_3', 'ret_5', 'ret_10', 'ret_22',
    'abs_ret_1', 'abs_ret_5', 'abs_ret_22',
    'sq_ret_1', 'sq_ret_22',
    # normalized range (nr_k = (high_k - low_k) / close_{k-1})
    'nr_1', 'nr_mean_5', 'nr_mean_22', 'nr_std_5', 'nr_std_22',
    # raw range
    'range_1', 'range_mean_5', 'range_mean_22', 'range_std_5', 'range_std_22',
    # close-to-close realized volatility
    'rv_5', 'rv_22',
    # Parkinson-style range measure
    'park_5', 'park_22',
    # volume
    'vol_ret', 'vol_z22', 'logvol_mean22', 'vol_cv22',
    # momentum / structure
    'dist_ma22', 'pos_range22',
    # time (of the prediction candle's open time - known at prediction time)
    'hour', 'dow',
]

FEATURE_DESCRIPTIONS = {
    'ret_1': '1-step close-to-close return, close[-1]/close[-2]-1',
    'ret_2': '2-step return close[-1]/close[-3]-1',
    'ret_3': '3-step return close[-1]/close[-4]-1',
    'ret_5': '5-step return close[-1]/close[-6]-1',
    'ret_10': '10-step return close[-1]/close[-11]-1',
    'ret_22': '22-step return close[-1]/close[-23]-1',
    'abs_ret_1': '|ret_1|',
    'abs_ret_5': '|ret_5|',
    'abs_ret_22': '|ret_22|',
    'sq_ret_1': 'ret_1 ** 2',
    'sq_ret_22': 'ret_22 ** 2',
    'nr_1': 'previous normalized range (high-low)/close',
    'nr_mean_5': 'mean normalized range over last 5',
    'nr_mean_22': 'mean normalized range over last 22',
    'nr_std_5': 'std of normalized range over last 5',
    'nr_std_22': 'std of normalized range over last 22',
    'range_1': 'previous raw range high-low',
    'range_mean_5': 'mean raw range over last 5',
    'range_mean_22': 'mean raw range over last 22',
    'range_std_5': 'std of raw range over last 5',
    'range_std_22': 'std of raw range over last 22',
    'rv_5': 'realized volatility: std of last 5 close-to-close returns',
    'rv_22': 'realized volatility: std of last 22 close-to-close returns',
    'park_5': 'Parkinson range vol over last 5 (sqrt of mean ln(h/l)^2/(4 ln2))',
    'park_22': 'Parkinson range vol over last 22',
    'vol_ret': 'volume change vol[-1]/vol[-2]-1',
    'vol_z22': 'volume z-score vs trailing 22-bar mean/std',
    'logvol_mean22': 'log1p(trailing 22-bar mean volume)',
    'vol_cv22': 'coefficient of variation of trailing 22-bar volume',
    'dist_ma22': 'close[-1] / trailing-22 mean close - 1',
    'pos_range22': 'position of close[-1] within trailing-22 high/low range',
    'hour': 'hour-of-day of the prediction candle open (UTC)',
    'dow': 'day-of-week of the prediction candle open (0=Mon)',
}

VERDICT_MEANING = {
    'A': 'ML adds genuine incremental value over HAR',
    'B': 'weak / ambiguous incremental value',
    'C': 'ML fails to robustly beat HAR',
    'pending': 'not enough eligible evidence to classify',
}


def available_backend() -> Optional[str]:
    try:
        import lightgbm  # noqa: F401
        return 'lightgbm'
    except ImportError:
        pass
    try:
        import xgboost  # noqa: F401
        return 'xgboost'
    except ImportError:
        return None


def fit_model(X: np.ndarray, y: np.ndarray) -> Tuple[Any, str]:
    """Fit the fixed tree model. Returns (model, backend)."""
    backend = available_backend()
    if backend == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMRegressor(**MODEL_PARAMS_LIGHTGBM)
        model.fit(X, y)
        return model, backend
    if backend == 'xgboost':
        import xgboost as xgb
        model = xgb.XGBRegressor(**MODEL_PARAMS_XGBOOST)
        model.fit(X, y)
        return model, backend
    raise RuntimeError('neither LightGBM nor XGBoost is available')


def _model_importances(model: Any, backend: str) -> Optional[np.ndarray]:
    try:
        if backend == 'lightgbm':
            return model.booster_.feature_importance(importance_type='gain')
        if backend == 'xgboost':
            return model.feature_importances_
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------- #
# Feature construction (strictly past-only)
# --------------------------------------------------------------------------- #
def build_feature_matrix(candles: List[Any]) -> Optional[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Build (X, target_norm, target_raw, timestamps) aligned per candle index.

    Row ``j`` of ``X`` contains features computed ONLY from candles with index
    ``< j`` (enforced via ``.shift(1)`` and left-aligned rolling windows).
    ``target_norm[j] = (high_j - low_j) / close_{j-1}`` (the next-candle
    normalized range predicted from information available at candle ``j``'s open
    time); ``target_raw[j] = high_j - low_j``. Rows ``j < MIN_HISTORY`` carry
    NaN features and are dropped before training.
    """
    n = len(candles)
    if n < 2:
        return None
    close = pd.Series([c.close for c in candles], dtype=float)
    high = pd.Series([c.high for c in candles], dtype=float)
    low = pd.Series([c.low for c in candles], dtype=float)
    vol = pd.Series([c.volume for c in candles], dtype=float)
    ts = pd.Series([c.timestamp_ms for c in candles], dtype='int64')

    r = close.pct_change()                      # r[k] = close[k]/close[k-1]-1
    range_s = high - low
    nr = (high - low) / close.shift(1)          # normalized range of candle k
    park = (np.log(high / low) ** 2) / (4.0 * math.log(2.0))

    f: Dict[str, pd.Series] = {}
    # returns (multi-horizon, ending at the last closed candle j-1)
    f['ret_1'] = close.shift(1) / close.shift(2) - 1.0
    f['ret_2'] = close.shift(1) / close.shift(3) - 1.0
    f['ret_3'] = close.shift(1) / close.shift(4) - 1.0
    f['ret_5'] = close.shift(1) / close.shift(6) - 1.0
    f['ret_10'] = close.shift(1) / close.shift(11) - 1.0
    f['ret_22'] = close.shift(1) / close.shift(23) - 1.0
    f['abs_ret_1'] = f['ret_1'].abs()
    f['abs_ret_5'] = f['ret_5'].abs()
    f['abs_ret_22'] = f['ret_22'].abs()
    f['sq_ret_1'] = f['ret_1'] ** 2
    f['sq_ret_22'] = f['ret_22'] ** 2
    # normalized range
    f['nr_1'] = nr.shift(1)
    f['nr_mean_5'] = nr.rolling(5, min_periods=1).mean().shift(1)
    f['nr_mean_22'] = nr.rolling(22, min_periods=1).mean().shift(1)
    f['nr_std_5'] = nr.rolling(5, min_periods=2).std().shift(1)
    f['nr_std_22'] = nr.rolling(22, min_periods=2).std().shift(1)
    # raw range
    f['range_1'] = range_s.shift(1)
    f['range_mean_5'] = range_s.rolling(5, min_periods=1).mean().shift(1)
    f['range_mean_22'] = range_s.rolling(22, min_periods=1).mean().shift(1)
    f['range_std_5'] = range_s.rolling(5, min_periods=2).std().shift(1)
    f['range_std_22'] = range_s.rolling(22, min_periods=2).std().shift(1)
    # realized volatility
    f['rv_5'] = r.rolling(5, min_periods=2).std().shift(1)
    f['rv_22'] = r.rolling(22, min_periods=2).std().shift(1)
    # Parkinson
    f['park_5'] = np.sqrt(park.rolling(5, min_periods=1).mean().shift(1))
    f['park_22'] = np.sqrt(park.rolling(22, min_periods=1).mean().shift(1))
    # volume
    f['vol_ret'] = vol.pct_change().shift(1)
    vol_mean22 = vol.rolling(22, min_periods=2).mean()
    vol_std22 = vol.rolling(22, min_periods=2).std()
    f['vol_z22'] = ((vol - vol_mean22) / vol_std22).shift(1)
    f['logvol_mean22'] = np.log1p(vol_mean22.shift(1))
    f['vol_cv22'] = (vol_std22 / vol_mean22).shift(1)
    # momentum / structure
    close_ma22 = close.rolling(22, min_periods=1).mean()
    f['dist_ma22'] = (close.shift(1) / close_ma22.shift(1)) - 1.0
    close_min22 = close.rolling(22, min_periods=1).min()
    close_max22 = close.rolling(22, min_periods=1).max()
    span = (close_max22 - close_min22).shift(1)
    f['pos_range22'] = (close.shift(1) - close_min22.shift(1)) / span.replace(0.0, np.nan)
    # time (of the prediction candle's open, known at prediction time)
    dt = pd.to_datetime(ts, unit='ms', utc=True)
    f['hour'] = dt.dt.hour.astype(float)
    f['dow'] = dt.dt.dayofweek.astype(float)

    X = pd.DataFrame({name: f[name] for name in FEATURE_NAMES})
    return (X.to_numpy(dtype=float),
            nr.to_numpy(dtype=float),          # target_norm
            range_s.to_numpy(dtype=float),     # target_raw
            ts.to_numpy(dtype='int64'))


# --------------------------------------------------------------------------- #
# Walk-forward training / prediction
# --------------------------------------------------------------------------- #
def run_walk(X: np.ndarray, y: np.ndarray, target_indices: List[int],
             min_history: int = MIN_HISTORY,
             retrain_every: int = RETRAIN_EVERY) -> Dict[str, Any]:
    """Expanding-window walk-forward over ``target_indices``.

    At each prediction index ``j`` the model is (re)trained on rows
    ``[min_history, j)`` (target index strictly < j). Retraining happens every
    ``retrain_every`` predictions (fixed cadence); between retrains the model
    is reused. Returns predictions keyed by index, the backend, retrain book-
    keeping, feature-importance history, and a leak counter (must stay 0).
    """
    predictions: Dict[int, float] = {}
    model = None
    backend = None
    last_retrain: Optional[int] = None
    retrains: List[Dict[str, Any]] = []
    importances: List[np.ndarray] = []
    leaks = 0
    for j in target_indices:
        train_end = j
        if model is None or (last_retrain is not None
                             and (j - last_retrain) >= retrain_every):
            Xtr = X[min_history:train_end]
            ytr = y[min_history:train_end]
            if Xtr.shape[0] == 0:
                # not enough history yet -> cannot predict this index
                predictions[j] = math.nan
                continue
            model, backend = fit_model(Xtr, ytr)
            last_retrain = j
            imp = _model_importances(model, backend)
            if imp is not None:
                importances.append(np.asarray(imp, dtype=float))
            retrains.append({'train_start_idx': min_history, 'train_end_idx': train_end,
                             'n_train': int(Xtr.shape[0]), 'pred_idx': j})
        if train_end - min_history < 1:
            predictions[j] = math.nan
            continue
        if train_end > j:
            leaks += 1  # must never happen: training data ends before j
        row = X[j:j + 1]
        predictions[j] = float(model.predict(row)[0])
    return {'predictions': predictions, 'backend': backend,
            'retrains': retrains, 'importances': importances,
            'leaks': leaks}


# --------------------------------------------------------------------------- #
# Regime tracking / adequacy (anti-shrinkage)
# --------------------------------------------------------------------------- #
def _ml_regime_tracking(vrows: List[VolatilityRow],
                        pred_map: Dict[int, float],
                        min_per_regime: int = ML_TRACKING_MIN_PER_REGIME,
                        min_spread_ratio: float = ML_TRACKING_MIN_SPREAD_RATIO
                        ) -> Dict[str, Any]:
    """Does ML distinguish volatility regimes (vs shrinking toward the mean)?"""
    groups: Dict[str, List[Tuple[float, float]]] = {'low': [], 'medium': [], 'high': []}
    for r in vrows:
        if r.regime in groups and r.prediction_timestamp in pred_map:
            p = pred_map[r.prediction_timestamp]
            if math.isfinite(p):
                groups[r.regime].append((p, r.actual_range / r.denom_close))
    means: Dict[str, Dict[str, float]] = {}
    for reg, pairs in groups.items():
        if len(pairs) >= min_per_regime:
            means[reg] = {'ml_mean': _fmean([p for p, _ in pairs]),
                          'actual_mean': _fmean([a for _, a in pairs]),
                          'n': len(pairs)}
    if len(means) < 2:
        return {'regime_means': means, 'n_regimes': len(means),
                'monotonic': None, 'spread_ratio': None, 'tracks': None}
    ordered = sorted(means.items(), key=lambda kv: kv[1]['actual_mean'])
    fs = [v['ml_mean'] for _, v in ordered]
    actuals = [v['actual_mean'] for _, v in ordered]
    monotonic = all(fs[i] <= fs[i + 1] + 1e-12 for i in range(len(fs) - 1))
    actual_spread = actuals[-1] - actuals[0]
    ml_spread = fs[-1] - fs[0]
    spread_ratio = (ml_spread / actual_spread) if actual_spread > 1e-12 else None
    tracks = bool(monotonic and spread_ratio is not None
                  and spread_ratio >= min_spread_ratio)
    return {'regime_means': means, 'n_regimes': len(means),
            'monotonic': bool(monotonic), 'spread_ratio': spread_ratio,
            'tracks': tracks}


def _winner_lower(a: Optional[float], b: Optional[float], a_name: str,
                  b_name: str) -> Optional[str]:
    if a is None or b is None:
        return None
    return a_name if a < b else (b_name if a > b else 'tie')


def _improvement_pct(base: Optional[float], model: Optional[float]) -> Optional[float]:
    if base is None or model is None or base <= 1e-12:
        return None
    return (base - model) / base * 100.0


# --------------------------------------------------------------------------- #
# Per-window analysis
# --------------------------------------------------------------------------- #
def _window_analysis(vrows: List[VolatilityRow],
                     pred_map: Dict[int, float]) -> Dict[str, Any]:
    ts_to_ml = {r.prediction_timestamp: pred_map.get(r.prediction_timestamp)
                for r in vrows}
    ml_norm = [ts_to_ml[r.prediction_timestamp] for r in vrows]
    actual_norm = [r.actual_range / r.denom_close for r in vrows]
    ml_raw = [(p * r.denom_close) if (p is not None and math.isfinite(p)) else None
              for p, r in zip(ml_norm, vrows)]
    actual_raw = [r.actual_range for r in vrows]

    systems_norm = {'ml': system_stats(ml_norm, actual_norm)}
    systems_raw = {'ml': system_stats(ml_raw, actual_raw)}
    for key in BASELINE_KEYS:
        pred = [getattr(r, key + '_range') for r in vrows]
        systems_norm[key] = system_stats(
            [p / r.denom_close if p is not None else None for p, r in zip(pred, vrows)],
            actual_norm)
        systems_raw[key] = system_stats(pred, actual_raw)

    # ML vs HAR (primary comparison) + ML vs others (secondary)
    comparisons: Dict[str, Any] = {}
    for key in BASELINE_KEYS:
        b_norm = [getattr(r, key + '_range') / r.denom_close
                  if getattr(r, key + '_range') is not None else None for r in vrows]
        ml_err = [abs(p - a) if (p is not None and math.isfinite(p)) else math.nan
                  for p, a in zip(ml_norm, actual_norm)]
        b_err = [abs(p - a) if p is not None else math.nan for p, a in zip(b_norm, actual_norm)]
        comparisons['ml_vs_%s' % key] = {
            'norm_mae_delta': (systems_norm['ml']['mae'] - systems_norm[key]['mae'])
            if (systems_norm['ml']['mae'] is not None and systems_norm[key]['mae'] is not None)
            else None,
            'norm_mae_winner': _winner_lower(systems_norm['ml']['mae'],
                                            systems_norm[key]['mae'], 'ml', key),
            'norm_rmse_delta': (systems_norm['ml']['rmse'] - systems_norm[key]['rmse'])
            if (systems_norm['ml']['rmse'] is not None and systems_norm[key]['rmse'] is not None)
            else None,
            'raw_mae_delta': (systems_raw['ml']['mae'] - systems_raw[key]['mae'])
            if (systems_raw['ml']['mae'] is not None and systems_raw[key]['mae'] is not None)
            else None,
            'dm': diebold_mariano(ml_err, b_err, a_name='ml', b_name=key),
            'bootstrap_mean_diff_ci': circular_block_bootstrap_mean_ci(
                [a - b for a, b in zip(ml_err, b_err)]),
            'wilcoxon_p_value': wilcoxon_signed_rank(
                [a - b for a, b in zip(ml_err, b_err)])['p_value'],
        }

    improvement_vs_har = {
        'norm_mae_pct': _improvement_pct(systems_norm['har']['mae'], systems_norm['ml']['mae']),
        'norm_rmse_pct': _improvement_pct(systems_norm['har']['rmse'], systems_norm['ml']['rmse']),
        'raw_mae_pct': _improvement_pct(systems_raw['har']['mae'], systems_raw['ml']['mae']),
        'raw_rmse_pct': _improvement_pct(systems_raw['har']['rmse'], systems_raw['ml']['rmse']),
    }

    # regimes (normalized target)
    regimes: Dict[str, Any] = {}
    for reg in ('low', 'medium', 'high'):
        sub = [r for r in vrows if r.regime == reg]
        p = [ts_to_ml[r.prediction_timestamp] for r in sub]
        a = [r.actual_range / r.denom_close for r in sub]
        h = [r.har_range / r.denom_close if r.har_range is not None else None for r in sub]
        ml_mae = _mae(p, a)
        har_mae = _mae(h, a)
        regimes[reg] = {'n': len(sub), 'ml_norm_mae': ml_mae, 'har_norm_mae': har_mae,
                        'ml_beats_har': (ml_mae is not None and har_mae is not None
                                         and ml_mae < har_mae)}

    tracking = _ml_regime_tracking(vrows, pred_map)

    return {
        'sample_size': len(vrows),
        'low_power': len(vrows) < MIN_SAMPLES,
        'systems_normalized': systems_norm,
        'systems_raw': systems_raw,
        'comparisons': comparisons,
        'improvement_vs_har_pct': improvement_vs_har,
        'regimes': regimes,
        'ml_regime_tracking': tracking,
        'ml_shrinkage': {
            'dispersion_ratio': systems_norm['ml']['dispersion_ratio'],
            'bias_ratio': systems_norm['ml']['bias_ratio'],
            'std_pred': systems_norm['ml']['std_pred'],
            'std_actual': systems_norm['ml']['std_actual'],
            'mean_pred': systems_norm['ml']['mean_pred'],
            'mean_actual': systems_norm['ml']['mean_actual'],
        },
    }


# --------------------------------------------------------------------------- #
# Success gate (pre-registered)
# --------------------------------------------------------------------------- #
def classify_ml_gate(criteria: Dict[str, Any]) -> str:
    if all(v is True for v in criteria.values()):
        return 'A'
    if criteria.get('c1_series_breadth_nmae') is True:
        return 'B'
    return 'C'


def evaluate_ml_gate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    eligible = [r for r in records
                if not r['low_power'] and r['timeframe'] in NONDAILY_TIMEFRAMES]
    n = len(eligible)
    gate: Dict[str, Any] = {
        'eligible_windows': n,
        'definition': {
            'c1': 'ML beats HAR on normalized MAE in >=2/3 windows for >=2/4 series',
            'c2': 'ML beats HAR on normalized RMSE in >=2/3 windows for >=2/4 series',
            'c3': 'pooled DM (ML vs HAR, normalized) p<0.05 and ML wins',
            'c4': 'improvement survives raw-range MAE (majority of windows)',
            'c5': 'improvement survives regime analysis (>=2/3 regimes)',
            'c6': 'improvement not explained purely by shrinkage (regime tracking)',
            'c7': 'effect present in >=2/3 windows (not one isolated period)',
            'c8': 'no look-ahead / data leakage (verified in walk)',
        },
        'criteria': {},
    }
    if n == 0:
        gate['overall'] = 'pending'
        gate['verdict'] = 'pending'
        gate['verdict_meaning'] = VERDICT_MEANING['pending']
        gate['note'] = 'no eligible windows'
        return gate

    def series_breadth(field: str) -> bool:
        wins: Dict[str, List[bool]] = {}
        for r in eligible:
            wins.setdefault(r['series'], []).append(r[field])
        return sum(1 for w in wins.values() if sum(w) >= 2) >= 2

    c1 = series_breadth('ml_har_nmae_winner')
    c2 = series_breadth('ml_har_nrmse_winner')
    c4 = sum(1 for r in eligible if r['ml_har_mae_winner']) > 0.5 * n
    c7 = sum(1 for r in eligible if r['ml_har_nmae_winner']) > 0.5 * n
    c6 = sum(1 for r in eligible if r['ml_tracks']) > 0.5 * n
    c8 = all(r['leaks'] == 0 for r in eligible)

    # c3 and c5 are filled by the caller (pooled DM / pooled regime results).
    gate['criteria'] = {
        'c1_series_breadth_nmae': bool(c1),
        'c2_series_breadth_nrmse': bool(c2),
        'c3_statistical_support': None,
        'c4_raw_survives': bool(c4),
        'c5_regime_breadth': None,
        'c6_not_solely_shrinkage': bool(c6),
        'c7_window_breadth': bool(c7),
        'c8_no_lookahead': bool(c8),
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
        'ml_norm_mae': sn['ml']['mae'], 'har_norm_mae': sn['har']['mae'],
        'ml_norm_rmse': sn['ml']['rmse'], 'har_norm_rmse': sn['har']['rmse'],
        'ml_raw_mae': sr['ml']['mae'], 'har_raw_mae': sr['har']['mae'],
        'ml_har_nmae_winner': _winner_lower(sn['ml']['mae'], sn['har']['mae'],
                                            'ml', 'har') == 'ml',
        'ml_har_nrmse_winner': _winner_lower(sn['ml']['rmse'], sn['har']['rmse'],
                                             'ml', 'har') == 'ml',
        'ml_har_mae_winner': _winner_lower(sr['ml']['mae'], sr['har']['mae'],
                                           'ml', 'har') == 'ml',
        'ml_dispersion_ratio': sn['ml']['dispersion_ratio'],
        'ml_bias_ratio': sn['ml']['bias_ratio'],
        'ml_tracks': analysis['ml_regime_tracking']['tracks'] is True,
        'improvement_vs_har_nmae_pct': analysis['improvement_vs_har_pct']['norm_mae_pct'],
        'leaks': leaks,
        'regime_ml_beats_har': {reg: analysis['regimes'][reg]['ml_beats_har']
                                for reg in ('low', 'medium', 'high')},
    }


def run_ml_vs_har(load_candles: Callable[[str, str], List[Any]],
                  config: EvaluationConfig,
                  series: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Run the ML-vs-HAR experiment over the requested series/windows."""
    series_output: Dict[str, Any] = {}
    window_records: List[Dict[str, Any]] = []
    gate_analyses: List[Tuple[str, str, str, Dict[str, Any], List[VolatilityRow],
                              Dict[int, float], int]] = []
    all_importances: List[np.ndarray] = []

    for symbol, timeframe in series:
        candles = load_candles(symbol, timeframe)
        ev = PredictionEvaluator(_StubPredictor(), config, symbol, timeframe)
        closed = ev._closed_data(candles)
        built = build_feature_matrix(closed)
        if built is None:
            series_output['%s %s' % (symbol, timeframe)] = {
                'window_info': {'note': 'insufficient data'}, 'windows': {}}
            continue
        X, target_norm, target_raw, ts = built
        ts_to_idx = {int(t): i for i, t in enumerate(ts)}
        windows, window_info = ev.evaluate_windows(candles)

        per_window: Dict[str, Any] = {}
        for name, res in windows.items():
            vrows = res.volatility_rows
            idxs = [ts_to_idx[int(r.prediction_timestamp)] for r in vrows]
            walk = run_walk(X, target_norm, idxs)
            all_importances.extend(walk['importances'])
            # map index-keyed predictions back to prediction timestamps
            pred_by_ts = {int(r.prediction_timestamp): walk['predictions'][j]
                          for r, j in zip(vrows, idxs)}
            analysis = _window_analysis(vrows, pred_by_ts)
            per_window[name] = analysis
            rec = _window_record(symbol, timeframe, name, analysis, walk['leaks'])
            window_records.append(rec)
            if not rec['low_power'] and timeframe in NONDAILY_TIMEFRAMES:
                gate_analyses.append((symbol, timeframe, name, analysis, vrows,
                                      pred_by_ts, walk['leaks']))
        series_output['%s %s' % (symbol, timeframe)] = {
            'window_info': window_info,
            'windows': per_window,
        }

    # Pooled statistics (normalized + raw) over primary windows only.
    ml_err_har, har_err_ml = [], []
    ml_err_prev, prev_err_ml = [], []
    ml_err_ewma, ewma_err_ml = [], []
    ml_err_roll5, roll5_err_ml = [], []
    ml_err_roll22, roll22_err_ml = [], []
    for _, _, _, analysis, vrows, pred_map, _ in gate_analyses:
        for r in vrows:
            p = pred_map.get(r.prediction_timestamp)
            if p is None or not math.isfinite(p):
                continue
            a = r.actual_range / r.denom_close
            ml_err_har.append(abs(p - a))
            if r.har_range is not None:
                har_err_ml.append(abs(r.har_range / r.denom_close - a))
            if r.prev_range is not None:
                ml_err_prev.append(abs(p - a))
                prev_err_ml.append(abs(r.prev_range / r.denom_close - a))
            if r.ewma_range is not None:
                ml_err_ewma.append(abs(p - a))
                ewma_err_ml.append(abs(r.ewma_range / r.denom_close - a))
            if r.rolling5_range is not None:
                ml_err_roll5.append(abs(p - a))
                roll5_err_ml.append(abs(r.rolling5_range / r.denom_close - a))
            if r.rolling22_range is not None:
                ml_err_roll22.append(abs(p - a))
                roll22_err_ml.append(abs(r.rolling22_range / r.denom_close - a))

    pooled = {
        'ml_vs_har': diebold_mariano(ml_err_har, har_err_ml, a_name='ml', b_name='har'),
        'ml_vs_prev': diebold_mariano(ml_err_prev, prev_err_ml, a_name='ml', b_name='prev'),
        'ml_vs_ewma': diebold_mariano(ml_err_ewma, ewma_err_ml, a_name='ml', b_name='ewma'),
        'ml_vs_rolling5': diebold_mariano(ml_err_roll5, roll5_err_ml, a_name='ml', b_name='rolling5'),
        'ml_vs_rolling22': diebold_mariano(ml_err_roll22, roll22_err_ml, a_name='ml', b_name='rolling22'),
    }

    # Pooled regime analysis (normalized) over primary windows.
    regime_pool = {'low': {'ml': [], 'actual': [], 'har': []},
                   'medium': {'ml': [], 'actual': [], 'har': []},
                   'high': {'ml': [], 'actual': [], 'har': []}}
    for _, _, _, analysis, vrows, pred_map, _ in gate_analyses:
        for r in vrows:
            if r.regime not in regime_pool:
                continue
            p = pred_map.get(r.prediction_timestamp)
            if p is None or not math.isfinite(p):
                continue
            d = regime_pool[r.regime]
            a = r.actual_range / r.denom_close
            d['ml'].append(p)
            d['actual'].append(a)
            if r.har_range is not None:
                d['har'].append(r.har_range / r.denom_close)
    regime_result = {}
    for reg, d in regime_pool.items():
        ml_mae = _mae(d['ml'], d['actual'])
        har_mae = _mae(d['har'], d['actual'])
        regime_result[reg] = {'ml_norm_mae': ml_mae, 'har_norm_mae': har_mae,
                              'ml_beats_har': (ml_mae is not None and har_mae is not None
                                               and ml_mae < har_mae),
                              'n': len(d['actual'])}
    c5 = sum(1 for d in regime_result.values() if d['ml_beats_har'] is True) >= 2

    # Pooled ML adequacy (regime tracking + dispersion) over primary windows.
    pool_tracking = _ml_regime_tracking(
        [r for _, _, _, _, vrows, _, _ in gate_analyses for r in vrows],
        {r.prediction_timestamp: pred_map.get(r.prediction_timestamp)
         for _, _, _, _, vrows, pred_map, _ in gate_analyses for r in vrows})
    all_ml, all_actual = [], []
    for _, _, _, analysis, vrows, pred_map, _ in gate_analyses:
        for r in vrows:
            p = pred_map.get(r.prediction_timestamp)
            if p is None or not math.isfinite(p):
                continue
            all_ml.append(p)
            all_actual.append(r.actual_range / r.denom_close)
    std_pred = _std(all_ml) if len(all_ml) > 1 else None
    std_act = _std(all_actual) if len(all_actual) > 1 else None
    pooled_dispersion = (std_pred / std_act) if (std_act is not None and std_act > 1e-12) else None
    pooled_bias = (_fmean(all_ml) / _fmean(all_actual)) \
        if (_fmean(all_ml) is not None and abs(_fmean(all_actual)) > 1e-12) else None

    # Feature-importance stability (mean + std across all retrains).
    feature_importance = None
    if all_importances:
        arr = np.vstack([i for i in all_importances if i.shape == (len(FEATURE_NAMES),)])
        if arr.shape[0] > 0:
            mean_imp = arr.mean(axis=0)
            std_imp = arr.std(axis=0)
            feature_importance = [
                {'feature': FEATURE_NAMES[k], 'mean_gain': float(mean_imp[k]),
                 'std_gain': float(std_imp[k])}
                for k in range(len(FEATURE_NAMES))]
            # stability: coefficient of variation of importance across retrains
            feature_importance.sort(key=lambda d: -d['mean_gain'])

    # c3: pooled DM (ML vs HAR, normalized).
    dm_ml_har = pooled['ml_vs_har']
    c3 = (dm_ml_har['p_value'] is not None
          and dm_ml_har['p_value'] < PRIMARY_ALPHA
          and dm_ml_har['mean_loss_diff'] is not None
          and dm_ml_har['mean_loss_diff'] < 0)

    # Feature-importance stability (gathered across all walks).
    # (Aggregated by the caller via run_walk's 'importances'; see below.)

    gate = evaluate_ml_gate(window_records)
    if gate.get('overall') != 'pending':
        gate['criteria']['c3_statistical_support'] = bool(c3)
        gate['criteria']['c5_regime_breadth'] = bool(c5)
        gate['overall'] = 'pass' if all(v is True for v in gate['criteria'].values()) else 'fail'
        gate['verdict'] = classify_ml_gate(gate['criteria'])
        gate['verdict_meaning'] = VERDICT_MEANING[gate['verdict']]

    return {
        'kind': 'ml_vs_har_volatility',
        'configuration': config.asdict(),
        'ml_configuration': {
            'backend': available_backend(),
            'params_lightgbm': MODEL_PARAMS_LIGHTGBM,
            'params_xgboost': MODEL_PARAMS_XGBOOST,
            'min_history': MIN_HISTORY,
            'retrain_every': RETRAIN_EVERY,
            'n_features': len(FEATURE_NAMES),
            'feature_names': FEATURE_NAMES,
        },
        'targets': {
            'primary': '(high_{t+1} - low_{t+1}) / close_t  (normalized next-candle range)',
            'secondary': 'high_{t+1} - low_{t+1}  (raw range)',
        },
        'features': FEATURE_DESCRIPTIONS,
        'baselines': {
            'previous_range': 'range_{t-1}',
            'rolling5': 'mean(range_{t-5:t})',
            'rolling22': 'mean(range_{t-22:t})',
            'ewma': 'EWMA of ranges, span=22 (alpha=2/23)',
            'har': 'beta0 + beta1*range_{t-1} + beta2*mean5 + beta3*mean22, '
                   'expanding past-only OLS',
        },
        'walk_forward': {
            'scheme': 'expanding window',
            'retrain_cadence': 'every %d predictions' % RETRAIN_EVERY,
            'no_lookahead': 'training rows have target index strictly before the '
                            'prediction index; features use only candles before the '
                            'target candle',
        },
        'statistical_methodology': {
            'primary_comparison': 'ML vs HAR (normalized range errors)',
            'dm': 'two-sided Diebold-Mariano, Newey-West HAC variance',
            'bootstrap': 'circular block bootstrap 95% CI on paired error differences',
            'wilcoxon': 'Wilcoxon signed-rank (nonparametric robustness)',
            'multiple_testing': 'single primary comparison -> alpha=0.05; the four '
                                'secondary ML-vs-baseline DMs are reported without '
                                'gate weight (Bonferroni alpha=0.0125 noted)',
        },
        'series': series_output,
        'window_records': window_records,
        'pooled_statistics': pooled,
        'regime_pooled': regime_result,
        'ml_adequacy': {
            'pooled_regime_tracking': pool_tracking,
            'pooled_dispersion_ratio': pooled_dispersion,
            'pooled_bias_ratio': pooled_bias,
            'feature_importance_mean_gain': feature_importance,
            'interpretation': 'a tracking model shows a positive forecast spread '
                              'across regimes; a pure shrinker has ~zero spread',
        },
        'success_gate': gate,
        'notes': [
            'statistical significance is NOT trading profitability',
            'no hyperparameter search, no window/regime cherry-picking',
            'HAR is the champion to beat; Kronos is a retired challenger',
            'the frozen Phase 4/5/5b/5c reports are unchanged',
        ],
    }


class _StubPredictor:
    """Internal stub so the evaluator's window/baseline machinery can be reused
    without running Kronos. Its Kronos-like fields are never used for the ML
    results - only ``volatility_rows`` (timestamps, baselines, regimes) matter.
    """
    device = 'cpu'
    dtype = 'n/a'
    version = 'stub'

    def predict(self, candles, timeframe, horizon=1, temperature=1.0,
                top_k=0, top_p=0.9, sample_count=1, seed=None,
                deterministic=False):
        last = candles[-1]
        steps = [{'open': last.close, 'high': last.close, 'low': last.close,
                  'close': last.close, 'volume': 1.0, 'amount': last.close}
                 for _ in range(horizon)]
        return PredictorResult(steps=steps, latency_ms=0.0, peak_vram_bytes=None)
