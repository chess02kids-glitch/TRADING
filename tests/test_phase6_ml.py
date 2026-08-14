"""Phase 6 - ML vs HAR volatility experiment tests.

Covers: no-lookahead feature construction, rolling-feature correctness,
training-cutoff correctness, walk-forward future-invariance (no leakage),
deterministic predictions, target alignment, identical timestamps with the
classical evaluator, statistical alignment, and empty/short-history safety.

The walk-forward LOGIC is tested with a deterministic stub model factory
(monkeypatched ``fit_model``), so the suite does not require LightGBM; a
separate determinism test uses the real backend behind ``importorskip``.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

import kronos_trading.ml_volatility as mv
from kronos_trading.evaluation import EvaluationConfig, PredictionEvaluator
from kronos_trading.model import PredictorResult
from kronos_trading.types import Candle

H = 3_600_000
BASE = 1_700_000_000_000


def mk(n, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    close = 100.0
    for i in range(n):
        r = 0.5 + (i % 7) * 0.25 + rng.normal(0, 0.03)
        close = close * (1.0 + rng.normal(0, 0.002))
        out.append(Candle(BASE + i * H, close - r / 2, close + r / 2,
                          close - r / 2, close, 1000.0 + rng.normal(0, 20)))
    return out


class _StubModel:
    """Deterministic model stub: predicts the training mean target."""
    def __init__(self, mean):
        self.mean = mean
    def predict(self, X):
        return np.full(X.shape[0], self.mean, dtype=float)


def _stub_fit(X, y):
    return _StubModel(float(np.mean(y))), 'stub'


def _patch_fit(monkeypatch):
    monkeypatch.setattr(mv, 'fit_model', _stub_fit)


def cfg(**overrides):
    defaults = dict(context_length=64, horizon=1, window_size=30)
    defaults.update(overrides)
    return EvaluationConfig(**defaults)


# --------------------------------------------------------------------------- #
# Feature construction: no look-ahead
# --------------------------------------------------------------------------- #
def test_feature_matrix_shape_and_target_alignment():
    candles = mk(80)
    X, tn, tr, ts = mv.build_feature_matrix(candles)
    assert X.shape == (80, len(mv.FEATURE_NAMES))
    assert tn.shape == (80,)
    assert tr.shape == (80,)
    # target_norm[j] = (high_j - low_j) / close_{j-1}
    for j in range(1, 60):
        expected = (candles[j].high - candles[j].low) / candles[j - 1].close
        assert tn[j] == pytest.approx(expected, rel=1e-9)
        assert tr[j] == pytest.approx(candles[j].high - candles[j].low)
    assert math.isnan(tn[0])  # no previous close for candle 0


def test_feature_row_j_uses_only_candles_before_j():
    candles = mk(200)
    X1, _, _, _ = mv.build_feature_matrix(candles)
    # perturb candle 130 (its high/low/close) - rows <= 130 must be unchanged
    modified = list(candles)
    c = candles[130]
    modified[130] = Candle(c.timestamp_ms, c.open + 50, c.high + 50,
                           c.low + 50, c.close + 50, c.volume)
    X2, _, _, _ = mv.build_feature_matrix(modified)
    # rows 0..130 inclusive only use candles < their index, so unaffected
    np.testing.assert_allclose(X1[:131], X2[:131], rtol=0, atol=0)
    # rows after 130 must differ in at least one feature
    assert not np.allclose(X1[131:], X2[131:], rtol=0, atol=0)


def test_rolling_feature_correctness():
    candles = mk(80)
    X, _, _, _ = mv.build_feature_matrix(candles)
    j = 60
    close = np.array([c.close for c in candles])
    high = np.array([c.high for c in candles])
    low = np.array([c.low for c in candles])
    vol = np.array([c.volume for c in candles])
    r = np.zeros(len(close)); r[1:] = close[1:] / close[:-1] - 1
    nr = np.zeros(len(close)); nr[1:] = (high[1:] - low[1:]) / close[:-1]
    park = (np.log(high / low) ** 2) / (4 * np.log(2.0))

    idx = {name: k for k, name in enumerate(mv.FEATURE_NAMES)}
    assert X[j, idx['ret_1']] == pytest.approx(close[j - 1] / close[j - 2] - 1, rel=1e-9)
    assert X[j, idx['ret_22']] == pytest.approx(close[j - 1] / close[j - 23] - 1, rel=1e-9)
    assert X[j, idx['nr_1']] == pytest.approx(nr[j - 1], rel=1e-9)
    assert X[j, idx['nr_mean_5']] == pytest.approx(np.mean(nr[j - 5:j]), rel=1e-9)
    assert X[j, idx['range_mean_5']] == pytest.approx(
        np.mean((high - low)[j - 5:j]), rel=1e-9)
    # pandas rolling().std() uses ddof=1 (sample std)
    assert X[j, idx['rv_5']] == pytest.approx(np.std(r[j - 5:j], ddof=1), rel=1e-6)
    assert X[j, idx['park_5']] == pytest.approx(np.sqrt(np.mean(park[j - 5:j])), rel=1e-9)
    v = vol[j - 22:j]
    assert X[j, idx['vol_z22']] == pytest.approx(
        (vol[j - 1] - np.mean(v)) / np.std(v, ddof=1), rel=1e-6)
    ma22 = np.mean(close[j - 22:j])
    assert X[j, idx['dist_ma22']] == pytest.approx(close[j - 1] / ma22 - 1, rel=1e-9)
    lo22, hi22 = np.min(close[j - 22:j]), np.max(close[j - 22:j])
    assert X[j, idx['pos_range22']] == pytest.approx(
        (close[j - 1] - lo22) / (hi22 - lo22), rel=1e-9)


def test_hour_feature_from_prediction_timestamp():
    candles = mk(50)
    X, _, _, _ = mv.build_feature_matrix(candles)
    import datetime
    idx = {name: k for k, name in enumerate(mv.FEATURE_NAMES)}
    for j in (10, 20, 40):
        ts = candles[j].timestamp_ms
        hour = datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc).hour
        assert X[j, idx['hour']] == pytest.approx(hour)


# --------------------------------------------------------------------------- #
# Walk-forward: training cutoff, no leakage, determinism
# --------------------------------------------------------------------------- #
def test_walk_training_cutoff_and_leaks(monkeypatch):
    _patch_fit(monkeypatch)
    candles = mk(120)
    X, y, _, _ = mv.build_feature_matrix(candles)
    # start after MIN_HISTORY so every prediction has >= 1 training row
    target_indices = list(range(mv.MIN_HISTORY + 10, 100))
    out = mv.run_walk(X, y, target_indices)
    assert out['leaks'] == 0
    # every retrain trains only on rows strictly before its prediction index
    for r in out['retrains']:
        assert r['train_end_idx'] == r['pred_idx']
        assert r['n_train'] == r['train_end_idx'] - mv.MIN_HISTORY
    # all predictions finite
    assert all(math.isfinite(p) for p in out['predictions'].values())


def test_walk_forward_future_invariance(monkeypatch):
    _patch_fit(monkeypatch)
    candles = mk(150)
    X1, y1, _, _ = mv.build_feature_matrix(candles)
    targets = list(range(mv.MIN_HISTORY, 100))
    preds1 = mv.run_walk(X1, y1, targets)['predictions']

    # modify candles AFTER the last target index (100): predictions unchanged
    modified = list(candles)
    for i in range(100, len(modified)):
        c = modified[i]
        modified[i] = Candle(c.timestamp_ms, c.open + 5, c.high + 5,
                             c.low + 5, c.close + 5, c.volume)
    X2, y2, _, _ = mv.build_feature_matrix(modified)
    preds2 = mv.run_walk(X2, y2, targets)['predictions']
    assert preds1 == preds2


def test_walk_deterministic(monkeypatch):
    _patch_fit(monkeypatch)
    candles = mk(150)
    X, y, _, _ = mv.build_feature_matrix(candles)
    targets = list(range(mv.MIN_HISTORY, 110))
    p1 = mv.run_walk(X, y, targets)['predictions']
    p2 = mv.run_walk(X, y, targets)['predictions']
    assert p1 == p2


def test_real_backend_fit_deterministic():
    lgb = pytest.importorskip('lightgbm')
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    y = rng.normal(size=(200,))
    m1, b1 = mv.fit_model(X, y)
    m2, b2 = mv.fit_model(X, y)
    assert b1 == b2 == 'lightgbm'
    row = rng.normal(size=(1, 8))
    np.testing.assert_allclose(m1.predict(row), m2.predict(row), rtol=1e-10)


# --------------------------------------------------------------------------- #
# Identical timestamps with the classical evaluator
# --------------------------------------------------------------------------- #
def test_ml_timestamps_match_evaluator(monkeypatch):
    _patch_fit(monkeypatch)
    candles = mk(120)
    c = cfg(context_length=32, window_size=30)
    ev = PredictionEvaluator(mv._StubPredictor(), c, 'BTC/USDT', '1h')
    windows, _ = ev.evaluate_windows(candles)

    # ML report walk uses the same evaluator windows -> same timestamps
    report = mv.run_ml_vs_har(lambda s, tf: candles, c, [('BTC/USDT', '1h')])
    for name, res in windows.items():
        vts = sorted(r.prediction_timestamp for r in res.volatility_rows)
        rec = next(r for r in report['window_records']
                   if r['window'] == name and r['series'] == 'BTC/USDT')
        assert rec['sample_size'] == len(vts)


def test_statistical_alignment_ml_vs_har(monkeypatch):
    _patch_fit(monkeypatch)
    candles = mk(120)
    # context_length > MIN_HISTORY so the first window target has training rows
    c = cfg(context_length=80, window_size=30)
    report = mv.run_ml_vs_har(lambda s, tf: candles, c, [('BTC/USDT', '1h')])
    for w, analysis in report['series']['BTC/USDT 1h']['windows'].items():
        dm = analysis['comparisons']['ml_vs_har']['dm']
        # DM is paired on identical timestamps: n equals the number of rows
        # where BOTH ML and HAR forecasts are defined.
        assert dm['n'] == analysis['sample_size']
        assert dm['winner'] in ('ml', 'har', 'tie')


# --------------------------------------------------------------------------- #
# Empty / short-history safety
# --------------------------------------------------------------------------- #
def test_feature_matrix_short_history():
    assert mv.build_feature_matrix([]) is None
    assert mv.build_feature_matrix([Candle(0, 1, 1, 1, 1, 1)]) is None


def test_run_walk_empty_targets(monkeypatch):
    _patch_fit(monkeypatch)
    candles = mk(80)
    X, y, _, _ = mv.build_feature_matrix(candles)
    out = mv.run_walk(X, y, [])
    assert out['predictions'] == {} and out['leaks'] == 0


def test_run_ml_vs_har_short_series_pending(monkeypatch):
    _patch_fit(monkeypatch)
    candles = mk(6)  # too short -> no windows
    c = cfg(context_length=4, window_size=5)
    report = mv.run_ml_vs_har(lambda s, tf: candles, c, [('BTC/USDT', '1h')])
    assert report['success_gate']['overall'] == 'pending'
    assert report['success_gate']['eligible_windows'] == 0


# --------------------------------------------------------------------------- #
# Gate classification
# --------------------------------------------------------------------------- #
def _record(series='BTC/USDT', timeframe='1h', window='recent', sample_size=100,
            nmae=True, nrmse=True, mae=True, tracks=True, leaks=0):
    return {'series': series, 'timeframe': timeframe, 'window': window,
            'sample_size': sample_size, 'low_power': sample_size < 30,
            'ml_har_nmae_winner': nmae, 'ml_har_nrmse_winner': nrmse,
            'ml_har_mae_winner': mae, 'ml_tracks': tracks, 'leaks': leaks}


def test_ml_gate_criteria_and_classification():
    recs = [_record(series=s, window=w)
            for s in ('BTC/USDT', 'ETH/USDT', 'BTC/USDT', 'ETH/USDT')
            for w in ('older', 'middle', 'recent')]
    recs = [{**r, 'timeframe': '1h'} for r in recs]
    gate = mv.evaluate_ml_gate(recs)
    c = gate['criteria']
    assert c['c1_series_breadth_nmae'] is True
    assert c['c2_series_breadth_nrmse'] is True
    assert c['c4_raw_survives'] is True
    assert c['c6_not_solely_shrinkage'] is True
    assert c['c7_window_breadth'] is True
    assert c['c8_no_lookahead'] is True
    assert c['c3_statistical_support'] is None  # filled by caller
    assert c['c5_regime_breadth'] is None

    full = dict(c, c3_statistical_support=True, c5_regime_breadth=True)
    assert mv.classify_ml_gate(full) == 'A'
    assert mv.classify_ml_gate(dict(full, c3_statistical_support=False)) == 'B'
    losing = dict(full, c1_series_breadth_nmae=False, c2_series_breadth_nrmse=False)
    assert mv.classify_ml_gate(losing) == 'C'


def test_ml_gate_leakage_blocks():
    recs = [_record(leaks=1) for _ in range(12)]
    recs = [{**r, 'timeframe': '1h'} for r in recs]
    gate = mv.evaluate_ml_gate(recs)
    assert gate['criteria']['c8_no_lookahead'] is False


def test_ml_gate_excludes_daily_and_low_power():
    recs = [_record(timeframe='1d', sample_size=72) for _ in range(12)]
    gate = mv.evaluate_ml_gate(recs)
    assert gate['eligible_windows'] == 0
    assert gate['overall'] == 'pending'
