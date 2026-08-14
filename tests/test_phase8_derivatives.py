"""Phase 8 - derivatives positioning tests.

Covers: alignment, settled-funding correctness, OI timestamp cutoff, basis
timestamp cutoff, no-forward-fill, missing-history safety, no-future
perturbation invariance, training cutoff, deterministic coefficients, identical
OOS timestamps, leakage counter, and frozen-HAR-unchanged. Synthetic aligned
data is used; no network, no credentials.
"""
import math

import numpy as np
import pytest

from kronos_trading.derivatives_data import (align_derivatives, as_of_series,
                                             oi_log_change_22)
from kronos_trading.derivatives_volatility import (
    FEATURE_NAMES,
    build_derivatives_features,
    classify_derivatives_gate,
    evaluate_derivatives_gate,
    run_derivatives_volatility,
    run_walk_ols,
)
from kronos_trading.evaluation import EvaluationConfig
from kronos_trading.types import Candle

H = 3_600_000
BASE = 1_700_000_000_000


def mk(n, seed=7, step=H, base=BASE):
    rng = np.random.default_rng(seed)
    out = []
    close = 100.0
    r = 1.0
    for i in range(n):
        r = max(0.05, 0.9 * r + 0.1 * 0.5 + rng.normal(0, 0.04))
        close = close * (1.0 + rng.normal(0, 0.001))
        out.append(Candle(base + i * step, close - r / 2, close + r / 2,
                          close - r / 2, close, 1000.0 + rng.normal(0, 30)))
    return out


def make_derivatives(n, step=H, base=BASE, seed=1):
    """Synthetic point-in-time derivatives aligned to candle open times."""
    rng = np.random.default_rng(seed)
    funding, oi, basis = [], [], []
    oi_level = 1e6
    for i in range(n):
        t = base + i * step
        # settled funding (previous interval), snapshot OI, basis at t
        funding.append({'timestamp_ms': t, 'funding_rate': float(rng.normal(0, 1e-4))})
        oi_level *= (1.0 + rng.normal(0, 0.002))
        oi.append({'timestamp_ms': t, 'open_interest': oi_level})
        basis.append({'timestamp_ms': t, 'basis': float(rng.normal(0, 1e-4))})
    return {'funding': funding, 'open_interest': oi, 'basis': basis}


def cfg(**overrides):
    defaults = dict(context_length=64, horizon=1, window_size=30)
    defaults.update(overrides)
    return EvaluationConfig(**defaults)


# --------------------------------------------------------------------------- #
# Alignment helpers
# --------------------------------------------------------------------------- #
def test_as_of_series_point_in_time():
    rows = [{'timestamp_ms': 100, 'v': 1.0}, {'timestamp_ms': 300, 'v': 3.0}]
    assert as_of_series(rows, [50, 100, 200, 300, 400], 'v') == \
        [None, 1.0, 1.0, 3.0, 3.0]


def test_as_of_series_no_forward_fill():
    rows = [{'timestamp_ms': 500, 'v': 5.0}]
    # timestamps before the first observation -> None (skip, never forward-filled)
    assert as_of_series(rows, [100, 200, 500], 'v') == [None, None, 5.0]


def test_oi_log_change_22():
    oi = [float(i) for i in range(1, 60)]
    out = oi_log_change_22(oi)
    assert out[21] is None
    assert out[22] == pytest.approx(math.log(23 / 1))
    assert out[40] == pytest.approx(math.log(41 / 19))
    # missing value -> None
    oi2 = oi[:]
    oi2[30] = None
    out2 = oi_log_change_22(oi2)
    assert out2[52] is None  # depends on index 30


def test_align_derivatives_missing_is_none():
    data = {'funding': [{'timestamp_ms': 200, 'funding_rate': 0.1}],
            'open_interest': [{'timestamp_ms': 200, 'open_interest': 1e6}],
            'basis': [{'timestamp_ms': 200, 'basis': 0.0}]}
    out = align_derivatives(data, [100, 200, 300])
    assert out['funding'] == [None, 0.1, 0.1]
    assert out['basis'] == [None, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# Feature construction: point-in-time, no future, no forward-fill
# --------------------------------------------------------------------------- #
def test_feature_shapes_and_validity():
    target = mk(200, 11)
    deriv = make_derivatives(200)
    out = build_derivatives_features(target, deriv, H)
    assert out['X'].shape == (200, len(FEATURE_NAMES))
    assert out['valid'][24:].all()
    assert not out['valid'][:24].any()
    assert out['missing'] == 0


def test_settled_funding_correctness():
    target = mk(60, 12)
    deriv = make_derivatives(60)
    out = build_derivatives_features(target, deriv, H)
    idx = {name: k for k, name in enumerate(FEATURE_NAMES)}
    # row j uses derivative snapshot at T_j - step = index j-1
    for j in (30, 45):
        expected_funding = deriv['funding'][j - 1]['funding_rate']
        assert out['X'][j, idx['funding']] == pytest.approx(expected_funding)


def test_no_future_information():
    target = mk(200, 13)
    deriv = make_derivatives(200)
    out1 = build_derivatives_features(target, deriv, H)
    # perturb derivative values at/after T_60 (indices >= 60)
    modified = {k: list(v) for k, v in deriv.items()}
    for k in ('funding', 'open_interest', 'basis'):
        for i in range(60, len(modified[k])):
            r = dict(modified[k][i])
            val = r.get('funding_rate', r.get('open_interest', r.get('basis')))
            r[list(r.keys())[1]] = val + 999.0
            modified[k][i] = r
    out2 = build_derivatives_features(target, modified, H)
    np.testing.assert_allclose(out1['X'][24:61], out2['X'][24:61], rtol=0, atol=0)
    assert not np.allclose(out1['X'][61:], out2['X'][61:], rtol=0, atol=0)


def test_no_forward_fill_missing_row():
    target = mk(200, 14)
    deriv = make_derivatives(200)
    # remove a RUN of basis snapshots (indices 90..99) -> a genuine gap longer
    # than one candle step. Rows whose last available basis is older than the
    # staleness bound must be SKIPPED (never forward-filled).
    deriv['basis'] = deriv['basis'][:90] + deriv['basis'][100:]
    out = build_derivatives_features(target, deriv, H)
    # rows 91..100 query at ts[90..99]; last available basis is index 89
    # (age >= step for row 91, growing to 11 steps for row 100) -> skipped
    assert not out['valid'][100]
    assert out['missing'] >= 1
    # row 90 queries at ts[89] (age 1 step, within tolerance) -> still valid
    assert out['valid'][90]


# --------------------------------------------------------------------------- #
# Walk-forward OLS: cutoff, no leakage, determinism
# --------------------------------------------------------------------------- #
def test_walk_ols_training_cutoff_and_leaks():
    target = mk(150, 15)
    deriv = make_derivatives(150)
    out = build_derivatives_features(target, deriv, H)
    idxs = list(range(48, 110))
    walk = run_walk_ols(out['X'], out['y_raw'], out['valid'], idxs)
    assert walk['leaks'] == 0
    for r in walk['retrains']:
        assert r['train_end_idx'] == r['pred_idx']
        assert r['n_train'] == r['pred_idx'] - 24


def test_walk_ols_deterministic():
    target = mk(150, 16)
    deriv = make_derivatives(150)
    out = build_derivatives_features(target, deriv, H)
    idxs = list(range(48, 110))
    w1 = run_walk_ols(out['X'], out['y_raw'], out['valid'], idxs)
    w2 = run_walk_ols(out['X'], out['y_raw'], out['valid'], idxs)
    assert w1['predictions'] == w2['predictions']
    for a, b in zip(w1['coefficients'], w2['coefficients']):
        np.testing.assert_allclose(a, b)


def test_walk_ols_future_invariance():
    target = mk(150, 17)
    deriv = make_derivatives(150)
    out1 = build_derivatives_features(target, deriv, H)
    idxs = list(range(48, 100))
    w1 = run_walk_ols(out1['X'], out1['y_raw'], out1['valid'], idxs)
    modified = list(target)
    for i in range(100, 150):
        c = modified[i]
        modified[i] = Candle(c.timestamp_ms, c.open + 5, c.high + 5,
                             c.low + 5, c.close + 5, c.volume)
    out2 = build_derivatives_features(modified, deriv, H)
    w2 = run_walk_ols(out2['X'], out2['y_raw'], out2['valid'], idxs)
    assert w1['predictions'] == w2['predictions']


def test_missing_history_safe():
    target = mk(60, 18)
    deriv = make_derivatives(60)
    out = build_derivatives_features(target, deriv, H)
    w = run_walk_ols(out['X'], out['y_raw'], out['valid'], [10, 20])
    assert math.isnan(w['predictions'][10])
    assert math.isnan(w['predictions'][20])
    assert w['leaks'] == 0


# --------------------------------------------------------------------------- #
# End-to-end + identical timestamps + gate
# --------------------------------------------------------------------------- #
def test_run_derivatives_structure_and_determinism():
    data = {('BTC/USDT', '1h'): mk(300, 21), ('ETH/USDT', '1h'): mk(300, 22)}
    deriv = {'BTCUSDT': make_derivatives(300, seed=30),
             'ETHUSDT': make_derivatives(300, seed=31)}
    c = cfg(context_length=64, window_size=40)
    series = [('BTC/USDT', '1h'), ('ETH/USDT', '1h')]
    r1 = run_derivatives_volatility(lambda s, tf: data[(s, tf)],
                                    lambda sym: deriv[sym], c, series)
    r2 = run_derivatives_volatility(lambda s, tf: data[(s, tf)],
                                    lambda sym: deriv[sym], c, series)
    assert r1['kind'] == 'derivatives_volatility'
    assert r1['window_records'] == r2['window_records']
    assert r1['success_gate'] == r2['success_gate']
    assert r1['pooled_primary'] == r2['pooled_primary']
    for sid in ('BTC/USDT', 'ETH/USDT'):
        assert set(r1['series'][sid]['windows']) == {'older', 'middle', 'recent'}


def test_identical_oos_timestamps_with_frozen_har():
    data = {('BTC/USDT', '1h'): mk(300, 23)}
    deriv = {'BTCUSDT': make_derivatives(300, seed=32)}
    c = cfg(context_length=64, window_size=40)
    report = run_derivatives_volatility(lambda s, tf: data[(s, tf)],
                                        lambda sym: deriv[sym], c,
                                        [('BTC/USDT', '1h')])
    for w, analysis in report['series']['BTC/USDT']['windows'].items():
        dm = analysis['comparisons']['ext_vs_har']['dm']
        assert dm['n'] == analysis['sample_size']
        assert dm['winner'] in ('ext', 'har', 'tie')


def test_frozen_har_unchanged():
    from kronos_trading.volatility_baselines import (EWMA_SPAN, HAR_MIN_TRAIN,
                                                     ROLLING_WINDOWS, har_forecast)
    assert HAR_MIN_TRAIN == 24
    assert ROLLING_WINDOWS == (5, 22)
    assert EWMA_SPAN == 22
    assert har_forecast([3.0] * 50) == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
def _record(asset='BTC/USDT', timeframe='1h', window='recent', sample_size=100,
            nmae=True, nrmse=True, mae=True, leaks=0):
    return {'asset': asset, 'timeframe': timeframe, 'window': window,
            'sample_size': sample_size, 'low_power': sample_size < 30,
            'ext_har_nmae_winner': nmae, 'ext_har_nrmse_winner': nrmse,
            'ext_har_mae_winner': mae, 'leaks': leaks}


def _full_records(nmae=True, nrmse=True, mae=True, leaks=0):
    recs = []
    for asset in ('BTC/USDT', 'ETH/USDT'):
        for tf in ('1h', '4h'):
            for w in ('older', 'middle', 'recent'):
                recs.append(_record(asset=asset, timeframe=tf, window=w,
                                    nmae=nmae, nrmse=nrmse, mae=mae, leaks=leaks))
    return recs


def test_gate_classification():
    gate = evaluate_derivatives_gate(_full_records())
    c = gate['criteria']
    assert c['c1_window_breadth_per_asset'] is True
    assert c['c2_both_assets'] is True
    assert c['c3_rmse_pattern'] is True
    assert c['c4_raw_survives'] is True
    assert c['c7_no_leakage'] is True
    full = dict(c, c5_statistical_support=True, c6_regime_breadth=True)
    assert classify_derivatives_gate(full) == 'PASS'
    assert classify_derivatives_gate(dict(full, c5_statistical_support=False)) == 'B'
    losing = dict(full, c1_window_breadth_per_asset=False)
    assert classify_derivatives_gate(losing) == 'C'


def test_gate_leakage_and_breadth():
    gate = evaluate_derivatives_gate(_full_records(leaks=1))
    assert gate['criteria']['c7_no_leakage'] is False

    recs = []
    for asset in ('BTC/USDT', 'ETH/USDT'):
        for tf in ('1h', '4h'):
            for w in ('older', 'middle', 'recent'):
                recs.append(_record(asset=asset, timeframe=tf, window=w,
                                    nmae=(asset == 'BTC/USDT'),
                                    nrmse=(asset == 'BTC/USDT'),
                                    mae=(asset == 'BTC/USDT')))
    gate2 = evaluate_derivatives_gate(recs)
    assert gate2['criteria']['c1_window_breadth_per_asset'] is True
    assert gate2['criteria']['c2_both_assets'] is False


def test_gate_excludes_daily_and_low_power():
    recs = [_record(timeframe='1d', sample_size=72) for _ in range(12)]
    gate = evaluate_derivatives_gate(recs)
    assert gate['eligible_windows'] == 0
    assert gate['overall'] == 'pending'
