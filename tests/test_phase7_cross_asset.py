"""Phase 7 - cross-asset information experiment tests.

Covers: timestamp alignment, no future information, no forward-fill,
missing-history handling, deterministic coefficients/predictions, identical OOS
timestamps, and the leakage counter. All tests use synthetic aligned data and
the frozen HAR machinery from ``volatility_baselines``.
"""
import math

import numpy as np
import pytest

from kronos_trading.cross_asset import (
    FEATURE_NAMES,
    build_aligned_features,
    classify_cross_gate,
    evaluate_cross_gate,
    run_cross_asset,
    run_walk_ols,
)
from kronos_trading.evaluation import EvaluationConfig
from kronos_trading.types import Candle

H = 3_600_000
BASE = 1_700_000_000_000


def mk(n, seed, step=H, base=BASE):
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


def cfg(**overrides):
    defaults = dict(context_length=64, horizon=1, window_size=30)
    defaults.update(overrides)
    return EvaluationConfig(**defaults)


# --------------------------------------------------------------------------- #
# Feature construction: alignment + no future + no forward-fill
# --------------------------------------------------------------------------- #
def test_feature_shapes_and_validity():
    target = mk(200, 1)
    other = mk(200, 2)
    out = build_aligned_features(target, other, H)
    assert out['X'].shape == (200, len(FEATURE_NAMES))
    assert out['y_raw'].shape == (200,)
    assert out['y_norm'].shape == (200,)
    # rows before MIN_HISTORY are invalid
    assert not out['valid'][:24].any()
    assert out['valid'][24:].all()
    assert out['cross_missing'] == 0


def test_target_normalized_alignment():
    target = mk(120, 3)
    other = mk(120, 4)
    out = build_aligned_features(target, other, H)
    for j in range(24, 60):
        expected = (target[j].high - target[j].low) / target[j - 1].close
        assert out['y_norm'][j] == pytest.approx(expected, rel=1e-9)
        assert out['y_raw'][j] == pytest.approx(target[j].high - target[j].low)


def test_cross_features_correctness():
    target = mk(120, 5)
    other = mk(120, 6)
    out = build_aligned_features(target, other, H)
    idx = {name: k for k, name in enumerate(FEATURE_NAMES)}
    o_close = np.array([c.close for c in other], dtype=float)
    o_range = np.array([c.high - c.low for c in other], dtype=float)
    for j in (30, 60, 90):
        # cross candle k = other candle at open time T_j - H = index j-1
        k = j - 1
        row = out['X'][j]
        # cross_nr_1 = (high_other - low_other) / close_other  (OWN close)
        assert row[idx['x_nr_prev']] == pytest.approx(o_range[k] / o_close[k], rel=1e-9)
        # cross_ret1 = log(close_other[t]/close_other[t-1])
        assert row[idx['x_ret_1']] == pytest.approx(
            np.log(o_close[k] / o_close[k - 1]), rel=1e-9)
        # cross_ret22 = log(close_other[t]/close_other[t-22])
        assert row[idx['x_ret_22']] == pytest.approx(
            np.log(o_close[k] / o_close[k - 22]), rel=1e-9)
        # cross_rv22 = std of 22 log returns ending at t (same convention as code)
        logret = np.zeros(len(o_close)); logret[1:] = np.log(o_close[1:] / o_close[:-1])
        rv = logret[k - 21:k + 1]
        assert row[idx['x_rv_22']] == pytest.approx(np.std(rv, ddof=1), rel=1e-6)


def test_har_features_correctness():
    target = mk(120, 7)
    other = mk(120, 8)
    out = build_aligned_features(target, other, H)
    idx = {name: k for k, name in enumerate(FEATURE_NAMES)}
    t_range = np.array([c.high - c.low for c in target], dtype=float)
    for j in (30, 60):
        row = out['X'][j]
        assert row[idx['har_range_prev']] == pytest.approx(t_range[j - 1])
        assert row[idx['har_mean5']] == pytest.approx(t_range[j - 5:j].mean())
        assert row[idx['har_mean22']] == pytest.approx(t_range[j - 22:j].mean())


def test_no_future_information():
    target = mk(200, 9)
    other = mk(200, 10)
    out1 = build_aligned_features(target, other, H)
    # Row j's cross feature uses the other candle with open time T_j - H (index
    # j-1). Perturb the other candles with open time >= T_60 (indices >= 60).
    # Rows j <= 60 use other index j-1 <= 59 (unperturbed) -> unchanged.
    modified = list(other)
    for i in range(60, len(modified)):
        c = modified[i]
        modified[i] = Candle(c.timestamp_ms, c.open + 99, c.high + 99,
                             c.low + 99, c.close + 99, c.volume)
    out2 = build_aligned_features(target, modified, H)
    # rows 24..60 inclusive only use other candles with open time < T_j
    np.testing.assert_allclose(out1['X'][24:61], out2['X'][24:61], rtol=0, atol=0)
    # rows after 60 must differ (their cross candle changed)
    assert not np.allclose(out1['X'][61:], out2['X'][61:], rtol=0, atol=0)


def test_no_forward_fill_on_missing_cross_candle():
    target = mk(200, 11)
    other = mk(200, 12)
    # remove the other candle at index 99 (open time = T_100 - H) -> a gap
    gapped = other[:99] + other[100:]
    out = build_aligned_features(target, gapped, H)
    # the target row whose cross candle was removed is INVALID, not forward-filled
    assert out['cross_missing'] >= 1
    # target index 100 needs other candle at open time T_100 - H (removed)
    assert not out['valid'][100]
    # surrounding rows remain valid (their cross candles still exist)
    assert out['valid'][99] and out['valid'][101]


# --------------------------------------------------------------------------- #
# Walk-forward OLS: cutoff, no leakage, determinism, future-invariance
# --------------------------------------------------------------------------- #
def _target_indices(n, start=24, stop=None):
    return list(range(start, stop or n))


def test_walk_ols_training_cutoff_and_leaks():
    target = mk(150, 13)
    other = mk(150, 14)
    out = build_aligned_features(target, other, H)
    idxs = _target_indices(120, 48, 110)  # >= MIN_HISTORY + MIN_TRAIN_ROWS
    walk = run_walk_ols(out['X'], out['y_raw'], out['valid'], idxs)
    assert walk['leaks'] == 0
    # every refit trains strictly before its prediction index
    for r in walk['retrains']:
        assert r['train_end_idx'] == r['pred_idx']
        assert r['n_train'] == r['pred_idx'] - 24
    # predictions exist and are finite
    assert all(math.isfinite(p) for p in walk['predictions'].values())


def test_walk_ols_deterministic():
    target = mk(150, 15)
    other = mk(150, 16)
    out = build_aligned_features(target, other, H)
    idxs = _target_indices(120, 48, 110)
    w1 = run_walk_ols(out['X'], out['y_raw'], out['valid'], idxs)
    w2 = run_walk_ols(out['X'], out['y_raw'], out['valid'], idxs)
    assert w1['predictions'] == w2['predictions']
    for a, b in zip(w1['coefficients'], w2['coefficients']):
        np.testing.assert_allclose(a, b)


def test_walk_ols_future_invariance():
    target = mk(150, 17)
    other = mk(150, 18)
    out1 = build_aligned_features(target, other, H)
    idxs = _target_indices(120, 48, 100)
    w1 = run_walk_ols(out1['X'], out1['y_raw'], out1['valid'], idxs)
    # modify target candles AFTER the last prediction index (100) -> unchanged
    modified = list(target)
    for i in range(100, 150):
        c = modified[i]
        modified[i] = Candle(c.timestamp_ms, c.open + 5, c.high + 5,
                             c.low + 5, c.close + 5, c.volume)
    out2 = build_aligned_features(modified, other, H)
    w2 = run_walk_ols(out2['X'], out2['y_raw'], out2['valid'], idxs)
    assert w1['predictions'] == w2['predictions']


def test_walk_ols_missing_history_safe():
    target = mk(60, 19)
    other = mk(60, 20)
    out = build_aligned_features(target, other, H)
    # indices below MIN_HISTORY cannot be predicted
    w = run_walk_ols(out['X'], out['y_raw'], out['valid'], [10, 20])
    assert math.isnan(w['predictions'][10])
    assert math.isnan(w['predictions'][20])
    assert w['leaks'] == 0


# --------------------------------------------------------------------------- #
# End-to-end experiment + identical timestamps + gate
# --------------------------------------------------------------------------- #
def test_run_cross_asset_structure_and_determinism():
    data = {
        ('BTC/USDT', '1h'): mk(300, 21),
        ('ETH/USDT', '1h'): mk(300, 22),
    }
    c = cfg(context_length=64, window_size=40)
    pairs = [('BTC/USDT', 'ETH/USDT', '1h'), ('ETH/USDT', 'BTC/USDT', '1h')]
    r1 = run_cross_asset(lambda s, tf: data[(s, tf)], c, pairs)
    r2 = run_cross_asset(lambda s, tf: data[(s, tf)], c, pairs)
    assert r1['kind'] == 'cross_asset_volatility'
    assert r1['window_records'] == r2['window_records']
    assert r1['success_gate'] == r2['success_gate']
    assert r1['pooled_statistics'] == r2['pooled_statistics']
    for sid in ('BTC/USDT<-other(ETH/USDT)', 'ETH/USDT<-other(BTC/USDT)'):
        assert set(r1['series'][sid]['windows']) == {'older', 'middle', 'recent'}


def test_identical_oos_timestamps_with_frozen_har():
    data = {('BTC/USDT', '1h'): mk(300, 23), ('ETH/USDT', '1h'): mk(300, 24)}
    c = cfg(context_length=64, window_size=40)
    report = run_cross_asset(lambda s, tf: data[(s, tf)], c,
                             [('BTC/USDT', 'ETH/USDT', '1h')])
    for w, analysis in report['series']['BTC/USDT<-other(ETH/USDT)']['windows'].items():
        # DM (cross vs har) is paired on identical timestamps
        dm = analysis['comparisons']['cross_vs_har']['dm']
        assert dm['n'] == analysis['sample_size']
        assert dm['winner'] in ('cross', 'har', 'tie')


def test_cross_vs_har_labels_not_kronos():
    data = {('BTC/USDT', '1h'): mk(200, 25), ('ETH/USDT', '1h'): mk(200, 26)}
    c = cfg(context_length=64, window_size=30)
    report = run_cross_asset(lambda s, tf: data[(s, tf)], c,
                             [('BTC/USDT', 'ETH/USDT', '1h')])
    for w, analysis in report['series']['BTC/USDT<-other(ETH/USDT)']['windows'].items():
        for key, comp in analysis['comparisons'].items():
            assert comp['dm']['winner'] in ('cross', 'har', 'prev', 'ewma',
                                            'rolling5', 'rolling22', 'tie')


def test_run_cross_asset_empty_safe():
    data = {('BTC/USDT', '1h'): mk(5, 27), ('ETH/USDT', '1h'): mk(5, 28)}
    c = cfg(context_length=4, window_size=5)
    report = run_cross_asset(lambda s, tf: data[(s, tf)], c,
                             [('BTC/USDT', 'ETH/USDT', '1h')])
    assert report['success_gate']['overall'] == 'pending'
    assert report['success_gate']['eligible_windows'] == 0


# --------------------------------------------------------------------------- #
# Gate classification
# --------------------------------------------------------------------------- #
def _record(asset='BTC/USDT', timeframe='1h', window='recent', sample_size=100,
            nmae=True, nrmse=True, mae=True, leaks=0):
    return {'asset': asset, 'series': asset, 'timeframe': timeframe,
            'window': window, 'sample_size': sample_size,
            'low_power': sample_size < 30,
            'cross_har_nmae_winner': nmae, 'cross_har_nrmse_winner': nrmse,
            'cross_har_mae_winner': mae, 'leaks': leaks}


def _full_records(nmae=True, nrmse=True, mae=True, leaks=0):
    # BTC and ETH each win all 6 primary windows (1h+4h x 3 windows)
    recs = []
    for asset in ('BTC/USDT', 'ETH/USDT'):
        for tf in ('1h', '4h'):
            for w in ('older', 'middle', 'recent'):
                recs.append(_record(asset=asset, timeframe=tf, window=w,
                                    nmae=nmae, nrmse=nrmse, mae=mae, leaks=leaks))
    return recs


def test_cross_gate_classification():
    recs = _full_records()
    gate = evaluate_cross_gate(recs)
    c = gate['criteria']
    assert c['c1_window_breadth_per_asset'] is True
    assert c['c2_both_assets'] is True
    assert c['c3_rmse_pattern'] is True
    assert c['c4_raw_survives'] is True
    assert c['c7_no_leakage'] is True
    assert c['c5_statistical_support'] is None  # filled by caller
    assert c['c6_regime_breadth'] is None

    full = dict(c, c5_statistical_support=True, c6_regime_breadth=True)
    assert classify_cross_gate(full) == 'PASS'
    assert classify_cross_gate(dict(full, c5_statistical_support=False)) == 'B'
    losing = dict(full, c1_window_breadth_per_asset=False)
    assert classify_cross_gate(losing) == 'C'


def test_cross_gate_leakage_and_asset_breadth():
    recs = _full_records(leaks=1)
    gate = evaluate_cross_gate(recs)
    assert gate['criteria']['c7_no_leakage'] is False

    # only BTC wins -> C1 true (BTC >=2/3) but C2 false (ETH fails)
    recs2 = []
    for asset in ('BTC/USDT', 'ETH/USDT'):
        for tf in ('1h', '4h'):
            for w in ('older', 'middle', 'recent'):
                recs2.append(_record(asset=asset, timeframe=tf, window=w,
                                     nmae=(asset == 'BTC/USDT'),
                                     nrmse=(asset == 'BTC/USDT'),
                                     mae=(asset == 'BTC/USDT')))
    gate2 = evaluate_cross_gate(recs2)
    assert gate2['criteria']['c1_window_breadth_per_asset'] is True
    assert gate2['criteria']['c2_both_assets'] is False
    assert classify_cross_gate(dict(gate2['criteria'],
                                    c5_statistical_support=False,
                                    c6_regime_breadth=False)) == 'B'


def test_cross_gate_excludes_daily_and_low_power():
    recs = [_record(timeframe='1d', sample_size=72) for _ in range(12)]
    gate = evaluate_cross_gate(recs)
    assert gate['eligible_windows'] == 0
    assert gate['overall'] == 'pending'


def test_frozen_har_specification_unchanged():
    """The frozen single-asset HAR benchmark constants and formula are intact."""
    from kronos_trading.volatility_baselines import (EWMA_SPAN, HAR_MIN_TRAIN,
                                                     ROLLING_WINDOWS, har_forecast)
    assert HAR_MIN_TRAIN == 24
    assert ROLLING_WINDOWS == (5, 22)
    assert EWMA_SPAN == 22
    # constant series recovers the constant exactly (formula intact)
    assert har_forecast([3.0] * 50) == pytest.approx(3.0)
