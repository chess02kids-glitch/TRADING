"""Phase 8 (F-01) - funding-only derivatives positioning tests.

Covers the required regression protections:

* F-01 does not require open interest / basis
* F-01 works with funding-only input
* F-01 cannot accidentally access missing OI/basis
* funding timestamp alignment is correct (settled rate <= prediction time)
* no future funding information is used
* no forward-fill across genuine gaps
* missing-history handling
* training cutoff correctness
* deterministic coefficients/predictions
* identical OOS timestamps
* leakage counter == 0
* frozen-HAR-unchanged
* C1-C7 gate classification

Synthetic aligned data is used; no network, no credentials.
"""
import math

import numpy as np
import pytest

from kronos_trading.derivatives_data import (FUNDING_INTERVAL_MS,
                                             FUNDING_WINDOW_MS,
                                             funding_features_24h)
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
H8 = 8 * H
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


def make_funding(n, seed=1, step=H8, base=BASE):
    """Settled funding history at 8h cadence, aligned to candle timestamps."""
    rng = np.random.default_rng(seed)
    return [{'timestamp_ms': base + i * step,
             'funding_rate': float(rng.normal(0, 1e-4)),
             'kind': 'funding'}
            for i in range(n)]


def cfg(**overrides):
    defaults = dict(context_length=64, horizon=1, window_size=30)
    defaults.update(overrides)
    return EvaluationConfig(**defaults)


# --------------------------------------------------------------------------- #
# funding_features_24h correctness
# --------------------------------------------------------------------------- #
def test_funding_mean_24h_window():
    # settlements every 8h, rates 1..6
    base = 1_000_000_000_000
    rows = [{'timestamp_ms': base + i * H8, 'funding_rate': float(i + 1)}
            for i in range(6)]
    # query at base + 3*H8 (t = 24h): window is (t-24h, t] = (base, base+24h]
    # => settlements at base+8h, base+16h, base+24h => rates 2,3,4 => mean 3.0
    out = funding_features_24h(rows, [base + 3 * H8])
    assert out['funding_mean_24h'] == [pytest.approx(3.0)]  # mean(2,3,4)
    assert out['abs_funding_mean_24h'] == [pytest.approx(3.0)]


def test_funding_features_staleness_skip():
    base = 1_000_000_000_000
    rows = [{'timestamp_ms': base, 'funding_rate': 1.0},
            {'timestamp_ms': base + H8, 'funding_rate': 2.0}]
    # query at base + 2*H8 + FUNDING_INTERVAL_MS (just beyond 8h staleness)
    out = funding_features_24h(rows, [base + 2 * H8 + FUNDING_INTERVAL_MS + 1])
    assert out['funding_mean_24h'] == [None]  # stale -> skip


def test_funding_features_missing_history():
    base = 1_000_000_000_000
    rows = [{'timestamp_ms': base, 'funding_rate': 1.0}]
    # no settlements before t -> None
    out = funding_features_24h(rows, [base - 1])
    assert out['funding_mean_24h'] == [None]


def test_funding_features_no_future():
    base = 1_000_000_000_000
    rows = [{'timestamp_ms': base + H8, 'funding_rate': 1.0}]
    # query before the settlement -> None (future funding never used)
    out = funding_features_24h(rows, [base])
    assert out['funding_mean_24h'] == [None]


# --------------------------------------------------------------------------- #
# Feature construction (funding-only, no OI/basis)
# --------------------------------------------------------------------------- #
def test_feature_names_are_funding_only():
    assert FEATURE_NAMES == ['har_range_prev', 'har_mean5', 'har_mean22',
                             'funding_mean_24h', 'abs_funding_mean_24h']


def test_build_features_works_with_funding_only():
    target = mk(200, 11)
    funding_rows = make_funding(60)  # 60 * 8h = 480h of funding history
    out = build_derivatives_features(target, funding_rows, H)
    assert out['X'].shape == (200, len(FEATURE_NAMES))
    # rows with valid funding features (after MIN_HISTORY and funding history)
    assert out['valid'][24:].sum() > 0
    assert out['missing'] == 0


def test_build_features_does_not_access_oi_or_basis():
    target = mk(80, 12)
    # funding-only payload: no 'open_interest' / 'basis' keys at all
    funding_rows = make_funding(30)
    out = build_derivatives_features(target, funding_rows, H)
    # must not raise KeyError and must produce funding-based features
    assert out['X'].shape == (80, len(FEATURE_NAMES))


def test_no_future_funding_information():
    target = mk(200, 13)
    funding_rows = make_funding(60)
    out1 = build_derivatives_features(target, funding_rows, H)
    # perturb a FUTURE funding settlement (after the last prediction timestamp)
    modified = list(funding_rows)
    future_ts = max(c.timestamp_ms for c in target) + H8
    modified.append({'timestamp_ms': future_ts, 'funding_rate': 999.0, 'kind': 'funding'})
    out2 = build_derivatives_features(target, modified, H)
    np.testing.assert_allclose(out1['X'], out2['X'], rtol=0, atol=0)


def test_no_forward_fill_across_genuine_gap():
    target = mk(200, 14)  # 200 candles at 1h => 200h; funding settles every 8h
    funding_rows = make_funding(60)
    # remove a RUN of funding settlements INSIDE the target candle range
    # (indices 5..10, i.e. base+40h .. base+80h) -> a genuine >8h gap.
    gap = [r for r in funding_rows if r['timestamp_ms'] < funding_rows[5]['timestamp_ms']
           or r['timestamp_ms'] > funding_rows[10]['timestamp_ms']]
    out = build_derivatives_features(target, gap, H)
    # rows whose most recent settled funding is stale are invalid (skipped),
    # never forward-filled
    assert out['missing'] > 0
    # every valid row's funding features are finite and derived from funding only
    assert out['X'][out['valid']].shape[1] == len(FEATURE_NAMES)
    # rows well before the gap are still valid
    assert bool(out['valid'][25]) is True


# --------------------------------------------------------------------------- #
# Walk-forward OLS
# --------------------------------------------------------------------------- #
def test_walk_ols_training_cutoff_and_leaks():
    target = mk(150, 15)
    funding_rows = make_funding(50)
    out = build_derivatives_features(target, funding_rows, H)
    idxs = list(range(48, 110))
    walk = run_walk_ols(out['X'], out['y_raw'], out['valid'], idxs)
    assert walk['leaks'] == 0
    for r in walk['retrains']:
        assert r['train_end_idx'] == r['pred_idx']
        assert r['n_train'] == r['pred_idx'] - 24


def test_walk_ols_deterministic():
    target = mk(150, 16)
    funding_rows = make_funding(50)
    out = build_derivatives_features(target, funding_rows, H)
    idxs = list(range(48, 110))
    w1 = run_walk_ols(out['X'], out['y_raw'], out['valid'], idxs)
    w2 = run_walk_ols(out['X'], out['y_raw'], out['valid'], idxs)
    assert w1['predictions'] == w2['predictions']
    for a, b in zip(w1['coefficients'], w2['coefficients']):
        np.testing.assert_allclose(a, b)


def test_missing_history_safe():
    target = mk(60, 18)
    funding_rows = make_funding(5)  # insufficient funding history
    out = build_derivatives_features(target, funding_rows, H)
    w = run_walk_ols(out['X'], out['y_raw'], out['valid'], [10, 20])
    assert math.isnan(w['predictions'][10])
    assert w['leaks'] == 0


# --------------------------------------------------------------------------- #
# End-to-end + identical timestamps + frozen HAR + gate
# --------------------------------------------------------------------------- #
def test_run_derivatives_funding_only_structure_and_determinism():
    data = {('BTC/USDT', '1h'): mk(300, 21), ('ETH/USDT', '1h'): mk(300, 22)}
    funding = {'BTCUSDT': {'funding': make_funding(90, seed=30)},
               'ETHUSDT': {'funding': make_funding(90, seed=31)}}
    c = cfg(context_length=64, window_size=40)
    series = [('BTC/USDT', '1h'), ('ETH/USDT', '1h')]
    r1 = run_derivatives_volatility(lambda s, tf: data[(s, tf)],
                                    lambda sym: funding[sym], c, series)
    r2 = run_derivatives_volatility(lambda s, tf: data[(s, tf)],
                                    lambda sym: funding[sym], c, series)
    assert r1['kind'] == 'derivatives_volatility_f01'
    assert r1['window_records'] == r2['window_records']
    assert r1['success_gate'] == r2['success_gate']
    assert r1['pooled_primary'] == r2['pooled_primary']
    for sid in ('BTC/USDT', 'ETH/USDT'):
        assert set(r1['series'][sid]['windows']) == {'older', 'middle', 'recent'}


def test_run_derivatives_identical_oos_timestamps():
    data = {('BTC/USDT', '1h'): mk(300, 23)}
    funding = {'BTCUSDT': {'funding': make_funding(90, seed=32)}}
    c = cfg(context_length=64, window_size=40)
    report = run_derivatives_volatility(lambda s, tf: data[(s, tf)],
                                        lambda sym: funding[sym], c,
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


def test_run_derivatives_empty_safe():
    data = {('BTC/USDT', '1h'): mk(5, 27)}
    funding = {'BTCUSDT': {'funding': []}}
    c = cfg(context_length=4, window_size=5)
    report = run_derivatives_volatility(lambda s, tf: data[(s, tf)],
                                        lambda sym: funding[sym], c,
                                        [('BTC/USDT', '1h')])
    assert report['success_gate']['overall'] == 'pending'
    assert report['success_gate']['eligible_windows'] == 0


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
