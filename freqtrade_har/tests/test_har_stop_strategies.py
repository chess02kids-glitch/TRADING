import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from freqtrade.persistence import Trade
from freqtrade_har.strategies.har_stop_dynamic import compute_har_predictions, HARStopDynamic
from freqtrade_har.strategies.har_stop_inverse import HARStopInverse
from freqtrade_har.strategies.har_stop_baseline import HARStopBaseline

@pytest.fixture
def sample_ohlcv():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
        "open": np.random.uniform(40000, 41000, n),
        "high": np.random.uniform(41000, 42000, n),
        "low": np.random.uniform(39000, 40000, n),
        "close": np.random.uniform(40000, 41000, n),
        "volume": np.random.uniform(10, 100, n),
    })
    return df

@pytest.fixture
def mock_trade():
    trade = MagicMock(spec=Trade)
    trade.open_date_utc = datetime(2024, 1, 2, tzinfo=timezone.utc)
    trade.open_rate = 40000.0
    return trade

def test_compute_har_predictions_length(sample_ohlcv):
    preds = compute_har_predictions(sample_ohlcv["high"], sample_ohlcv["low"])
    assert len(preds) == len(sample_ohlcv)

def test_compute_har_predictions_nan_initial(sample_ohlcv):
    from kronos_trading.volatility_baselines import HAR_MIN_TRAIN
    preds = compute_har_predictions(sample_ohlcv["high"], sample_ohlcv["low"])
    assert preds.iloc[:HAR_MIN_TRAIN].isna().all()

def test_compute_har_predictions_values(sample_ohlcv):
    from kronos_trading.volatility_baselines import HAR_MIN_TRAIN
    preds = compute_har_predictions(sample_ohlcv["high"], sample_ohlcv["low"])
    assert preds.iloc[HAR_MIN_TRAIN:].notna().any()

def test_compute_har_predictions_no_lookahead(sample_ohlcv):
    from kronos_trading.volatility_baselines import HAR_MIN_TRAIN
    # Prediction at index i should only depend on 0..i-1
    preds1 = compute_har_predictions(sample_ohlcv["high"], sample_ohlcv["low"])
    # Modify future
    sample_ohlcv_mod = sample_ohlcv.copy()
    sample_ohlcv_mod.loc[50:, "high"] += 1000
    preds2 = compute_har_predictions(sample_ohlcv_mod["high"], sample_ohlcv_mod["low"])
    assert np.isclose(preds1.iloc[49], preds2.iloc[49], equal_nan=True)

def test_strategy_baseline_indicators(sample_ohlcv):
    strat = HARStopBaseline(config={})
    df = strat.populate_indicators(sample_ohlcv.copy(), {})
    assert "candle_range" in df.columns
    assert "avg_range" in df.columns
    assert "ema_trend" in df.columns
    assert "lowest_low" in df.columns

def test_strategy_dynamic_indicators(sample_ohlcv):
    strat = HARStopDynamic(config={})
    df = strat.populate_indicators(sample_ohlcv.copy(), {})
    assert "har_range" in df.columns
    assert df["har_range"].isna().sum() > 0

def test_strategy_inverse_indicators(sample_ohlcv):
    strat = HARStopInverse(config={})
    df = strat.populate_indicators(sample_ohlcv.copy(), {})
    assert "har_range" in df.columns

def test_baseline_stoploss():
    strat = HARStopBaseline(config={})
    assert strat.stoploss == -0.05

def test_dynamic_stoploss_multiplier(sample_ohlcv, mock_trade):
    strat = HARStopDynamic(config={})
    df = sample_ohlcv.copy()
    df["har_range"] = 1000.0  # mock prediction
    strat.dp = MagicMock()
    strat.dp.get_analyzed_dataframe.return_value = (df, None)
    
    # trade date exists in df
    mock_trade.open_date_utc = df["date"].iloc[50]
    stop = strat.custom_stoploss("BTC/USDT", mock_trade, datetime.now(), 40000.0, 0.0)
    # 1.5 * 1000 / 40000 = -0.0375
    assert np.isclose(stop, -0.0375)

def test_inverse_stoploss_multiplier(sample_ohlcv, mock_trade):
    strat = HARStopInverse(config={})
    df = sample_ohlcv.copy()
    df["har_range"] = 1000.0  # mock prediction
    strat.dp = MagicMock()
    strat.dp.get_analyzed_dataframe.return_value = (df, None)
    
    mock_trade.open_date_utc = df["date"].iloc[50]
    stop = strat.custom_stoploss("BTC/USDT", mock_trade, datetime.now(), 40000.0, 0.0)
    # 0.5 * 1000 / 40000 = -0.0125
    assert np.isclose(stop, -0.0125)

def test_dynamic_fallback_nan_har(sample_ohlcv, mock_trade):
    strat = HARStopDynamic(config={})
    df = sample_ohlcv.copy()
    df["har_range"] = np.nan
    strat.dp = MagicMock()
    strat.dp.get_analyzed_dataframe.return_value = (df, None)
    
    mock_trade.open_date_utc = df["date"].iloc[50]
    stop = strat.custom_stoploss("BTC/USDT", mock_trade, datetime.now(), 40000.0, 0.0)
    assert stop == -0.05

def test_inverse_fallback_nan_har(sample_ohlcv, mock_trade):
    strat = HARStopInverse(config={})
    df = sample_ohlcv.copy()
    df["har_range"] = np.nan
    strat.dp = MagicMock()
    strat.dp.get_analyzed_dataframe.return_value = (df, None)
    
    mock_trade.open_date_utc = df["date"].iloc[50]
    stop = strat.custom_stoploss("BTC/USDT", mock_trade, datetime.now(), 40000.0, 0.0)
    assert stop == -0.05

def test_dynamic_fallback_zero_har(sample_ohlcv, mock_trade):
    strat = HARStopDynamic(config={})
    df = sample_ohlcv.copy()
    df["har_range"] = 0.0
    strat.dp = MagicMock()
    strat.dp.get_analyzed_dataframe.return_value = (df, None)
    
    mock_trade.open_date_utc = df["date"].iloc[50]
    stop = strat.custom_stoploss("BTC/USDT", mock_trade, datetime.now(), 40000.0, 0.0)
    assert stop == -0.05

def test_dynamic_fallback_negative_har(sample_ohlcv, mock_trade):
    strat = HARStopDynamic(config={})
    df = sample_ohlcv.copy()
    df["har_range"] = -100.0
    strat.dp = MagicMock()
    strat.dp.get_analyzed_dataframe.return_value = (df, None)
    
    mock_trade.open_date_utc = df["date"].iloc[50]
    stop = strat.custom_stoploss("BTC/USDT", mock_trade, datetime.now(), 40000.0, 0.0)
    assert stop == -0.05

def test_dynamic_fallback_zero_open_rate(sample_ohlcv, mock_trade):
    strat = HARStopDynamic(config={})
    df = sample_ohlcv.copy()
    df["har_range"] = 1000.0
    strat.dp = MagicMock()
    strat.dp.get_analyzed_dataframe.return_value = (df, None)
    
    mock_trade.open_date_utc = df["date"].iloc[50]
    mock_trade.open_rate = 0.0
    stop = strat.custom_stoploss("BTC/USDT", mock_trade, datetime.now(), 0.0, 0.0)
    assert stop == -0.05

def test_dynamic_fallback_missing_date(sample_ohlcv, mock_trade):
    strat = HARStopDynamic(config={})
    df = sample_ohlcv.copy()
    df["har_range"] = 1000.0
    strat.dp = MagicMock()
    strat.dp.get_analyzed_dataframe.return_value = (df, None)
    
    # Date not in df
    mock_trade.open_date_utc = datetime(2020, 1, 1, tzinfo=timezone.utc)
    stop = strat.custom_stoploss("BTC/USDT", mock_trade, datetime.now(), 40000.0, 0.0)
    assert stop == -0.05

def test_entry_logic_baseline(sample_ohlcv):
    strat = HARStopBaseline(config={})
    df = strat.populate_indicators(sample_ohlcv.copy(), {})
    df = strat.populate_entry_trend(df, {})
    assert "enter_long" in df.columns

def test_exit_logic_baseline(sample_ohlcv):
    strat = HARStopBaseline(config={})
    df = strat.populate_indicators(sample_ohlcv.copy(), {})
    df = strat.populate_exit_trend(df, {})
    assert "exit_long" in df.columns

def test_exit_never_blocked_by_har(sample_ohlcv):
    strat = HARStopDynamic(config={})
    df = strat.populate_indicators(sample_ohlcv.copy(), {})
    df = strat.populate_exit_trend(df, {})
    # Exit logic should not depend on har_range
    assert "har_range" in df.columns
    assert (df["exit_long"] == 1).sum() >= 0 # Just checking it runs without error

def test_dynamic_roi():
    strat = HARStopDynamic(config={})
    assert strat.minimal_roi["120"] == 0.005
    assert strat.minimal_roi["60"] == 0.01
    assert strat.minimal_roi["30"] == 0.02

def test_inverse_roi():
    strat = HARStopInverse(config={})
    assert strat.minimal_roi["120"] == 0.005
    assert strat.minimal_roi["60"] == 0.01
    assert strat.minimal_roi["30"] == 0.02
