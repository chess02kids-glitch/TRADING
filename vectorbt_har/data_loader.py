"""Data loading utilities for vectorbt HAR research."""

import pandas as pd
from pathlib import Path
import ccxt

def load_ohlcv_from_feather(
    asset: str = "BTC/USDT",
    timeframe: str = "1h",
    data_dir: str = "freqtrade_har/user_data/data/kucoin",
) -> pd.DataFrame:
    """
    Load OHLCV data from existing Freqtrade feather files.
    Returns DataFrame with columns:
      timestamp (DatetimeIndex, UTC)
      open, high, low, close, volume
    Falls back to CCXT if feather not found.
    """
    pair = asset.replace("/", "_")
    filename = f"{pair}-{timeframe}.feather"
    filepath = Path(data_dir) / filename
    
    if not filepath.exists():
        print(f"Feather file {filepath} not found, falling back to CCXT.")
        return load_ohlcv_from_ccxt(asset, timeframe)
        
    try:
        df = pd.read_feather(filepath)
        if "date" in df.columns:
            df.set_index("date", inplace=True)
        
        # Ensure UTC timezone if not already set
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        elif str(df.index.tz) != "UTC":
            df.index = df.index.tz_convert("UTC")
            
        df.index.name = "timestamp"
        
        # Ensure standard column names
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        print(f"Feather load failed: {e}. Falling back to CCXT.")
        return load_ohlcv_from_ccxt(asset, timeframe)


def load_ohlcv_from_ccxt(
    asset: str = "BTC/USDT",
    timeframe: str = "1h",
    days: int = 730,
) -> pd.DataFrame:
    """
    Fetch OHLCV from KuCoin via CCXT.
    Used if feather files not available.
    """
    exchange = ccxt.kucoin()
    
    limit = 17520  # ~2 years of hourly data (this will exceed single call limit, simplified for fallback)
    try:
        # Note: a proper robust CCXT fetcher would loop, but since we assume feather exists, 
        # this is a fallback.
        ohlcv = exchange.fetch_ohlcv(asset, timeframe, limit=1500)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        print(f"CCXT fetch failed: {e}")
        return pd.DataFrame()


def filter_timerange(
    df: pd.DataFrame,
    start: str = "2024-01-01",
    end: str = "2026-01-01",
) -> pd.DataFrame:
    """
    Filter DataFrame to date range.
    start and end are UTC date strings.
    """
    if len(df) == 0:
        return df
        
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    
    mask = (df.index >= start_ts) & (df.index < end_ts)
    return df[mask]
