import os
import logging
from typing import Tuple, Dict
import pandas as pd
import psycopg

logger = logging.getLogger(__name__)

def load_data_from_db(
    symbol: str, 
    timeframe: str, 
    max_timestamp_ms: int = None
) -> pd.DataFrame:
    """
    Loads OHLCV data directly from Supabase to guarantee identical inputs.
    Returns a dataframe indexed by timestamp_utc (DatetimeIndex).
    """
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        raise ValueError("SUPABASE_DB_URL not found in environment")
        
    query = """
        SELECT timestamp_utc, open, high, low, close, volume, timestamp_ms
        FROM ohlcv_raw
        WHERE symbol = %s AND timeframe = %s
    """
    params = [symbol, timeframe]
    
    if max_timestamp_ms:
        query += " AND timestamp_ms <= %s"
        params.append(max_timestamp_ms)
        
    query += " ORDER BY timestamp_ms ASC"

    with psycopg.connect(db_url, prepare_threshold=None) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            
    if not rows:
        raise ValueError(f"No data found for {symbol} {timeframe}")
        
    df = pd.DataFrame(rows, columns=["timestamp_utc", "open", "high", "low", "close", "volume", "timestamp_ms"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df.set_index("timestamp_utc", inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    
    # Cast to float
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
        
    logger.info(f"Loaded {len(df)} rows for {symbol} {timeframe}")
    return df


def split_data_chronological(df: pd.DataFrame, train_pct: float = 0.6, val_pct: float = 0.2) -> Dict[str, pd.DataFrame]:
    """
    Strictly splits the dataset chronologically.
    Returns a dict with 'train', 'validation', and 'holdout' DataFrames.
    """
    total_len = len(df)
    train_end = int(total_len * train_pct)
    val_end = train_end + int(total_len * val_pct)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    holdout_df = df.iloc[val_end:].copy()
    
    return {
        "train": train_df,
        "validation": val_df,
        "holdout": holdout_df
    }

class ResearchDataLoader:
    def __init__(self, max_timestamp_ms: int = None):
        self.max_timestamp_ms = max_timestamp_ms
        self.cache = {}
        
    def get_split_data(self, symbol: str, timeframe: str) -> Dict[str, pd.DataFrame]:
        cache_key = f"{symbol}_{timeframe}"
        if cache_key not in self.cache:
            df = load_data_from_db(symbol, timeframe, self.max_timestamp_ms)
            self.cache[cache_key] = split_data_chronological(df)
            
        return self.cache[cache_key]
