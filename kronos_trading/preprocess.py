"""Immutable raw-candle preprocessing; deliberately never fills a gap."""
import math
from .types import Candle
TF={'1h':3600000,'4h':14400000,'1d':86400000}
def closed(candles,timeframe,now_ms):
 return [c for c in candles if c.timestamp_ms+TF[timeframe] <= now_ms]
def validate_context(candles,timeframe,length):
 if len(candles)<length: raise ValueError('insufficient closed candles')
 xs=list(candles[-length:]); step=TF[timeframe]
 for a,b in zip(xs,xs[1:]):
  if b.timestamp_ms-a.timestamp_ms != step: raise ValueError('missing or misordered candle in context')
 for c in xs:
  vals=(c.open,c.high,c.low,c.close,c.volume)
  if not all(math.isfinite(v) for v in vals) or min(c.open,c.high,c.low,c.close)<=0 or c.volume<0: raise ValueError('invalid numeric candle')
 return xs
def normalize_ohlcv(candles):
 """Relative-to-first-close representation, without mutating raw data."""
 base=candles[0].close
 return [[c.open/base,c.high/base,c.low/base,c.close/base,math.log1p(c.volume)] for c in candles]
