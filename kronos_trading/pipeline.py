import time
from .types import Candle, Prediction
from .preprocess import closed,validate_context,normalize_ohlcv,TF
class PredictionPipeline:
 def __init__(self,predictor, model_version=None): self.predictor=predictor; self.model_version=model_version or getattr(predictor,'version','Kronos')
 def predict(self,symbol,timeframe,candles,context_length,horizon,now_ms):
  if horizon != 1: raise ValueError('only one-candle horizon is supported by this adapter')
  ctx=validate_context(closed(candles,timeframe,now_ms),timeframe,context_length)
  value,elapsed=self.predictor.predict_close(normalize_ohlcv(ctx))
  # predictor works in relative units normalized to first close
  predicted=value*ctx[0].close; last=ctx[-1].close
  return Prediction(symbol,timeframe,ctx[-1].timestamp_ms+TF[timeframe],horizon,predicted,(predicted-last)/last,ctx[0].timestamp_ms,ctx[-1].timestamp_ms,int(time.time()*1000),self.model_version,getattr(self.predictor,'device','unknown'),elapsed)
