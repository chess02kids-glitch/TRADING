import math
from .core import SignalEngine,PaperBroker,metrics
class Backtester:
 def __init__(self,pipeline,signal=None):self.pipeline=pipeline;self.signal=signal or SignalEngine()
 def run(self,symbol,timeframe,candles,context=64,horizon=1):
  broker=PaperBroker();equity=[broker.initial];pred=[]
  for i in range(context,len(candles)-horizon):
   # now is next candle open: only candles [:i] are closed/visible.
   p=self.pipeline.predict(symbol,timeframe,candles[:i],context,horizon,candles[i].timestamp_ms);pred.append((p,candles[i].close)); broker.execute(self.signal.generate(p),candles[i].open);equity.append(broker.equity({symbol:candles[i].close}))
  errs=[p.predicted_close-a for p,a in pred];return {'forecast':{'mae':sum(map(abs,errs))/len(errs) if errs else None,'rmse':math.sqrt(sum(x*x for x in errs)/len(errs)) if errs else None,'directional_accuracy':sum((p.expected_return>0)==(a>0) for p,a in pred)/len(pred) if pred else None,'samples':len(pred)},'trading':metrics(equity,broker.trades),'equity':equity}
