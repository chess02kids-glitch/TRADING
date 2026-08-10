"""Structured observability, walk-forward splits, reports and restart-safe paper loop."""
import json,logging,time,hashlib
from dataclasses import asdict
class JsonLogger:
 def __init__(self,name='kronos_trading'):self.log=logging.getLogger(name)
 def event(self,component,status,**fields):self.log.info(json.dumps({'timestamp_ms':int(time.time()*1000),'component':component,'status':status,**fields},sort_keys=True,default=str))
def config_hash(config):return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
def walk_forward(candles,development=.5,validation=.25):
 n=len(candles);a=int(n*development);b=int(n*(development+validation));return {'development':candles[:a],'validation':candles[a:b],'holdout':candles[b:]}
def report(kind,run_id,config,data_range,metrics,warnings=None,errors=None):
 return {'kind':kind,'run_id':run_id,'timestamp_ms':int(time.time()*1000),'configuration_hash':config_hash(config),'data_range':data_range,'metrics':metrics,'warnings':warnings or [],'errors':errors or []}
class PaperLoop:
 """One closed-candle iteration. Scheduler ownership is external for safe shutdown."""
 def __init__(self,pipeline,signals,risk,broker,store,logger=None):self.pipeline=pipeline;self.signals=signals;self.risk=risk;self.broker=broker;self.store=store;self.logger=logger or JsonLogger()
 def process(self,symbol,timeframe,candles,context,now_ms):
  p=self.pipeline.predict(symbol,timeframe,candles,context,1,now_ms);s=self.signals.generate(p)
  # Persist prediction before action. Timestamp key makes restarts idempotent.
  if not self.store.record('prediction',p.prediction_timestamp_ms,p.asdict(),f'prediction:{symbol}:{timeframe}:{p.prediction_timestamp_ms}'):
   return {'status':'duplicate_candle'}
  ok,reason=self.risk.check(s,self.broker.equity({symbol:candles[-1].close}),0,0,0,len(self.broker.positions))
  if not ok:self.store.record('risk_event',s.timestamp_ms,{'symbol':symbol,'reason':reason});return {'status':'rejected','reason':reason}
  result=self.broker.execute(s,candles[-1].close);self.logger.event('paper_loop','ok',symbol=symbol,timeframe=timeframe,action=result);return result
