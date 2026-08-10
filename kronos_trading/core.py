"""Signals, deterministic risk controls, paper-only execution and evaluation."""
from dataclasses import dataclass,asdict
from typing import Dict,List
import sqlite3,json,uuid,math
from .types import Signal
@dataclass
class RiskConfig:
 max_position_pct:float=.4; max_portfolio_pct:float=.6; max_daily_loss_pct:float=.03; max_drawdown_pct:float=.15; max_positions:int=2; cooldown_candles:int=6
class SignalEngine:
 def __init__(self,min_expected_return=.002, fee=.001, slippage=.001): self.edge=min_expected_return+2*(fee+slippage)
 def generate(self,p):
  if p.expected_return>self.edge:return Signal(p.symbol,p.prediction_timestamp_ms,'LONG',p.expected_return,'net edge exceeds costs')
  if p.expected_return<-self.edge:return Signal(p.symbol,p.prediction_timestamp_ms,'SHORT',p.expected_return,'net edge exceeds costs')
  return Signal(p.symbol,p.prediction_timestamp_ms,'HOLD',p.expected_return,'edge below costs')
class RiskManager:
 def __init__(self,cfg=RiskConfig()):self.cfg=cfg
 def check(self,signal,equity,exposure,drawdown,daily_pnl,open_positions,last_signal=None):
  if signal.side=='HOLD': return False,'HOLD'
  if equity<=0:return False,'INSUFFICIENT_BALANCE'
  if drawdown>=self.cfg.max_drawdown_pct:return False,'MAX_DRAWDOWN'
  if daily_pnl<=-equity*self.cfg.max_daily_loss_pct:return False,'MAX_DAILY_LOSS'
  if open_positions>=self.cfg.max_positions:return False,'MAX_POSITIONS'
  if exposure>=self.cfg.max_portfolio_pct:return False,'MAX_EXPOSURE'
  if last_signal and signal.timestamp_ms-last_signal<self.cfg.cooldown_candles*3600000:return False,'COOLDOWN'
  return True,'APPROVED'
class StateStore:
 def __init__(self,path):
  self.conn=sqlite3.connect(path); self.conn.execute('CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,kind TEXT,ts INTEGER,payload TEXT)');self.conn.commit()
 def record(self,kind,ts,payload,event_id=None):
  event_id=event_id or f'{kind}:{ts}:{payload.get("symbol","")}'
  cursor=self.conn.execute('INSERT OR IGNORE INTO events VALUES(?,?,?,?)',(event_id,kind,ts,json.dumps(payload,sort_keys=True)));self.conn.commit();return cursor.rowcount == 1
 def events(self,kind=None):
  q='SELECT payload FROM events'+(' WHERE kind=?' if kind else ''); return [json.loads(x[0]) for x in self.conn.execute(q,([kind] if kind else []))]
class PaperBroker:
 """No CCXT import and no order API: this class is paper execution only."""
 def __init__(self,capital=100.,fee=.001,slippage=.001,store=None):self.cash=capital;self.initial=capital;self.fee=fee;self.slippage=slippage;self.positions={};self.trades=[];self.store=store
 def execute(self,signal,price,quantity=1.):
  if signal.side=='HOLD':return None
  key=f'paper:{signal.symbol}:{signal.timestamp_ms}'
  if self.store and not self.store.record('signal',signal.timestamp_ms,asdict(signal),key):return None
  side=signal.side; fill=price*(1+self.slippage if side=='LONG' else 1-self.slippage); cost=fill*quantity; fee=cost*self.fee
  if signal.symbol not in self.positions:
   # Conservative paper margin: both directions require notional collateral.
   if cost+fee>self.cash:return {'rejected':'INSUFFICIENT_BALANCE'}
   if side=='LONG': self.cash-=cost+fee
   else: self.cash-=fee  # sale proceeds are retained in collateral, not spendable cash
   self.positions[signal.symbol]={'side':side,'qty':quantity,'entry':fill,'fees':fee,'collateral':cost};out={'action':'OPEN','fill':fill,'fee':fee}
  else:
   p=self.positions.pop(signal.symbol); pnl=(fill-p['entry'])*p['qty']*(1 if p['side']=='LONG' else -1)-fee-p['fees']
   self.cash += (p['collateral'] if p['side']=='SHORT' else 0) + (cost-fee if p['side']=='LONG' else 0)
   if p['side']=='SHORT': self.cash += pnl
   out={'action':'CLOSE','fill':fill,'fee':fee,'pnl':pnl};self.trades.append(out)
  if self.store:self.store.record('paper_order',signal.timestamp_ms,{'symbol':signal.symbol,**out},key+':order')
  return out
 def equity(self,prices):
  total=self.cash
  for s,p in self.positions.items():
   price=prices.get(s,p['entry'])
   total += p['qty']*price if p['side']=='LONG' else p['collateral']+(p['entry']-price)*p['qty']
  return total
def metrics(equity,trades):
 returns=[equity[i]/equity[i-1]-1 for i in range(1,len(equity)) if equity[i-1]]; peak=equity[0] if equity else 0;dd=0
 for x in equity:peak=max(peak,x);dd=max(dd,(peak-x)/peak if peak else 0)
 wins=[t.get('pnl',0) for t in trades if t.get('pnl',0)>0]; losses=[-t.get('pnl',0) for t in trades if t.get('pnl',0)<0]
 return {'total_return':equity[-1]/equity[0]-1 if len(equity)>1 else 0,'max_drawdown':dd,'trades':len(trades),'win_rate':len(wins)/len(trades) if trades else 0,'profit_factor':sum(wins)/sum(losses) if losses else None,'sharpe':(sum(returns)/len(returns))/((sum((r-sum(returns)/len(returns))**2 for r in returns)/len(returns))**.5) if len(returns)>1 else None}
