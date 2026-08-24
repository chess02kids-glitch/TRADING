"""Gen 5: manual honest two-leg simulator and new-structure search."""
import os,sys,json,hashlib,sqlite3
from datetime import datetime,timezone
import numpy as np,pandas as pd
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..')); sys.path.insert(0,ROOT)
BTC=pd.read_csv(os.path.join(ROOT,'../sandbox/pattern_research/cache/BTCUSDT_1h_730d.csv')); ETH=pd.read_csv(os.path.join(ROOT,'../sandbox/pattern_research/cache/ETHUSDT_1h_730d.csv'))
for d in (BTC,ETH): d.timestamp=pd.to_datetime(d.timestamp,utc=True);d.set_index('timestamp',inplace=True)
# Returns equity and closed trade percentage PnLs. Signals at t execute at t+1.
def twoleg(b,e,en,ex,direction,size,fee=.001,slip=.0005):
 cap=1.; eq=[cap]; trades=[]; live=False
 for i in range(1,len(b)):
  if not live and en[i-1]:
   notional=cap*size; eb,ee=b[i],e[i]; cap-=notional*2*(fee+slip); live=True
  elif live and ex[i-1]:
   rp=(e[i]/ee-1)-(b[i]/eb-1) if direction=='long_eth_short_btc' else (b[i]/eb-1)-(e[i]/ee-1)
   pnl=rp*notional-notional*2*(fee+slip); cap+=pnl; trades.append(pnl/notional);live=False
  eq.append(cap)
 return np.array(eq),np.array(trades)
def fixture():
 # signal executes bar 1, exits bar 2: entry values 100/100.
 for eth,btc,expect in [(110,105,.05-4*.0015),(97,104,-.07-4*.0015),(105,105,-4*.0015)]:
  eq,t=twoleg(np.array([100,100,btc]),np.array([100,100,eth]),[1,0,0],[0,1,0],'long_eth_short_btc',1)
  assert abs((eq[-1]-1)-expect)<1e-8,(eq,expect)
 print('TWO-LEG FIXTURES: PASS')
fixture()
def state(entry,exit,hold):
 en=np.zeros(len(entry),bool);ex=np.zeros(len(entry),bool);live=False;s=0
 for i in range(len(entry)):
  if not live and entry[i]:en[i]=1;live=True;s=i
  elif live and (exit[i] or i-s>=hold):ex[i]=1;live=False
 return en,ex
def metrics(eq,tr):
 r=np.diff(eq)/eq[:-1]; sharpe=0 if r.std()==0 else float(r.mean()/r.std()*np.sqrt(24*365)); wins=tr[tr>0].sum();loss=-tr[tr<0].sum(); pf=float(wins/loss) if loss>0 else (99 if wins>0 else 0);dd=float((eq/np.maximum.accumulate(eq)-1).min()*100);return len(tr),pf,sharpe,dd,r
def evaluate(g):
 c=BTC.close.values; eth=ETH.close.values
 if g['signal_type']=='spread_two_leg':
  lb=np.log(c);le=np.log(eth);w=g['zscore_window']; beta=1 if g['hedge_ratio_window']==0 else pd.Series(le).rolling(g['hedge_ratio_window']).cov(pd.Series(lb)).div(pd.Series(lb).rolling(g['hedge_ratio_window']).var()).shift(1).values
  z=(le-beta*lb-pd.Series(le-beta*lb).rolling(w).mean())/pd.Series(le-beta*lb).rolling(w).std();z=z.values
  if g['direction']=='long_eth_short_btc': en,ex=state(z<-g['entry_zscore'],z>-g['exit_zscore'],g['holding_bars_max'])
  else: en,ex=state(z>g['entry_zscore'],z<g['exit_zscore'],g['holding_bars_max'])
  eq,tr=twoleg(c,eth,en,ex,g['direction'],g['size_pct'],g['fee_per_leg'],g['slippage_per_leg'])
 else:
  close=pd.Series(c); ret=close.pct_change();
  if g['signal_type']=='realized_vol_regime':
   vf=ret.rolling(g['vol_window_fast']).std();vs=ret.rolling(g['vol_window_slow']).std();reg=vf<vs if g['regime_type']=='falling' else vf>vs; mean=close.rolling(g['lookback_bars']).mean()
   # long-only implementation for pipeline parity; falling reversion enters after dip, rising momentum after strength
   cond=(close<mean*(1-g['entry_threshold'])) if g['strategy_in_regime']=='reversion' else (close.pct_change(g['lookback_bars'])>g['entry_threshold']);en,ex=state((reg&cond).fillna(False).values,np.zeros(len(c),bool),g['holding_bars'])
  elif g['signal_type']=='open_interest_delta':
   oi=pd.Series(BTC.volume).rolling(24).sum();oc=oi.pct_change(g['oi_change_window']);pc=close.pct_change(g['price_change_window']);
   cond=(oc>g['oi_threshold'])&(pc>g['price_threshold']) if g['signal_mode']=='trend_confirm' else (oc<-g['oi_threshold'])&(pc<-g['price_threshold']);en,ex=state(cond.fillna(False).values,np.zeros(len(c),bool),g['holding_bars'])
  else:
   vr=BTC.volume/pd.Series(BTC.volume).rolling(g['vol_window']).mean();liq=close.pct_change()<-g['liq_threshold'];en,ex=state((liq&(vr>g['vol_multiplier'])).shift(g['bars_after_liq']).fillna(False).values,np.zeros(len(c),bool),g['holding_bars'])
  # same manual single-leg fair costs
  eq,tr=twoleg(c,c,en,ex,'long_eth_short_btc',g['size_pct']) # replace with direct
  cap=1; E=[cap];T=[];live=False
  for i in range(1,len(c)):
   if not live and en[i-1]: n=cap*g['size_pct'];ep=c[i];cap-=n*.0015;live=True
   elif live and ex[i-1]: p=(c[i]/ep-1)*n-n*.0015;cap+=p;T.append(p/n);live=False
   E.append(cap)
  eq,tr=np.array(E),np.array(T)
 n,pf,sh,dd,r=metrics(eq,tr)
 # gates: G1 then 3 chronological OOS blocks; concentration; costs approximated from base actual already
 gate='SCREENING'; reason='LOW_TRADE_COUNT' if n<50 else ('LOW_PROFIT_FACTOR' if pf<1.05 else None)
 oos=None; top=None; stab=None
 if reason is None:
  blocks=np.array_split(r[int(.6*len(r)):],3); ss=[0 if x.std()==0 else x.mean()/x.std()*np.sqrt(24*365) for x in blocks];oos=float(np.mean(ss))
  if sum(x>0 for x in ss)<2 or oos<=0: gate='WALK_FORWARD';reason='FAILED_OOS_CONSISTENCY'
  else:
   pos=tr[tr>0]; total=tr.sum();top=float(np.sort(pos)[-5:].sum()/total) if total>0 and len(pos) else 99
   if total<=0 or (len(pos) and (pos.max()/total>.2 or top>.6)):gate='CONCENTRATION';reason='HIGH_CONCENTRATION'
   else: gate=None
 h=hashlib.sha256(json.dumps(g,sort_keys=True).encode()).hexdigest()[:12];g['genome_id']=h
 return {'generation':5,'genome_id':h,'signal_type':g['signal_type'],'genome':g,'total_trades':n,'profit_factor':pf,'sharpe_ratio':sh,'max_drawdown':dd,'oos_sharpe':oos,'concentration_score':top,'passed_all_gates':False,'gate_failed':gate,'failure_reason':reason,'seed':20260825,'created_at':datetime.now(timezone.utc).isoformat()}
R=[]
def add(g):R.append(evaluate(g))
rng=np.random.RandomState(20260825)
for direction in ['long_eth_short_btc','long_btc_short_eth']:
 for _ in range(10):add({'signal_type':'spread_two_leg','direction':direction,'zscore_window':int(rng.randint(48,337)),'entry_zscore':float(rng.uniform(1,2.5)),'exit_zscore':float(rng.uniform(0,.5)),'hedge_ratio_window':int(rng.choice([0,rng.randint(72,337)])),'holding_bars_max':int(rng.randint(12,169)),'size_pct':float(rng.uniform(.05,.2)),'fee_per_leg':.001,'slippage_per_leg':.0005})
rng=np.random.RandomState(20260826)
for mode in [('falling','reversion'),('rising','momentum')]:
 for _ in range(5):add({'signal_type':'realized_vol_regime','vol_window_fast':int(rng.randint(6,25)),'vol_window_slow':int(rng.randint(24,169)),'regime_type':mode[0],'strategy_in_regime':mode[1],'lookback_bars':int(rng.randint(12,73)),'entry_threshold':float(rng.uniform(.005,.03)),'holding_bars':int(rng.randint(2,25)),'size_pct':float(rng.uniform(.1,.3))})
rng=np.random.RandomState(20260827)
for mode in ['trend_confirm','exhaustion_fade']:
 for _ in range(5):add({'signal_type':'open_interest_delta','oi_change_window':int(rng.randint(4,25)),'oi_threshold':float(rng.uniform(.005,.03)),'price_change_window':int(rng.randint(4,25)),'price_threshold':float(rng.uniform(.003,.02)),'signal_mode':mode,'holding_bars':int(rng.randint(2,25)),'size_pct':float(rng.uniform(.1,.3))})
rng=np.random.RandomState(20260828)
for _ in range(10):add({'signal_type':'liquidation_bounce','liq_threshold':float(rng.uniform(.01,.04)),'vol_multiplier':float(rng.uniform(1.5,4)),'vol_window':int(rng.randint(12,73)),'bars_after_liq':int(rng.randint(1,7)),'holding_bars':int(rng.randint(2,25)),'stop_loss_pct':float(rng.uniform(.01,.03)),'size_pct':float(rng.uniform(.1,.3))})
open(os.path.join(ROOT,'data/gen5_results.jsonl'),'w').write('\n'.join(json.dumps(x) for x in R)+'\n');db=sqlite3.connect(os.path.join(ROOT,'data/research_generations.sqlite'));db.execute('create table if not exists gen5_results (genome_id text primary key, record text)');db.executemany('insert or replace into gen5_results values (?,?)',[(x['genome_id'],json.dumps(x)) for x in R]);db.commit();json.dump(R,open(os.path.join(ROOT,'data/gen5_run.json'),'w'),indent=2)
for i,x in enumerate(R,1):print(f"[{i:2d}/50] {x['genome_id']} | {x['signal_type'][:22]:22s} | trades={x['total_trades']:4d} | PF={x['profit_factor']:5.2f} | Sharpe={x['sharpe_ratio']:5.2f} | {x['gate_failed'] or 'PASS'}")
