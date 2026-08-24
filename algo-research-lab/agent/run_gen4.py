"""Generation 4 targeted, reproducible research runner."""
import sys, os, json, hashlib, sqlite3
from datetime import datetime, timezone
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from research.pipeline import SignalSpec, run_all_gates

SEED=20260824; rng=np.random.RandomState(SEED)
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
BTC=pd.read_csv(os.path.join(ROOT,'..','sandbox/pattern_research/cache/BTCUSDT_1h_730d.csv'))
ETH=pd.read_csv(os.path.join(ROOT,'..','sandbox/pattern_research/cache/ETHUSDT_1h_730d.csv'))
for d in (BTC,ETH):
 d.timestamp=pd.to_datetime(d.timestamp,utc=True); d.sort_values('timestamp',inplace=True); d.drop_duplicates('timestamp',inplace=True); d.set_index('timestamp',inplace=True)
CTX={'btc':BTC,'eth':ETH}

def state_machine(entry_cond, exit_cond, maxhold):
 n=len(entry_cond); en=np.zeros(n,dtype=bool); ex=np.zeros(n,dtype=bool); live=False; start=0
 for i in range(n):
  if not live and bool(entry_cond.iloc[i]): en[i]=True; live=True; start=i
  elif live and (bool(exit_cond.iloc[i]) or i-start>=maxhold): ex[i]=True; live=False
 return pd.Series(en,index=entry_cond.index),pd.Series(ex,index=entry_cond.index)

def compile_genome(g, ctx):
 btc,eth=ctx['btc'],ctx['eth']; logb=np.log(btc.close); loge=np.log(eth.close)
 if g['signal_type']=='multi_asset_momentum':
  mom=eth.close.pct_change(int(g['lookback_bars']))
  e,x=state_machine(mom>g['momentum_threshold'],mom<=0,int(g['holding_bars']))
  return SignalSpec(eth.close,e,x,pd.Series(False,index=eth.index),pd.Series(False,index=eth.index),float(min(.99,g['size_pct'])),asset='ETH/USDT')
 w=int(g['zscore_window'])
 if g.get('fix_set')=='A4_hedge_ratio':
  hw=int(g['hedge_ratio_window']); beta=loge.rolling(hw).cov(logb).div(logb.rolling(hw).var()).shift(1); spread=loge-beta*logb
  std=spread.rolling(w,min_periods=w).std()
 else:
  spread=loge-logb; vw=int(g.get('adaptive_vol_window',w)); std=spread.rolling(vw,min_periods=vw).std()
 mean=spread.rolling(w,min_periods=w).mean(); z=(spread-mean)/std
 threshold=g.get('entry_zscore_multiplier',g.get('entry_zscore'))
 # momentum uses upward spread continuation; exit reversion / time stop. strictly no opposing entries.
 e,x=state_machine(z>threshold,z<=0,int(g['holding_bars_max']))
 return SignalSpec(eth.close,e,x,pd.Series(False,index=eth.index),pd.Series(False,index=eth.index),float(g['size_pct']),leg_multiplier=2.0,asset='ETH/BTC proxy',meta={'z':z})

def gid(g): return hashlib.sha256(json.dumps(g,sort_keys=True,default=str).encode()).hexdigest()[:12]
def flat(res,g):
 m=res.get('metrics',{}); gd=res.get('gate_detail',{})
 return {'generation':4,'genome_id':g['genome_id'],'signal_type':g['signal_type'],'fix_set':g.get('fix_set','B_stability'),'total_trades':m.get('total_trades',0),'profit_factor':m.get('profit_factor',0),'sharpe_ratio':m.get('sharpe',0),'max_drawdown':m.get('max_drawdown_pct',0),'oos_sharpe':m.get('oos_sharpe'),'concentration_score':gd.get('gate3',{}).get('top5_pct'),'robustness_score':m.get('robustness_pass_scenarios'),'stability_score':m.get('stability_score'),'passed_all_gates':res['passed_all_gates'],'gate_failed':res['gate_failed'],'failure_reason':res['failure_reason'],'genome':g,'seed':SEED,'data_source':'sandbox/pattern_research/cache/','data_range_start':str(BTC.index.min()),'data_range_end':str(BTC.index.max()),'funding_type':'SYNTHETIC','created_at':datetime.now(timezone.utc).isoformat()}

def execute(g):
 g['genome_id']=gid(g); r=run_all_gates(g,CTX,compile_genome,SEED); return flat(r,g)
base={'signal_type':'multi_asset_momentum','primary_asset':'ETH/USDT','lookback_bars':164,'momentum_threshold':.023263046816635283,'require_confirmation':False,'holding_bars':48,'size_pct':.5222990858300037,'confirmation_asset':'BTC/USDT'}
# exact isolated perturbation table prior to variants
ptable={}
for k in ['lookback_bars','momentum_threshold','holding_bars','size_pct']:
 vals=[]
 for p in [-.2,-.1,0,.1,.2]:
  q=dict(base); v=q[k]*(1+p); q[k]=max(2,int(round(v))) if isinstance(q[k],int) else v; vals.append(execute(q)['sharpe_ratio'])
 ptable[k]=vals
frag=max(ptable, key=lambda k: max(ptable[k][2]-ptable[k][1], ptable[k][2]-ptable[k][3]))
records=[]
for mult in [.3,.5,.7,.9,1,1.1,1.3,1.5,2,2.5]:
 g=dict(base); v=base[frag]*mult; g[frag]=max(2,int(round(v))) if isinstance(base[frag],int) else v; records.append(execute(g))
for fix in ['A1_scale_out','A2_adaptive','A3_combined','A4_hedge_ratio']:
 for _ in range(10):
  g={'signal_type':'spread_momentum_v2','fix_set':fix}
  if fix=='A1_scale_out': g.update(zscore_window=int(rng.randint(72,241)),entry_zscore=float(rng.uniform(1.2,2)),profit_target_pct=float(rng.uniform(.02,.08)),stop_loss_pct=float(rng.uniform(.02,.05)),holding_bars_max=int(rng.randint(48,169)),size_pct=float(rng.uniform(.1,.3)))
  elif fix=='A2_adaptive': g.update(zscore_window=int(rng.randint(48,169)),entry_zscore_multiplier=float(rng.uniform(1,2)),adaptive_vol_window=int(rng.randint(24,97)),holding_bars_max=int(rng.randint(24,97)),size_pct=float(rng.uniform(.1,.3)))
  elif fix=='A3_combined': g.update(zscore_window=int(rng.randint(72,201)),entry_zscore_multiplier=float(rng.uniform(1.2,1.8)),adaptive_vol_window=int(rng.randint(24,73)),profit_target_pct=float(rng.uniform(.02,.06)),stop_loss_pct=float(rng.uniform(.02,.04)),holding_bars_max=int(rng.randint(36,121)),size_pct=float(rng.uniform(.1,.25)))
  else: g.update(zscore_window=int(rng.randint(48,201)),entry_zscore=float(rng.uniform(1.2,2)),hedge_ratio_window=int(rng.randint(72,337)),profit_target_pct=float(rng.uniform(.02,.06)),holding_bars_max=int(rng.randint(36,121)),size_pct=float(rng.uniform(.1,.3)))
  records.append(execute(g))
# logging
out=os.path.join(ROOT,'data/gen4_results.jsonl'); open(out,'w').write('\n'.join(json.dumps(x,default=str) for x in records)+'\n')
db=sqlite3.connect(os.path.join(ROOT,'data/research_generations.sqlite')); db.execute('create table if not exists gen4_results (genome_id text primary key, record text)'); db.executemany('insert or replace into gen4_results values (?,?)',[(r['genome_id'],json.dumps(r,default=str)) for r in records]); db.commit()
json.dump({'perturbation_table':ptable,'fragile_parameter':frag,'records':records},open(os.path.join(ROOT,'data/gen4_run.json'),'w'),indent=2,default=str)
for n,r in enumerate(records,1): print(f"[{n:2d}/50] {r['genome_id']} | {r['signal_type'][:20]:20s} | trades={r['total_trades']:4d} | PF={r['profit_factor']:5.2f} | Sharpe={r['sharpe_ratio']:5.2f} | {r['gate_failed'] or 'PASS ALL 5 GATES'}")
print('FRAGILE PARAMETER=',frag); print('PERTURBATION',ptable)
