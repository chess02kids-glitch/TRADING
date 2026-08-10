"""Coherent PAPER-only CLI; existing fetch CLI remains unchanged."""
import argparse,json,sqlite3,sys,time
from pathlib import Path
from .types import Candle
from .model import ModelManager,DeterministicMockPredictor
from .pipeline import PredictionPipeline
from .backtest import Backtester
ROOT=Path(__file__).resolve().parents[1]
def candles(db,symbol,tf):
 c=sqlite3.connect(db);rows=c.execute('SELECT timestamp_ms,open,high,low,close,volume FROM ohlcv_raw WHERE exchange=? AND symbol=? AND timeframe=? ORDER BY timestamp_ms',('binance',symbol,tf)).fetchall();return [Candle(*r) for r in rows]
def main(argv=None):
 p=argparse.ArgumentParser(prog='kronos-trading'); sub=p.add_subparsers(dest='cmd',required=True)
 for x in ('predict','backtest'): q=sub.add_parser(x);q.add_argument('--db',default=str(ROOT/'data/db/kronos_trading.db'));q.add_argument('--symbol',default='BTC/USDT');q.add_argument('--timeframe',default='1h');q.add_argument('--context',type=int,default=64);q.add_argument('--mock',action='store_true')
 a=p.parse_args(argv); xs=candles(a.db,a.symbol,a.timeframe); pred=DeterministicMockPredictor() if a.mock else ModelManager().load()
 if not a.mock: print('Model unavailable or unsupported; use --mock only for offline testing.',file=sys.stderr);return 2
 pipe=PredictionPipeline(pred)
 if a.cmd=='predict':out=pipe.predict(a.symbol,a.timeframe,xs,a.context,1,int(time.time()*1000)).asdict()
 else:out=Backtester(pipe).run(a.symbol,a.timeframe,xs,a.context)
 print(json.dumps(out,indent=2,default=str));return 0
if __name__=='__main__':raise SystemExit(main())
