from kronos_trading.types import Candle
from kronos_trading.preprocess import validate_context
from kronos_trading.model import DeterministicMockPredictor
from kronos_trading.pipeline import PredictionPipeline
from kronos_trading.core import SignalEngine,RiskManager,PaperBroker,StateStore
from kronos_trading.types import Signal
from kronos_trading.backtest import Backtester
H=3600000
def xs(n=100):return [Candle(i*H,100+i*.1,101+i*.1,99+i*.1,100+i*.1,10) for i in range(n)]
def test_context_rejects_gap():
 a=xs(5);a[3]=Candle(9*H,1,1,1,1,1)
 try:validate_context(a,'1h',5);assert False
 except ValueError:pass
def test_end_to_end_mock_no_lookahead_and_paper(tmp_path):
 p=PredictionPipeline(DeterministicMockPredictor()); pred=p.predict('BTC/USDT','1h',xs(),64,1,80*H)
 assert pred.input_end_ms < pred.prediction_timestamp_ms
 sig=SignalEngine(min_expected_return=0,fee=0,slippage=0).generate(pred); assert sig.side=='LONG'
 assert RiskManager().check(sig,100,0,0,0,0)[0]
 store=StateStore(tmp_path/'state.db'); b=PaperBroker(store=store); assert b.execute(sig,100); assert b.execute(sig,100) is None
 assert Backtester(p).run('BTC/USDT','1h',xs(),64)['forecast']['samples']==35
def test_hold_and_risk_rejection():
 p=PredictionPipeline(DeterministicMockPredictor()).predict('BTC/USDT','1h',xs(),64,1,80*H)
 assert SignalEngine(min_expected_return=1).generate(p).side=='HOLD'
 assert RiskManager().check(Signal('BTC/USDT',1,'LONG',.1,'test'),100,.7,0,0,0)[1]=='MAX_EXPOSURE'
