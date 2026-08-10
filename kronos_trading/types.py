from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
@dataclass(frozen=True)
class Candle:
 timestamp_ms:int; open:float; high:float; low:float; close:float; volume:float
@dataclass
class Prediction:
 symbol:str; timeframe:str; prediction_timestamp_ms:int; horizon:int; predicted_close:float; expected_return:float; input_start_ms:int; input_end_ms:int; generated_at_ms:int; model_version:str; device:str; inference_ms:float; confidence:Optional[float]=None
 def asdict(self): return asdict(self)
@dataclass
class Signal:
 symbol:str; timestamp_ms:int; side:str; expected_return:float; reason:str
