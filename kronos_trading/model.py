"""Kronos adapter. Upstream is loaded read-only and never patched."""
import importlib, time
class ModelManager:
 def __init__(self, model_name='NeoQuasar/Kronos-small', allow_cuda=True): self.model_name=model_name; self.allow_cuda=allow_cuda; self.model=None; self.device='cpu'; self.error=None
 def load(self):
  try:
   import torch
   self.device='cuda:0' if self.allow_cuda and torch.cuda.is_available() else 'cpu'
  except ImportError: self.device='cpu'
  # Kronos APIs differ across upstream revisions: discover only known loader entrypoints.
  try:
   mod=importlib.import_module('model.kronos')
   cls=getattr(mod,'Kronos',None)
   if cls and hasattr(cls,'from_pretrained'): self.model=cls.from_pretrained(self.model_name); self.model.to(self.device); self.model.eval()
  except Exception as exc: self.error=str(exc)
  return self
 @property
 def available(self): return self.model is not None
class KronosPredictor:
 def __init__(self, manager): self.manager=manager
 def predict_close(self, normalized):
  """Returns model output only when compatible upstream API is available."""
  if not self.manager.available: raise RuntimeError('Kronos model unavailable: '+str(self.manager.error))
  start=time.perf_counter()
  # Explicit contract avoids guessing an upstream generation API and false predictions.
  if not hasattr(self.manager.model,'predict'): raise RuntimeError('loaded Kronos revision has no supported predict API')
  output=self.manager.model.predict(normalized)
  return float(output), (time.perf_counter()-start)*1000
class DeterministicMockPredictor:
 """Offline-only test predictor; never selected by production CLI unless --mock."""
 version='deterministic-momentum-v1'; device='mock'
 def predict_close(self, normalized):
  last=normalized[-1][3]; prior=normalized[-2][3]; return last+(last-prior),0.0
