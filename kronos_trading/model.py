"""Kronos model adapters.

There are exactly two predictors in this system, and they are never silently
interchangeable:

* ``KronosRealPredictor`` - runs the *real* upstream Kronos model
  (``KronosTokenizer`` + ``Kronos`` + upstream ``KronosPredictor``) with real
  weights. If the model, tokenizer, or weights cannot be loaded, it raises
  ``ModelUnavailableError``. It never falls back to the mock.

* ``DeterministicMockPredictor`` - a tiny offline momentum stub used only for
  unit tests and the explicit ``--mock`` CLI flag. It is never selected by the
  real inference path.

The upstream ``Kronos/`` source tree is imported read-only and is never
patched or modified.
"""
from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .preprocess import to_kronos_frame, future_timestamps
from .types import Candle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KRONOS_DIR = PROJECT_ROOT / "Kronos"


class ModelUnavailableError(RuntimeError):
    """Raised when the real Kronos model/tokenizer/weights cannot be loaded."""


def set_seed(seed: int) -> None:
    """Best-effort global seed for deterministic Kronos inference.

    Mirrors the upstream regression-test recipe: Python/NumPy/Torch RNGs plus
    deterministic cuDNN flags when CUDA is present.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve_device(requested: Optional[str] = None) -> str:
    """Resolve the execution device with explicit, non-silent semantics.

    * ``None`` -> auto: ``cuda:0`` if available, else ``cpu`` (CPU fallback).
    * ``cpu`` -> force CPU.
    * ``cuda`` / ``cuda:N`` -> require CUDA; raise if unavailable (an explicit
      CUDA request is never silently downgraded).
    * anything else -> ``ValueError`` (invalid device).
    """
    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    if requested is None:
        return "cuda:0" if cuda else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda" or requested.startswith("cuda:"):
        if not cuda:
            raise ModelUnavailableError(
                "device %r was requested but CUDA is not available in this "
                "environment; choose --device cpu or omit --device for "
                "automatic fallback" % requested)
        return requested if requested.startswith("cuda:") else "cuda:0"
    raise ValueError("invalid device: %r" % requested)


def _import_upstream():
    """Import the upstream Kronos package (read-only) and return its symbols."""
    if str(KRONOS_DIR) not in sys.path:
        sys.path.insert(0, str(KRONOS_DIR))
    if not (KRONOS_DIR / "model" / "kronos.py").exists():
        raise ModelUnavailableError(
            "upstream Kronos source not found at %s (run `git submodule update "
            "--init`)" % KRONOS_DIR)
    from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: F401
    return Kronos, KronosTokenizer, KronosPredictor


def _snapshot_revision(repo_id: str) -> Optional[str]:
    """Resolve the commit hash of the locally-cached model/tokenizer snapshot."""
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(repo_id, "config.json")
        if path:
            parts = Path(path).parts
            if "snapshots" in parts:
                return parts[parts.index("snapshots") + 1]
    except Exception:
        pass
    return None


class ModelManager:
    """Loads the real Kronos model + tokenizer + upstream predictor.

    Guarantees:

    * CUDA when available, CPU fallback otherwise (never a silent mock swap);
    * an explicit ``ModelUnavailableError`` message when the model, tokenizer,
      or weights cannot be loaded;
    * measured (not assumed) parameter count and dtype;
    * clear device reporting.
    """

    def __init__(self,
                 model_name: str = "NeoQuasar/Kronos-small",
                 tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
                 model_revision: Optional[str] = None,
                 tokenizer_revision: Optional[str] = None,
                 device: Optional[str] = None,
                 max_context: int = 512,
                 local_files_only: bool = False,
                 cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.requested_device = device
        self.max_context = max_context
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir

        self.device: str = "cpu"
        self.model = None
        self.tokenizer = None
        self.predictor = None
        self.error: Optional[str] = None
        self.total_params: Optional[int] = None
        self.dtype: Optional[str] = None
        self.resolved_model_revision: Optional[str] = None
        self.resolved_tokenizer_revision: Optional[str] = None

    @staticmethod
    def _is_local_path(name: str) -> bool:
        return bool(name) and (os.sep in name or os.path.exists(name))

    def load(self) -> "ModelManager":
        try:
            self.device = resolve_device(self.requested_device)
            Kronos, KronosTokenizer, KronosPredictor = _import_upstream()

            # A revision only applies to Hub ids, never to local directories.
            tokenizer_rev = None if self._is_local_path(self.tokenizer_name) else self.tokenizer_revision
            model_rev = None if self._is_local_path(self.model_name) else self.model_revision

            tokenizer = KronosTokenizer.from_pretrained(
                self.tokenizer_name,
                revision=tokenizer_rev,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
            model = Kronos.from_pretrained(
                self.model_name,
                revision=model_rev,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
            model.eval()
            tokenizer.eval()

            predictor = KronosPredictor(model, tokenizer,
                                        device=self.device,
                                        max_context=self.max_context)
            self.model = model
            self.tokenizer = tokenizer
            self.predictor = predictor
            self.total_params = int(sum(p.numel() for p in model.parameters()))
            try:
                self.dtype = str(next(model.parameters()).dtype)
            except StopIteration:
                self.dtype = "unknown"
            self.resolved_model_revision = (
                self.model_revision or _snapshot_revision(self.model_name))
            self.resolved_tokenizer_revision = (
                self.tokenizer_revision or _snapshot_revision(self.tokenizer_name))
        except ModelUnavailableError as exc:
            self.error = str(exc)
        except Exception as exc:  # noqa: BLE001 - surfaced to caller, never swallowed
            self.error = self._describe_load_failure(exc)
        return self

    @staticmethod
    def _describe_load_failure(exc: Exception) -> str:
        name = type(exc).__name__
        msg = str(exc)
        if name == "LocalEntryNotFoundError" or "cannot find the requested files" in msg:
            return ("model weights/tokenizer are not available locally and the "
                    "Hugging Face Hub is unreachable. Download them first (e.g. "
                    "python scripts/setup/download_models.py) or pass an "
                    "accessible model path. Underlying error: %s" % msg)
        if "TLS/SSL" in msg or "connection" in msg.lower() or "resolve" in msg.lower():
            return ("could not reach the Hugging Face Hub to load model weights. "
                    "Check network connectivity. Underlying error: %s: %s" % (name, msg))
        return "Kronos model load failed: %s: %s" % (name, msg)

    @property
    def available(self) -> bool:
        return self.predictor is not None and self.error is None

    def report(self) -> Dict:
        return {
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
            "model_revision": self.resolved_model_revision,
            "tokenizer_revision": self.resolved_tokenizer_revision,
            "device": self.device,
            "dtype": self.dtype,
            "total_params": self.total_params,
            "max_context": self.max_context,
            "available": self.available,
            "error": self.error,
        }


@dataclass
class PredictorResult:
    """What the real Kronos model actually produced for one inference call."""
    steps: List[Dict[str, float]] = field(default_factory=list)
    latency_ms: float = 0.0
    peak_vram_bytes: Optional[int] = None


class KronosRealPredictor:
    """Thin, honest wrapper around the upstream ``KronosPredictor``.

    Feeds *raw* OHLCV candles (already validated and closed-candle only) into
    the upstream predictor and returns the exact columns Kronos emits:
    ``open, high, low, close, volume, amount`` per horizon step. Nothing else
    is fabricated.
    """

    def __init__(self, manager: ModelManager):
        if not manager.available:
            raise ModelUnavailableError(
                "Kronos model unavailable: %s" % (manager.error or "not loaded"))
        self.manager = manager

    @property
    def device(self) -> str:
        return self.manager.device

    @property
    def dtype(self) -> Optional[str]:
        return self.manager.dtype

    @property
    def version(self) -> str:
        rev = self.manager.resolved_model_revision
        return self.manager.model_name if not rev else "%s@%s" % (self.manager.model_name, rev)

    @property
    def max_context(self) -> int:
        return self.manager.max_context

    def predict(self,
                candles: List[Candle],
                timeframe: str,
                horizon: int = 1,
                temperature: float = 1.0,
                top_k: int = 0,
                top_p: float = 0.9,
                sample_count: int = 1,
                seed: Optional[int] = None,
                deterministic: bool = False) -> PredictorResult:
        """Run real Kronos inference for ``horizon`` candles.

        ``deterministic=True`` forces the upstream argmax recipe (seed +
        ``top_k=1`` + ``top_p=1.0``) which removes sampling randomness.
        """
        if not self.manager.available:
            raise ModelUnavailableError(
                "Kronos model unavailable: %s" % (self.manager.error or "not loaded"))
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        if deterministic:
            set_seed(seed if seed is not None else 0)
            top_k, top_p, sample_count = 1, 1.0, 1
        elif seed is not None:
            set_seed(seed)

        import torch
        on_cuda = self.device.startswith("cuda")
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()

        df, x_timestamp = to_kronos_frame(candles)
        y_timestamp = future_timestamps(candles[-1].timestamp_ms, timeframe, horizon)

        start = time.perf_counter()
        pred_df = self.manager.predictor.predict(
            df=df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=horizon,
            T=temperature,
            top_k=top_k,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False,
        )
        if on_cuda:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        cols = ["open", "high", "low", "close", "volume", "amount"]
        steps = [dict(zip(cols, [float(v) for v in row])) for row in pred_df.values]

        peak = None
        if on_cuda:
            peak = int(torch.cuda.max_memory_allocated())

        return PredictorResult(steps=steps, latency_ms=elapsed_ms,
                               peak_vram_bytes=peak)


class DeterministicMockPredictor:
    """Offline-only test predictor; never selected by the real path.

    Selected only explicitly (``--mock``) or by unit tests.
    """
    version = 'deterministic-momentum-v1'
    device = 'mock'

    def predict_close(self, normalized):
        last = normalized[-1][3]
        prior = normalized[-2][3]
        return last + (last - prior), 0.0
