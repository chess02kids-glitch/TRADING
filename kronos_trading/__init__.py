"""Paper-only Kronos trading research system."""
from .model import (ModelManager, KronosRealPredictor, ModelUnavailableError,
                    DeterministicMockPredictor)
from .pipeline import PredictionPipeline
from .evaluation import (EvaluationConfig, EvaluationRow, EvaluationResult,
                         PredictionEvaluator, compute_metrics, run_evaluation)

__all__ = [
    "ModelManager",
    "KronosRealPredictor",
    "ModelUnavailableError",
    "DeterministicMockPredictor",
    "PredictionPipeline",
    "EvaluationConfig",
    "EvaluationRow",
    "EvaluationResult",
    "PredictionEvaluator",
    "compute_metrics",
    "run_evaluation",
]
