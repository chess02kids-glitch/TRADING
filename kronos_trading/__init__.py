"""Paper-only Kronos trading research system."""
from .model import (ModelManager, KronosRealPredictor, ModelUnavailableError,
                    DeterministicMockPredictor)
from .pipeline import PredictionPipeline
from .evaluation import (EvaluationConfig, EvaluationRow, EvaluationResult,
                         PredictionEvaluator, compute_metrics, run_evaluation)
from .baselines import (persistence_prediction, previous_direction_prediction,
                        baseline_rows_for, build_model_comparison)
from .robustness import (run_robustness, run_series_robustness,
                         build_consolidated_report, summarize_windows)

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
    "persistence_prediction",
    "previous_direction_prediction",
    "baseline_rows_for",
    "build_model_comparison",
    "run_robustness",
    "run_series_robustness",
    "build_consolidated_report",
    "summarize_windows",
]
