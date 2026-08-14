"""Paper-only Kronos trading research system."""
from .model import (ModelManager, KronosRealPredictor, ModelUnavailableError,
                    DeterministicMockPredictor, REFERENCE_MODEL_NAME,
                    REFERENCE_TOKENIZER_NAME, REFERENCE_MODEL_REVISION,
                    REFERENCE_TOKENIZER_REVISION)
from .pipeline import PredictionPipeline
from .evaluation import (EvaluationConfig, EvaluationRow, EvaluationResult,
                         PredictionEvaluator, compute_metrics, run_evaluation)
from .baselines import (persistence_prediction, previous_direction_prediction,
                        baseline_rows_for, build_model_comparison)
from .robustness import (run_robustness, run_series_robustness,
                         build_consolidated_report, summarize_windows)
from .research_targets import (TargetSpec, TARGET_SPECS, SELECTED_TARGETS,
                               ARCHITECTURE_CHECK, frozen_baseline,
                               frozen_baseline_verified, compute_target_metrics,
                               run_research_experiment)
from .reference_validation import (REFERENCE, run_contract_comparison,
                                   build_validation_report,
                                   upstream_reference_constants)

__all__ = [
    "ModelManager",
    "KronosRealPredictor",
    "ModelUnavailableError",
    "DeterministicMockPredictor",
    "REFERENCE_MODEL_NAME",
    "REFERENCE_TOKENIZER_NAME",
    "REFERENCE_MODEL_REVISION",
    "REFERENCE_TOKENIZER_REVISION",
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
    "TargetSpec",
    "TARGET_SPECS",
    "SELECTED_TARGETS",
    "ARCHITECTURE_CHECK",
    "frozen_baseline",
    "frozen_baseline_verified",
    "compute_target_metrics",
    "run_research_experiment",
    "REFERENCE",
    "run_contract_comparison",
    "build_validation_report",
    "upstream_reference_constants",
]
