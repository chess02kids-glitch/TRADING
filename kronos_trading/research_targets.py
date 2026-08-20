"""Phase 5 - model/target research (target formulation, NOT optimization).

Phase 4 established (frozen, see ``docs/phase4_baseline_experiment.json``) that
the current Kronos-small configuration does not demonstrate incremental
price-prediction value over persistence for the original OHLCV close target.
This module researches whether that result is a property of the *target
formulation* rather than of the model itself.

Design principles enforced here:

* the frozen Phase 4 experiment is immutable (hash-locked) and never
  overwritten;
* exactly one variable changes per research target (the derived target and its
  horizon), never the model, revision, tokenizer, context length, deterministic
  recipe, threshold, baselines, or no-lookahead rules;
* every target preserves the upstream Kronos contract (raw OHLCV in, OHLCV out)
  - classification is deliberately NOT forced onto a sequence forecaster;
* no hyperparameter search, no window cherry-picking, no tuning.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from .evaluation import PredictionEvaluator, direction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_BASELINE_PATH = PROJECT_ROOT / "docs" / "phase4_baseline_experiment.json"
FROZEN_BASELINE_SHA256 = "6abbfa8dfc360546ce23d309a04991fbebe48cda0cb2b8d711ba77b316fab8ba"

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Frozen baseline (immutable)
# --------------------------------------------------------------------------- #
def frozen_baseline() -> Dict[str, Any]:
    """Load the frozen Phase 4 baseline experiment (read-only)."""
    with open(FROZEN_BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def frozen_baseline_hash() -> str:
    """SHA-256 over the canonical (configuration, results) of the frozen record."""
    data = frozen_baseline()
    canonical = json.dumps(
        {"configuration": data["configuration"], "results": data["results"]},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def frozen_baseline_verified() -> bool:
    """True iff the frozen record still matches its lock hash (no accidental edits)."""
    return frozen_baseline_hash() == FROZEN_BASELINE_SHA256


# --------------------------------------------------------------------------- #
# Architecture check (upstream facts, cited)
# --------------------------------------------------------------------------- #
ARCHITECTURE_CHECK = {
    "finding": (
        "Kronos is an autoregressive sequence-to-sequence forecaster of the "
        "6-dimensional (open, high, low, close, volume, amount) vector. It has "
        "no classification head; output is continuous OHLCV reconstructed from "
        "quantized tokens."
    ),
    "evidence": [
        "KronosPredictor.price_cols = ['open','high','low','close']; "
        "vol_col='volume'; amt_vol='amount' (Kronos/model/kronos.py:489-491).",
        "predict() requires the price columns (524-525), derives amount as "
        "volume * mean price (531-532), z-scores the INPUT sequence with its "
        "own per-sequence mean/std (544-547), clips to [-5,5], decodes tokens "
        "back to values and rescales by (x_std+1e-5)+x_mean (556), returning a "
        "DataFrame of open/high/low/close/volume/amount (558).",
        "auto_regressive_inference() (389) generates pred_len future tokens "
        "autoregressively and reconstructs continuous values via the tokenizer "
        "decoder.",
    ],
    "consequence": (
        "Target reformulations must preserve the raw-OHLCV-in / OHLCV-out "
        "contract. A direction-classification target (candidate D) has no "
        "native class head and would require discarding the trained decoder, "
        "which is not justified; direction is instead DERIVED from predicted "
        "returns (as in Phase 4)."
    ),
    "allowed_changes": [
        "derived target metric only (return / residual / range / normalized return)",
        "pred_len (horizon) - natively supported by predict()",
        "high/low-derived range - a native output sub-space",
    ],
}


# --------------------------------------------------------------------------- #
# Selected target formulations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    name: str
    candidate: str
    definition: str
    justification: str
    contract_preservation: str
    reversible: str
    primary_metric: str
    baselines: Tuple[str, ...]
    horizon: int

    def asdict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


TARGET_SPECS: Dict[str, TargetSpec] = {
    "multi_period_return": TargetSpec(
        target_id="multi_period_return",
        name="Multi-period return (next 4 candles)",
        candidate="E",
        definition=(
            "predicted_return = predicted_close[T+4] / close[T] - 1 versus "
            "actual_return = actual_close[T+4] / close[T] - 1. Kronos generates "
            "4 steps autoregressively (pred_len=4) and the last step's close is "
            "used. horizon=4."
        ),
        justification=(
            "Kronos is a multi-step autoregressive forecaster, yet the frozen "
            "experiment used only horizon=1, where 1h returns are dominated by "
            "high-frequency noise. A 4-step target aggregates signal, reduces "
            "noise, and tests whether the model's value emerges at longer "
            "horizons (the upstream examples use pred_len up to 120)."
        ),
        contract_preservation=(
            "Preserved: raw OHLCV in, OHLCV out. Only pred_len/horizon changes, "
            "which upstream predict() supports natively."
        ),
        reversible="Yes: predicted_close[T+4] = close[T] * (1 + return).",
        primary_metric="return MAE/RMSE and return correlation",
        baselines=("persistence", "previous_direction"),
        horizon=4,
    ),
    "range_volatility": TargetSpec(
        target_id="range_volatility",
        name="Next-candle range / volatility",
        candidate="F",
        definition=(
            "predicted_range = predicted_high - predicted_low versus "
            "actual_range = actual_high - actual_low for the next candle. "
            "horizon=1."
        ),
        justification=(
            "Kronos natively outputs high and low, so intra-candle range is a "
            "supported sub-target that the frozen experiment never scored. It "
            "isolates volatility structure from direction and level, asking "
            "whether the model captures dispersion even if it does not capture "
            "direction."
        ),
        contract_preservation=(
            "Preserved: raw OHLCV in/out; high/low are already part of the "
            "upstream output."
        ),
        reversible=(
            "Not reversible to price (a projection of the high/low pair), but it "
            "is a supported output sub-space, not a model modification."
        ),
        primary_metric="range MAE/RMSE and range correlation",
        baselines=("persistence",),  # previous-direction is N/A for a non-directional target
        horizon=1,
    ),
    "vol_normalized_return": TargetSpec(
        target_id="vol_normalized_return",
        name="Volatility-normalized next-period return",
        candidate="B",
        definition=(
            "next-candle return divided by a past-only volatility scale: "
            "scale = std(context returns) * sqrt(horizon). Predicted and actual "
            "normalized returns use the same per-row scale. horizon=1."
        ),
        justification=(
            "Crypto returns are heteroskedastic; normalizing by trailing "
            "realized volatility yields a more stationary target and down-"
            "weights high-volatility regimes in the aggregate error, testing "
            "whether Kronos's return ranking is more informative in "
            "volatility-adjusted terms."
        ),
        contract_preservation=(
            "Preserved: raw OHLCV is still fed to Kronos (it z-scores inputs "
            "internally). Normalization is applied only to the derived target "
            "outside the model, using past-only context data."
        ),
        reversible=(
            "Yes: multiply by the stored per-row scale to recover the raw return."
        ),
        primary_metric="normalized return MAE/RMSE and correlation",
        baselines=("persistence", "previous_direction"),
        horizon=1,
    ),
}

# Selected 2-3 defensible formulations (A and C are subsumed by the frozen
# return metrics - see research report; D is rejected by the architecture check).
SELECTED_TARGETS: List[TargetSpec] = [
    TARGET_SPECS["multi_period_return"],
    TARGET_SPECS["range_volatility"],
    TARGET_SPECS["vol_normalized_return"],
]


# --------------------------------------------------------------------------- #
# Target metric helpers
# --------------------------------------------------------------------------- #
def _pairs(a, b):
    return [(x, y) for x, y in zip(a, b) if x is not None and y is not None]


def _mae_pairs(a, b):
    pairs = _pairs(a, b)
    if not pairs:
        return None
    return statistics.fmean(abs(x - y) for x, y in pairs)


def _rmse_pairs(a, b):
    pairs = _pairs(a, b)
    if not pairs:
        return None
    return math.sqrt(statistics.fmean((x - y) ** 2 for x, y in pairs))


def _pearson(a, b):
    pairs = _pairs(a, b)
    if len(pairs) < 2:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= EPS or vy <= EPS:
        return None
    return cov / math.sqrt(vx * vy)


def _dir_accuracy(rows, threshold):
    nonflat = [r for r in rows if direction(r.actual_return, threshold) != 0]
    if not nonflat:
        return None
    return sum(1 for r in nonflat if r.directional_correct) / len(nonflat)


def _compare(kronos: Dict, baseline: Dict, metric: str, lower_is_better: bool):
    k = kronos.get(metric)
    b = baseline.get(metric)
    if k is None or b is None:
        return {"delta": None, "winner": None}
    delta = k - b
    if delta == 0:
        winner = "tie"
    elif (delta < 0) == lower_is_better:
        winner = "kronos"
    else:
        winner = "baseline"
    return {"delta": delta, "winner": winner}


# --------------------------------------------------------------------------- #
# Per-target metric computation (derived from the SAME rows/timestamps)
# --------------------------------------------------------------------------- #
def _return_metrics(rows: List[Any], threshold: float) -> Dict[str, Any]:
    pred = [r.predicted_return for r in rows]
    act = [r.actual_return for r in rows]
    return {
        "return_mae": _mae_pairs(pred, act),
        "return_rmse": _rmse_pairs(pred, act),
        "return_correlation": _pearson(pred, act),
        "mean_predicted_return": (statistics.fmean([x for x in pred if x is not None])
                                  if any(x is not None for x in pred) else None),
        "mean_actual_return": statistics.fmean(act) if act else None,
        "directional_accuracy": _dir_accuracy(rows, threshold),
        "sample_size": len(rows),
    }


def _range_metrics(rows: List[Any], kind: str) -> Dict[str, Any]:
    act = [r.actual_high - r.actual_low for r in rows]
    if kind == "kronos":
        pred = [(r.predicted_high - r.predicted_low)
                if (r.predicted_high is not None and r.predicted_low is not None)
                else None for r in rows]
    elif kind == "persistence":
        pred = [r.context_last_range for r in rows]
    else:
        raise ValueError("range metrics only defined for kronos / persistence")
    return {
        "range_mae": _mae_pairs(pred, act),
        "range_rmse": _rmse_pairs(pred, act),
        "range_correlation": _pearson(pred, act),
        "sample_size": len(_pairs(pred, act)),
    }


def _norm_return_metrics(rows: List[Any]) -> Dict[str, Any]:
    pred, act = [], []
    for r in rows:
        vol = r.context_return_vol
        if vol is None or vol <= EPS:
            continue
        if r.predicted_return is None:
            continue
        scale = vol * math.sqrt(max(1, r.horizon))
        pred.append(r.predicted_return / scale)
        act.append(r.actual_return / scale)
    return {
        "norm_return_mae": _mae_pairs(pred, act),
        "norm_return_rmse": _rmse_pairs(pred, act),
        "norm_return_correlation": _pearson(pred, act),
        "sample_size": len(pred),
    }


def _return_comparison(kronos, persistence, previous_direction):
    return {
        "vs_persistence": {
            "return_mae": _compare(kronos, persistence, "return_mae", True),
            "return_rmse": _compare(kronos, persistence, "return_rmse", True),
            "return_correlation": _compare(kronos, persistence, "return_correlation", False),
            "directional_accuracy": _compare(kronos, persistence, "directional_accuracy", False),
        },
        "vs_previous_direction": {
            "return_mae": _compare(kronos, previous_direction, "return_mae", True),
            "return_rmse": _compare(kronos, previous_direction, "return_rmse", True),
            "return_correlation": _compare(kronos, previous_direction, "return_correlation", False),
            "directional_accuracy": _compare(kronos, previous_direction, "directional_accuracy", False),
        },
    }


def _norm_comparison(kronos, persistence, previous_direction):
    return {
        "vs_persistence": {
            "norm_return_mae": _compare(kronos, persistence, "norm_return_mae", True),
            "norm_return_rmse": _compare(kronos, persistence, "norm_return_rmse", True),
            "norm_return_correlation": _compare(kronos, persistence, "norm_return_correlation", False),
        },
        "vs_previous_direction": {
            "norm_return_mae": _compare(kronos, previous_direction, "norm_return_mae", True),
            "norm_return_rmse": _compare(kronos, previous_direction, "norm_return_rmse", True),
            "norm_return_correlation": _compare(kronos, previous_direction, "norm_return_correlation", False),
        },
    }


def _range_comparison(kronos, persistence):
    return {
        "vs_persistence": {
            "range_mae": _compare(kronos, persistence, "range_mae", True),
            "range_rmse": _compare(kronos, persistence, "range_rmse", True),
            "range_correlation": _compare(kronos, persistence, "range_correlation", False),
        },
        "vs_previous_direction": None,  # N/A for a non-directional target
    }


def compute_target_metrics(target_id: str, rows: List[Any],
                           baseline_rows: Dict[str, List[Any]],
                           direction_threshold: float) -> Dict[str, Any]:
    """Compute target-specific metrics + baselines from already-evaluated rows.

    ``rows`` and ``baseline_rows`` come from the SAME chronological window, so
    all systems share identical prediction timestamps and none can see the
    future.
    """
    persistence_rows = baseline_rows["persistence"]
    previous_direction_rows = baseline_rows["previous_direction"]

    if target_id == "multi_period_return":
        kronos = _return_metrics(rows, direction_threshold)
        persistence = _return_metrics(persistence_rows, direction_threshold)
        previous_direction = _return_metrics(previous_direction_rows, direction_threshold)
        horizon = rows[0].horizon if rows else None
        return {
            "target_id": target_id,
            "kronos": kronos,
            "persistence": persistence,
            "previous_direction": previous_direction,
            "comparison": _return_comparison(kronos, persistence, previous_direction),
            "notes": [
                "return is measured over horizon=%s candles" % horizon,
                "persistence predicts zero return, so its correlation is undefined",
            ],
        }

    if target_id == "range_volatility":
        kronos = _range_metrics(rows, "kronos")
        persistence = _range_metrics(persistence_rows, "persistence")
        return {
            "target_id": target_id,
            "kronos": kronos,
            "persistence": persistence,
            "previous_direction": None,
            "comparison": _range_comparison(kronos, persistence),
            "notes": [
                "previous-direction is N/A for a non-directional target",
                "persistence predicts the last observed candle range",
            ],
        }

    if target_id == "vol_normalized_return":
        kronos = _norm_return_metrics(rows)
        persistence = _norm_return_metrics(persistence_rows)
        previous_direction = _norm_return_metrics(previous_direction_rows)
        return {
            "target_id": target_id,
            "kronos": kronos,
            "persistence": persistence,
            "previous_direction": previous_direction,
            "comparison": _norm_comparison(kronos, persistence, previous_direction),
            "notes": [
                "per-row scale = std(context returns) * sqrt(horizon), past-only",
                "rows with zero/undefined context volatility are excluded",
            ],
        }

    raise ValueError("unknown target_id: %r" % target_id)


# --------------------------------------------------------------------------- #
# End-to-end research experiment
# --------------------------------------------------------------------------- #
def _run_all_series(predictor, config, series, load_candles) -> Dict[Tuple[str, str], Dict]:
    out = {}
    for symbol, timeframe in series:
        evaluator = PredictionEvaluator(predictor, config, symbol, timeframe)
        windows, window_info = evaluator.evaluate_windows(
            load_candles(symbol, timeframe))
        out[(symbol, timeframe)] = {"window_info": window_info, "windows": windows}
    return out


def run_research_experiment(predictor, base_config, series: List[Tuple[str, str]],
                            load_candles: Callable[[str, str], List[Any]],
                            targets: List[TargetSpec] = None) -> Dict[str, Any]:
    """Run the target-formulation research matrix.

    The base (horizon=1) evaluator is run once and reused for the
    range/volatility and normalized-return targets (no extra inference). A
    single additional horizon=4 pass serves the multi-period return target.
    Each target therefore changes exactly one variable (the derived target, and
    for multi-period return the native horizon).
    """
    targets = targets or SELECTED_TARGETS
    horizons = sorted({t.horizon for t in targets})
    results_by_horizon = {}
    for h in horizons:
        cfg = dataclasses.replace(base_config, horizon=h)
        results_by_horizon[h] = _run_all_series(predictor, cfg, series, load_candles)

    threshold = base_config.direction_threshold
    per_target = {}
    for t in targets:
        per_series = {}
        for (symbol, timeframe), res in results_by_horizon[t.horizon].items():
            per_series[symbol + " " + timeframe] = {
                "window_info": res["window_info"],
                "windows": {
                    name: compute_target_metrics(
                        t.target_id, wres.rows, wres.baseline_rows, threshold)
                    for name, wres in res["windows"].items()
                },
            }
        per_target[t.target_id] = {"spec": t.asdict(), "series": per_series}

    return {
        "kind": "phase5_research_targets",
        "frozen_baseline": frozen_baseline(),
        "frozen_baseline_sha256": frozen_baseline_hash(),
        "frozen_baseline_verified": frozen_baseline_verified(),
        "architecture_check": ARCHITECTURE_CHECK,
        "configuration": base_config.asdict(),
        "targets": per_target,
        "notes": [
            "exactly one variable changes per target: the derived target (and "
            "the native horizon for multi-period return); model, revision, "
            "tokenizer, context, threshold, baselines and no-lookahead rules "
            "are unchanged",
            "no hyperparameter search, no window cherry-picking, no tuning",
            "statistical significance is NOT trading profitability",
        ],
    }
