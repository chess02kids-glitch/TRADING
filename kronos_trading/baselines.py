"""Phase 4 - naive (no-model) baselines for fair comparison with Kronos.

Two scientifically standard, deterministic baselines, computed on exactly the
same prediction timestamps as the Kronos evaluation (they are derived from the
same validated context, so they can never see the future):

* **persistence / random-walk** — ``predicted_close = last observed close``,
  ``predicted_return = 0``. This is the canonical "no change" benchmark.

* **previous-direction** — extrapolates the last observed return:
  ``previous_return = close[-1] / close[-2] - 1``, predicts the same return,
  and predicts direction with the *same* flatness threshold used for Kronos
  (default 0.0005). This is the canonical "trend continuation" benchmark.

Neither baseline predicts open/high/low/volume (those fields are left ``None``
and are excluded from the comparison metrics, which cover close, returns, and
direction only). The baselines use only candles whose open time is strictly
before the prediction timestamp.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .evaluation import EvaluationRow, direction
from .types import Candle


def persistence_prediction(last_close: float) -> Tuple[float, float]:
    """Random-walk / persistence forecast: (predicted_close, predicted_return)."""
    return last_close, 0.0


def previous_direction_prediction(ctx: List[Candle], threshold: float) -> Tuple[float, float, int]:
    """Trend-continuation forecast from the last two *closed* candles.

    Returns ``(predicted_close, predicted_return, predicted_direction)`` where
    ``predicted_direction`` uses the same flatness threshold as Kronos.
    """
    prev_return = ctx[-1].close / ctx[-2].close - 1.0
    predicted_close = ctx[-1].close * (1.0 + prev_return)
    return predicted_close, prev_return, direction(prev_return, threshold)


def _row_from_template(template: EvaluationRow, *, kind: str,
                       predicted_close: float, predicted_return: float,
                       directional_correct: bool) -> EvaluationRow:
    """Build a baseline row that mirrors the Kronos row's actuals/timestamps."""
    actual_close = template.actual_close
    return EvaluationRow(
        symbol=template.symbol,
        timeframe=template.timeframe,
        context_end_timestamp=template.context_end_timestamp,
        prediction_timestamp=template.prediction_timestamp,
        actual_timestamp=template.actual_timestamp,
        context_length=template.context_length,
        # Baselines predict only close + return; open/high/low/volume are left
        # undefined (None) and excluded from the comparison metrics.
        predicted_open=None,
        predicted_high=None,
        predicted_low=None,
        predicted_close=predicted_close,
        predicted_volume=None,
        actual_open=template.actual_open,
        actual_high=template.actual_high,
        actual_low=template.actual_low,
        actual_close=actual_close,
        actual_volume=template.actual_volume,
        predicted_return=predicted_return,
        actual_return=template.actual_return,
        absolute_close_error=abs(predicted_close - actual_close),
        squared_close_error=(predicted_close - actual_close) ** 2,
        directional_correct=directional_correct,
        model_revision=kind,
        tokenizer_revision=None,
        inference_latency_ms=0.0,
        device='baseline',
        deterministic=True,
        seed=template.seed,
        top_k=template.top_k,
        top_p=template.top_p,
        sample_count=template.sample_count,
        horizon=template.horizon,
        direction_threshold=template.direction_threshold,
    )


def baseline_rows_for(template: EvaluationRow, ctx: List[Candle],
                      threshold: float) -> Tuple[EvaluationRow, Optional[EvaluationRow]]:
    """Produce (persistence_row, previous_direction_row) for one Kronos row.

    ``ctx`` is the validated closed-candle context already used for the Kronos
    prediction (its last candle ends strictly before the target). The
    previous-direction baseline requires at least two context candles; when
    ``len(ctx) < 2`` it is ``None``.
    """
    persistence = _row_from_template(
        template, kind='persistence',
        predicted_close=ctx[-1].close, predicted_return=0.0,
        directional_correct=False)  # a flat prediction never scores direction

    if len(ctx) < 2:
        return persistence, None

    predicted_close, prev_return, pred_dir = previous_direction_prediction(ctx, threshold)
    act_dir = direction(template.actual_return, threshold)
    prev_row = _row_from_template(
        template, kind='previous_direction',
        predicted_close=predicted_close, predicted_return=prev_return,
        directional_correct=(pred_dir != 0 and pred_dir == act_dir))
    return persistence, prev_row


def _delta(kronos: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if kronos is None or baseline is None:
        return None
    return kronos - baseline


def _winner_lower(kronos: Optional[float], baseline: Optional[float]) -> Optional[str]:
    """Winner for a metric where lower is better (errors)."""
    if kronos is None or baseline is None:
        return None
    if kronos < baseline:
        return 'kronos'
    if kronos > baseline:
        return 'baseline'
    return 'tie'


def _winner_higher(kronos: Optional[float], baseline: Optional[float]) -> Optional[str]:
    """Winner for a metric where higher is better (accuracy / correlation)."""
    if kronos is None or baseline is None:
        return None
    if kronos > baseline:
        return 'kronos'
    if kronos < baseline:
        return 'baseline'
    return 'tie'


def build_model_comparison(kronos: Dict[str, Any], persistence: Dict[str, Any],
                           previous_direction: Dict[str, Any]) -> Dict[str, Any]:
    """Build the ``model_comparison`` block of the evaluation report.

    Deltas are ``kronos - baseline``. Negative is better for error metrics
    (mae/rmse/mape/return_mae/return_rmse); positive is better for
    directional_accuracy and return_correlation. No statistical significance
    test is performed - winners are descriptive only.
    """
    def pair(name: str, base: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'mae_close_delta': _delta(kronos.get('mae_close'), base.get('mae_close')),
            'mae_close_winner': _winner_lower(kronos.get('mae_close'), base.get('mae_close')),
            'rmse_close_delta': _delta(kronos.get('rmse_close'), base.get('rmse_close')),
            'rmse_close_winner': _winner_lower(kronos.get('rmse_close'), base.get('rmse_close')),
            'mape_close_delta': _delta(kronos.get('mape_close'), base.get('mape_close')),
            'mape_close_winner': _winner_lower(kronos.get('mape_close'), base.get('mape_close')),
            'directional_accuracy_delta': _delta(
                kronos.get('directional_accuracy'), base.get('directional_accuracy')),
            'directional_accuracy_winner': _winner_higher(
                kronos.get('directional_accuracy'), base.get('directional_accuracy')),
            'return_mae_delta': _delta(kronos.get('return_mae'), base.get('return_mae')),
            'return_mae_winner': _winner_lower(kronos.get('return_mae'), base.get('return_mae')),
            'return_rmse_delta': _delta(kronos.get('return_rmse'), base.get('return_rmse')),
            'return_rmse_winner': _winner_lower(kronos.get('return_rmse'), base.get('return_rmse')),
            'return_correlation_delta': _delta(
                kronos.get('return_correlation'), base.get('return_correlation')),
            'return_correlation_winner': _winner_higher(
                kronos.get('return_correlation'), base.get('return_correlation')),
        }

    k_n = kronos.get('predictions')
    p_n = persistence.get('predictions')
    d_n = previous_direction.get('predictions')

    return {
        'note': ('deltas are (kronos - baseline); negative is better for error '
                 'metrics (mae/rmse/mape/return_mae/return_rmse), positive is '
                 'better for directional_accuracy and return_correlation. '
                 'No statistical significance test is performed - "winner" is '
                 'descriptive only.'),
        'prediction_count': {
            'kronos': k_n,
            'persistence': p_n,
            'previous_direction': d_n,
            'same_timestamps': (k_n is not None and k_n == p_n == d_n),
        },
        'kronos_vs_persistence': pair('kronos_vs_persistence', persistence),
        'kronos_vs_previous_direction': pair('kronos_vs_previous_direction', previous_direction),
    }
