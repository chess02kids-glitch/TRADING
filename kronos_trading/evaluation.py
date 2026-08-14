"""Phase 4 - strict chronological, no-lookahead prediction evaluator.

This module evaluates how accurately the *real* Kronos model predicts unseen
future candles using a strictly forward-in-time walk. It never shuffles, never
uses a future candle in model input, never normalises with future data, and
never fills gaps.

Guarantees enforced by construction:

* the model input at prediction time ``T`` contains only candles with open time
  strictly before ``T`` (the target candle and everything after are never
  supplied to the predictor);
* the newest candle in the dataset is treated as *currently forming* and is
  excluded from both inputs and targets;
* a gap, duplicate, NaN/inf, or invalid OHLC causes a *skip with a reason*,
  never a fabricated candle;
* predictions are generated strictly in chronological order and the actual
  target is read only after inference has completed.

The evaluator reuses the Phase 3 ``KronosRealPredictor`` (raw OHLCV in, real
model out) and the Phase 2 ``validate_context`` validation gate. It does not
compete with ``backtest.Backtester`` (a paper-P&L simulator): this module is
the model-accuracy evaluator.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .preprocess import TF, validate_context
from .types import Candle

# Default direction threshold: returns within +/-0.05% are treated as "flat"
# and are excluded from directional accuracy denominators.
DIRECTION_THRESHOLD_DEFAULT = 0.0005


def direction(value: float, threshold: float) -> int:
    """Classify a return as -1 / 0 / +1 using an explicit flatness threshold."""
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def parse_timestamp(value) -> Optional[int]:
    """Parse a CLI timestamp into UTC epoch milliseconds.

    Accepts an integer (epoch ms), an ISO-8601 string (``2026-08-01`` or
    ``2026-08-01T00:00:00``, with optional ``Z``), or ``None``.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip()
    if s.lstrip('-').isdigit():
        return int(s)
    s2 = s[:-1] + '+00:00' if s.endswith('Z') else s
    if 'T' not in s2 and ' ' not in s2:
        s2 = s2 + 'T00:00:00'
    dt = datetime.fromisoformat(s2)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


@dataclass
class EvaluationConfig:
    """Configuration for a reproducible chronological evaluation run."""
    context_length: int = 512
    horizon: int = 1
    # Deterministic Phase 3 inference recipe (top_k=1 + top_p=1.0 -> argmax).
    # When ``deterministic=False``, set top_k=0 / top_p=0.9 / sample_count=1
    # (the upstream stochastic sampling defaults) explicitly.
    deterministic: bool = True
    seed: int = 0
    temperature: float = 1.0
    top_k: int = 1
    top_p: float = 1.0
    sample_count: int = 1
    # Direction policy.
    direction_threshold: float = DIRECTION_THRESHOLD_DEFAULT
    # Window (epoch ms). None -> auto (recent ``max_predictions`` targets).
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    max_predictions: int = 1000
    # Evaluation boundary. None -> the newest candle is treated as forming.
    as_of_ms: Optional[int] = None

    def validate(self) -> None:
        if self.context_length < 1:
            raise ValueError('context_length must be >= 1')
        if self.horizon < 1:
            raise ValueError('horizon must be >= 1')
        if self.direction_threshold < 0:
            raise ValueError('direction_threshold must be >= 0')
        if self.max_predictions < 1:
            raise ValueError('max_predictions must be >= 1')

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationRow:
    """One prediction vs actual outcome, with the evaluation configuration.

    ``predicted_open/high/low/volume`` are ``Optional`` because naive baselines
    predict only close and return (those fields are ``None`` for baselines and
    excluded from the comparison metrics).
    """
    symbol: str
    timeframe: str
    context_end_timestamp: int
    prediction_timestamp: int
    actual_timestamp: int
    context_length: int
    predicted_open: Optional[float]
    predicted_high: Optional[float]
    predicted_low: Optional[float]
    predicted_close: float
    predicted_volume: Optional[float]
    actual_open: float
    actual_high: float
    actual_low: float
    actual_close: float
    actual_volume: float
    predicted_return: float
    actual_return: float
    absolute_close_error: float
    squared_close_error: float
    directional_correct: bool
    model_revision: Optional[str]
    tokenizer_revision: Optional[str]
    inference_latency_ms: float
    device: str
    deterministic: bool
    seed: int
    top_k: int
    top_p: float
    sample_count: int
    horizon: int
    direction_threshold: float

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)


def _mean(xs: List[float]) -> Optional[float]:
    return statistics.fmean(xs) if xs else None


def _valid_pairs(a, b):
    """Pairwise values where both sides are defined (skip None predictions)."""
    return [(x, y) for x, y in zip(a, b) if x is not None and y is not None]


def _mae(a: List[float], b: List[float]) -> Optional[float]:
    if not a:
        return None
    return statistics.fmean([abs(x - y) for x, y in zip(a, b)])


def _rmse(a: List[float], b: List[float]) -> Optional[float]:
    if not a:
        return None
    return math.sqrt(statistics.fmean([(x - y) ** 2 for x, y in zip(a, b)]))


def _mae_pairs(a, b) -> Optional[float]:
    """MAE over rows where both predicted and actual are defined."""
    pairs = _valid_pairs(a, b)
    if not pairs:
        return None
    return statistics.fmean([abs(x - y) for x, y in pairs])


def _rmse_pairs(a, b) -> Optional[float]:
    pairs = _valid_pairs(a, b)
    if not pairs:
        return None
    return math.sqrt(statistics.fmean([(x - y) ** 2 for x, y in pairs]))


def _pearson(a: List[float], b: List[float]) -> Optional[float]:
    """Pearson correlation, None when undefined (empty / zero variance)."""
    n = len(a)
    if n < 2 or len(b) != n:
        return None
    ma = statistics.fmean(a)
    mb = statistics.fmean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    return cov / math.sqrt(va * vb)


def compute_metrics(rows: List[EvaluationRow], direction_threshold: float) -> Dict[str, Any]:
    """Compute price / direction / return metrics from evaluation rows.

    Direction policy (explicit, not noise-driven):

    * ``actual_dir = direction(actual_return)``, ``pred_dir = direction(predicted_return)``
      using ``direction_threshold``.
    * ``directional_correct`` is True only when both are non-flat and equal.
    * ``directional_accuracy`` is the fraction of non-flat-actual candles whose
      direction was predicted correctly. Near-zero (flat) actual candles are
      excluded from all direction denominators.
    * ``bullish_accuracy`` / ``bearish_accuracy`` are conditioned on actual up /
      actual down respectively.

    Returns ``None`` (not fabricated zeros) for metrics that are undefined on
    the given rows (empty set, zero variance, all-flat directions).
    """
    n = len(rows)
    m: Dict[str, Any] = {'predictions': n}
    if n == 0:
        for key in ('mae_close', 'rmse_close', 'mae_open', 'mae_high', 'mae_low',
                    'mae_volume', 'mape_close', 'mape_close_valid_count',
                    'directional_accuracy', 'bullish_accuracy', 'bearish_accuracy',
                    'n_positive_actual', 'n_negative_actual', 'n_near_zero_actual',
                    'n_directional_correct', 'n_directional_incorrect',
                    'mean_predicted_return', 'mean_actual_return',
                    'return_mae', 'return_rmse', 'return_correlation'):
            m[key] = None
        return m

    pc = [r.predicted_close for r in rows]
    po = [r.predicted_open for r in rows]
    ph = [r.predicted_high for r in rows]
    pl = [r.predicted_low for r in rows]
    pv = [r.predicted_volume for r in rows]
    ac = [r.actual_close for r in rows]
    ao = [r.actual_open for r in rows]
    ah = [r.actual_high for r in rows]
    al = [r.actual_low for r in rows]
    av = [r.actual_volume for r in rows]

    m['mae_close'] = _mae_pairs(pc, ac)
    m['rmse_close'] = _rmse_pairs(pc, ac)
    m['mae_open'] = _mae_pairs(po, ao)
    m['mae_high'] = _mae_pairs(ph, ah)
    m['mae_low'] = _mae_pairs(pl, al)
    m['mae_volume'] = _mae_pairs(pv, av)

    # MAPE only where mathematically valid (|actual_close| > 0) and where the
    # prediction is defined.
    mape = [abs(p - a) / abs(a) for p, a in _valid_pairs(pc, ac) if abs(a) > 1e-12]
    m['mape_close'] = _mean(mape) if mape else None
    m['mape_close_valid_count'] = len(mape)

    pred_dir = [direction(r.predicted_return, direction_threshold) for r in rows]
    act_dir = [direction(r.actual_return, direction_threshold) for r in rows]

    m['n_positive_actual'] = sum(1 for d in act_dir if d > 0)
    m['n_negative_actual'] = sum(1 for d in act_dir if d < 0)
    m['n_near_zero_actual'] = sum(1 for d in act_dir if d == 0)

    nonflat = [i for i, d in enumerate(act_dir) if d != 0]
    if nonflat:
        correct = sum(1 for i in nonflat if pred_dir[i] == act_dir[i])
        m['directional_accuracy'] = correct / len(nonflat)
        m['n_directional_correct'] = correct
        m['n_directional_incorrect'] = len(nonflat) - correct
    else:
        m['directional_accuracy'] = None
        m['n_directional_correct'] = 0
        m['n_directional_incorrect'] = 0

    up = [i for i in nonflat if act_dir[i] > 0]
    down = [i for i in nonflat if act_dir[i] < 0]
    m['bullish_accuracy'] = (sum(1 for i in up if pred_dir[i] == 1) / len(up)) if up else None
    m['bearish_accuracy'] = (sum(1 for i in down if pred_dir[i] == -1) / len(down)) if down else None

    pr = [r.predicted_return for r in rows]
    ar = [r.actual_return for r in rows]
    m['mean_predicted_return'] = _mean(pr)
    m['mean_actual_return'] = _mean(ar)
    m['return_mae'] = _mae(pr, ar)
    m['return_rmse'] = _rmse(pr, ar)
    m['return_correlation'] = _pearson(pr, ar)
    return m


@dataclass
class EvaluationResult:
    """Report + rows + baseline rows + skip accounting for one series."""
    report: Dict[str, Any] = field(default_factory=dict)
    rows: List[EvaluationRow] = field(default_factory=list)
    baseline_rows: Dict[str, List[EvaluationRow]] = field(default_factory=dict)
    skip_reasons: Dict[str, int] = field(default_factory=dict)
    skipped: int = 0


class PredictionEvaluator:
    """Walk chronologically over closed candles and score real predictions."""

    def __init__(self, predictor, config: EvaluationConfig, symbol: str, timeframe: str):
        if timeframe not in TF:
            raise ValueError('unsupported timeframe: %r' % timeframe)
        config.validate()
        self.predictor = predictor
        self.config = config
        self.symbol = symbol
        self.timeframe = timeframe
        self.tf_ms = TF[timeframe]

    # ------------------------------------------------------------------ meta --
    def _meta(self) -> Dict[str, Any]:
        manager = getattr(self.predictor, 'manager', None)
        return {
            'model_name': getattr(manager, 'model_name', None),
            'model_revision': getattr(manager, 'resolved_model_revision', None),
            'tokenizer_revision': getattr(manager, 'resolved_tokenizer_revision', None),
            'device': getattr(self.predictor, 'device', 'unknown'),
            'dtype': getattr(self.predictor, 'dtype', None),
        }

    # -------------------------------------------------------------- history --
    def _closed(self, data: List[Candle]) -> List[Candle]:
        """Closed candles only. By default the newest candle is treated as the
        currently-forming candle and excluded."""
        if self.config.as_of_ms is not None:
            return [c for c in data if c.timestamp_ms + self.tf_ms <= self.config.as_of_ms]
        if not data:
            return []
        return data[:-1]

    @staticmethod
    def _bump(skips: Dict[str, int], reason: str) -> None:
        skips[reason] = skips.get(reason, 0) + 1

    # ------------------------------------------------------------ one target --
    def _evaluate_one(self, closed: List[Candle], i: int,
                      skips: Dict[str, int]):
        """Evaluate one target candle.

        Returns ``(kronos_row, persistence_row, previous_direction_row)`` or
        ``None`` when the step is skipped. The baseline rows are derived from
        the *same* validated context and target as the Kronos row, so they
        share identical prediction timestamps and never see the future.
        """
        cfg = self.config
        tf = self.tf_ms

        # Context: candles strictly before the target, validated (contiguous,
        # gap-free, finite, valid OHLC). The target candle is NOT included.
        ctx_slice = closed[i - cfg.context_length:i]
        try:
            ctx = validate_context(ctx_slice, self.timeframe, cfg.context_length)
        except ValueError:
            self._bump(skips, 'context_invalid')
            return None

        targets = closed[i:i + cfg.horizon]
        # Boundary between context end and the first target must be exactly one step.
        if targets[0].timestamp_ms != ctx[-1].timestamp_ms + tf:
            self._bump(skips, 'target_gap')
            return None
        # The target window must itself be contiguous.
        for a, b in zip(targets, targets[1:]):
            if b.timestamp_ms - a.timestamp_ms != tf:
                self._bump(skips, 'target_gap')
                return None
        # The target must be a valid, fully-materialised candle.
        try:
            self._validate_target(targets)
        except ValueError:
            self._bump(skips, 'invalid_target')
            return None

        # Real inference. Future candles (targets and beyond) are never passed
        # to the predictor - only ``ctx`` is supplied here.
        result = self.predictor.predict(
            ctx, self.timeframe, cfg.horizon,
            temperature=cfg.temperature, top_k=cfg.top_k, top_p=cfg.top_p,
            sample_count=cfg.sample_count, seed=cfg.seed,
            deterministic=cfg.deterministic,
        )
        if not result.steps or len(result.steps) < cfg.horizon:
            self._bump(skips, 'empty_prediction')
            return None

        # Only now (after inference) is the actual outcome read for comparison.
        actual = targets[-1]
        predicted = result.steps[-1]
        baseline = ctx[-1].close
        pred_ret = float(predicted['close']) / baseline - 1.0
        act_ret = actual.close / baseline - 1.0

        pred_dir = direction(pred_ret, cfg.direction_threshold)
        act_dir = direction(act_ret, cfg.direction_threshold)

        meta = self._meta()
        kronos_row = EvaluationRow(
            symbol=self.symbol,
            timeframe=self.timeframe,
            context_end_timestamp=ctx[-1].timestamp_ms,
            prediction_timestamp=targets[0].timestamp_ms,
            actual_timestamp=actual.timestamp_ms,
            context_length=len(ctx),
            predicted_open=float(predicted['open']),
            predicted_high=float(predicted['high']),
            predicted_low=float(predicted['low']),
            predicted_close=float(predicted['close']),
            predicted_volume=float(predicted['volume']),
            actual_open=actual.open,
            actual_high=actual.high,
            actual_low=actual.low,
            actual_close=actual.close,
            actual_volume=actual.volume,
            predicted_return=pred_ret,
            actual_return=act_ret,
            absolute_close_error=abs(float(predicted['close']) - actual.close),
            squared_close_error=(float(predicted['close']) - actual.close) ** 2,
            directional_correct=(pred_dir != 0 and pred_dir == act_dir),
            model_revision=meta['model_revision'],
            tokenizer_revision=meta['tokenizer_revision'],
            inference_latency_ms=result.latency_ms,
            device=meta['device'],
            deterministic=cfg.deterministic,
            seed=cfg.seed,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            sample_count=cfg.sample_count,
            horizon=cfg.horizon,
            direction_threshold=cfg.direction_threshold,
        )

        # Naive baselines on the SAME timestamps (derived from ctx, which ends
        # strictly before the target - no future access).
        from .baselines import baseline_rows_for  # local import avoids a cycle
        persistence_row, previous_direction_row = baseline_rows_for(
            kronos_row, ctx, cfg.direction_threshold)
        return kronos_row, persistence_row, previous_direction_row

    @staticmethod
    def _validate_target(targets: List[Candle]) -> None:
        for c in targets:
            vals = (c.open, c.high, c.low, c.close, c.volume)
            if not all(math.isfinite(v) for v in vals):
                raise ValueError('non-finite target candle')
            if min(c.open, c.high, c.low, c.close) <= 0:
                raise ValueError('non-positive target OHLC')
            if c.volume < 0:
                raise ValueError('negative target volume')
            if not (c.high >= c.low and c.high >= c.open and c.high >= c.close
                    and c.low <= c.open and c.low <= c.close):
                raise ValueError('invalid target OHLC relationship')

    # ------------------------------------------------------------- evaluate --
    def evaluate(self, candles: List[Candle]) -> EvaluationResult:
        cfg = self.config
        tf = self.tf_ms

        # Sort defensively (no fabrication); duplicates are detected downstream.
        data = sorted(candles, key=lambda c: c.timestamp_ms)
        closed = self._closed(data)
        n = len(closed)

        first_valid = cfg.context_length          # context_length candles before target
        last_valid = n - cfg.horizon              # last index whose target window exists
        rows: List[EvaluationRow] = []
        persistence_rows: List[EvaluationRow] = []
        previous_direction_rows: List[EvaluationRow] = []
        skips: Dict[str, int] = {}

        if n == 0 or last_valid < first_valid:
            return self._result(rows, persistence_rows, previous_direction_rows,
                                skips, closed, None, None)

        # Choose the last target index (end of the holdout window).
        if cfg.end_ms is not None:
            last_target = last_valid
            while last_target >= first_valid:
                if closed[last_target + cfg.horizon - 1].timestamp_ms + tf <= cfg.end_ms:
                    break
                last_target -= 1
        else:
            last_target = last_valid

        # Choose the first target index (start of the holdout window).
        if cfg.start_ms is not None or cfg.end_ms is not None:
            # Explicit window: honour --start/--end, no max_predictions cap.
            first_target = first_valid
            if cfg.start_ms is not None:
                while first_target <= last_target and closed[first_target].timestamp_ms < cfg.start_ms:
                    first_target += 1
        else:
            # Default: the most recent ``max_predictions`` valid targets.
            first_target = max(first_valid, last_target - cfg.max_predictions + 1)

        if first_target > last_target:
            return self._result(rows, persistence_rows, previous_direction_rows,
                                skips, closed, None, None)

        for i in range(first_target, last_target + 1):
            outcome = self._evaluate_one(closed, i, skips)
            if outcome is not None:
                kronos_row, persistence_row, previous_direction_row = outcome
                rows.append(kronos_row)
                persistence_rows.append(persistence_row)
                if previous_direction_row is not None:
                    previous_direction_rows.append(previous_direction_row)

        return self._result(rows, persistence_rows, previous_direction_rows,
                            skips, closed, first_target, last_target)

    def _result(self, rows: List[EvaluationRow],
                persistence_rows: List[EvaluationRow],
                previous_direction_rows: List[EvaluationRow],
                skips: Dict[str, int],
                closed: List[Candle],
                first_target: Optional[int], last_target: Optional[int]) -> EvaluationResult:
        cfg = self.config
        meta = self._meta()

        warmup_start = closed[0].timestamp_ms if closed else None
        warmup_end = closed[first_target - 1].timestamp_ms if first_target and first_target >= 1 else None
        eval_start = closed[first_target].timestamp_ms if first_target is not None else None
        eval_end = (closed[last_target + cfg.horizon - 1].timestamp_ms + self.tf_ms
                    if last_target is not None else None)

        kronos_metrics = compute_metrics(rows, cfg.direction_threshold)
        persistence_metrics = compute_metrics(persistence_rows, cfg.direction_threshold)
        previous_direction_metrics = compute_metrics(previous_direction_rows,
                                                     cfg.direction_threshold)

        from .baselines import build_model_comparison  # local import avoids a cycle
        model_comparison = build_model_comparison(
            kronos_metrics, persistence_metrics, previous_direction_metrics)

        report = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            **meta,
            'context_length': cfg.context_length,
            'horizon': cfg.horizon,
            'deterministic': cfg.deterministic,
            'seed': cfg.seed,
            'temperature': cfg.temperature,
            'top_k': cfg.top_k,
            'top_p': cfg.top_p,
            'sample_count': cfg.sample_count,
            'direction_threshold': cfg.direction_threshold,
            'as_of_ms': cfg.as_of_ms,
            'warmup_start_ms': warmup_start,
            'warmup_end_ms': warmup_end,
            'evaluation_start_ms': eval_start,
            'evaluation_end_ms': eval_end,
            'predictions': len(rows),
            'skipped': sum(skips.values()),
            'skip_reasons': dict(skips),
            'metrics': kronos_metrics,
            'baseline_results': {
                'persistence': persistence_metrics,
                'previous_direction': previous_direction_metrics,
            },
            'model_comparison': model_comparison,
        }
        baseline_rows = {
            'persistence': persistence_rows,
            'previous_direction': previous_direction_rows,
        }
        return EvaluationResult(report=report, rows=rows,
                                baseline_rows=baseline_rows,
                                skip_reasons=dict(skips),
                                skipped=sum(skips.values()))


def run_evaluation(predictor, config: EvaluationConfig, symbol: str, timeframe: str,
                   candles: List[Candle]) -> EvaluationResult:
    """Convenience wrapper: build an evaluator and run it once."""
    return PredictionEvaluator(predictor, config, symbol, timeframe).evaluate(candles)
