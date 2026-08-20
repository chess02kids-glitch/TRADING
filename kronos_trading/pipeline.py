"""Prediction pipeline: closed-candle validation -> predictor -> Prediction."""
import time
from typing import List, Optional

from .model import KronosRealPredictor
from .preprocess import closed, validate_context, normalize_ohlcv, TF
from .types import Candle, Prediction


class PredictionPipeline:
    """Runs a predictor over a validated closed-candle context.

    The real Kronos path and the offline mock path share the same validation
    gate (``validate_context``) but use different input representations, so the
    real model never sees the mock path's relative units and the mock path is
    never selected for real inference.
    """

    def __init__(self, predictor, model_version=None):
        self.predictor = predictor
        self.model_version = model_version or getattr(predictor, 'version', 'Kronos')

    def predict(self,
                symbol: str,
                timeframe: str,
                candles: List[Candle],
                context_length: int,
                horizon: int,
                now_ms: int,
                seed: Optional[int] = None,
                deterministic: bool = False) -> Prediction:
        if timeframe not in TF:
            raise ValueError('unsupported timeframe: %r' % timeframe)
        if horizon < 1:
            raise ValueError('horizon must be >= 1')
        ctx = validate_context(closed(candles, timeframe, now_ms), timeframe,
                               context_length)

        if isinstance(self.predictor, KronosRealPredictor):
            return self._predict_real(symbol, timeframe, ctx, horizon, seed,
                                      deterministic)
        return self._predict_mock(symbol, timeframe, ctx, horizon)

    # ------------------------------------------------------------------ real --
    def _predict_real(self, symbol, timeframe, ctx, horizon, seed, deterministic):
        result = self.predictor.predict(
            ctx, timeframe, horizon, seed=seed, deterministic=deterministic)
        steps = result.steps
        if not steps:
            raise RuntimeError('Kronos returned no prediction steps')

        last_close = ctx[-1].close
        final_close = steps[-1]['close']
        expected_return = (final_close - last_close) / last_close

        pred_start_ms = ctx[-1].timestamp_ms + TF[timeframe]
        pred_timestamps = [pred_start_ms + TF[timeframe] * i for i in range(horizon)]

        first = steps[0]
        manager = getattr(self.predictor, 'manager', None)
        return Prediction(
            symbol=symbol,
            timeframe=timeframe,
            prediction_timestamp_ms=pred_timestamps[0],
            horizon=horizon,
            predicted_close=final_close,
            expected_return=expected_return,
            input_start_ms=ctx[0].timestamp_ms,
            input_end_ms=ctx[-1].timestamp_ms,
            generated_at_ms=int(time.time() * 1000),
            model_version=self.model_version,
            device=getattr(self.predictor, 'device', 'unknown'),
            inference_ms=result.latency_ms,
            confidence=None,
            context_length=len(ctx),
            prediction_timestamps_ms=pred_timestamps,
            predicted_open=first.get('open'),
            predicted_high=first.get('high'),
            predicted_low=first.get('low'),
            predicted_volume=first.get('volume'),
            predicted_amount=first.get('amount'),
            predicted_ohlcv=steps,
            model_name=getattr(manager, 'model_name', None),
            model_revision=getattr(manager, 'resolved_model_revision', None),
            tokenizer_revision=getattr(manager, 'resolved_tokenizer_revision', None),
            dtype=getattr(self.predictor, 'dtype', None),
            peak_vram_bytes=result.peak_vram_bytes,
        )

    # ----------------------------------------------------------------- mock --
    def _predict_mock(self, symbol, timeframe, ctx, horizon):
        if horizon != 1:
            raise ValueError('only one-candle horizon is supported by the mock adapter')
        value, elapsed = self.predictor.predict_close(normalize_ohlcv(ctx))
        # predictor works in relative units normalized to first close
        predicted = value * ctx[0].close
        last = ctx[-1].close
        return Prediction(
            symbol=symbol,
            timeframe=timeframe,
            prediction_timestamp_ms=ctx[-1].timestamp_ms + TF[timeframe],
            horizon=horizon,
            predicted_close=predicted,
            expected_return=(predicted - last) / last,
            input_start_ms=ctx[0].timestamp_ms,
            input_end_ms=ctx[-1].timestamp_ms,
            generated_at_ms=int(time.time() * 1000),
            model_version=self.model_version,
            device=getattr(self.predictor, 'device', 'unknown'),
            inference_ms=elapsed,
        )
