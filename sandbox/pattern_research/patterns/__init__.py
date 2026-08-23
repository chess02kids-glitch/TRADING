"""Pattern detectors for the research sandbox.

All detectors share one contract:

* input: canonical candles ``pd.DataFrame`` (UTC DatetimeIndex; columns
  ``open, high, low, close, volume``),
* output: an int ``pd.Series`` aligned to the input index, values in
  ``{-1, 0, +1}``,
* timing: the pattern completes at bar ``t-1`` and the series is
  ``.shift(1)``-ed, so ``signal[t]`` is knowable before bar ``t`` opens and the
  trade is measured from ``close[t]`` — no look-ahead anywhere.
"""
from . import candlestick, momentum, time_of_day, volume_spike  # noqa: F401
from .momentum import compute_forward_return  # noqa: F401

__all__ = [
    "momentum",
    "candlestick",
    "time_of_day",
    "volume_spike",
    "compute_forward_return",
]
