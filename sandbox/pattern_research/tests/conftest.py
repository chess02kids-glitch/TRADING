"""Shared fixtures/helpers for the pattern-research sandbox tests."""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

# Make the repository root importable so `sandbox.pattern_research...` resolves
# no matter how pytest is invoked.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def make_candles(rows, start="2024-01-01T00:00:00Z"):
    """Build a canonical candles frame from ``(o, h, l, c, v)`` tuples."""
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"],
                      dtype=float)
    df.index = pd.date_range(start=start, periods=len(df), freq="1h",
                             tz="UTC", name="timestamp")
    return df


@pytest.fixture
def synthetic_candles():
    from sandbox.pattern_research.tools.make_synthetic_candles import make_synthetic_candles
    return make_synthetic_candles(n_bars=3000, seed=7)
