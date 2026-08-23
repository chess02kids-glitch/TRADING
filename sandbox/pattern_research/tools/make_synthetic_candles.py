"""Generate a deterministic synthetic OHLCV CSV (sandbox smoke-testing only).

This exists so the whole pipeline can be exercised on machines with **no
exchange egress**. The data is a seeded geometric random walk — it contains no
real market information and must never be presented as a research result.

    python -m sandbox.pattern_research.tools.make_synthetic_candles \\
        --out /tmp/BTCUSDT_synth.csv --bars 17520 --seed 1 --freq 1h
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def make_synthetic_candles(n_bars: int = 17520, seed: int = 1,
                           start: str = "2024-01-01T00:00:00Z",
                           start_price: float = 30000.0,
                           vol: float = 0.004,
                           freq: str = "1h") -> pd.DataFrame:
    """Seeded random-walk candles with the canonical sandbox schema.

    ``freq`` sets the bar spacing of the timestamp index (``"1h"`` default,
    ``"4h"`` / ``"1D"`` for the other supported timeframes) so the 4h and 1d
    code paths can be smoke-tested offline too.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, vol, size=n_bars)
    close = float(start_price) * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[start_price], close[:-1]])
    # independent upper/lower wicks so asymmetric shapes (hammers etc.) occur
    upper_wick = np.abs(rng.normal(0.0, vol, size=n_bars)) * close
    lower_wick = np.abs(rng.normal(0.0, vol, size=n_bars)) * close
    high = np.maximum(open_, close) + upper_wick
    low = np.minimum(open_, close) - lower_wick
    volume = np.abs(rng.lognormal(mean=3.0, sigma=0.6, size=n_bars))
    index = pd.date_range(start=start, periods=n_bars, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}, index=index)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--bars", type=int, default=17520)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--freq", default="1h",
                   help="Bar spacing: 1h (default), 4h or 1D.")
    args = p.parse_args(argv)
    df = make_synthetic_candles(args.bars, args.seed, freq=args.freq)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index_label="timestamp")
    print(f"Wrote {len(df)} synthetic {args.freq} candles to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
