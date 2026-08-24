"""Convert the raw Fear & Greed API response (cache/fng_raw.txt) into cache/fng.csv.

Provenance: cache/fng_raw.txt contains the verbatim (unix_timestamp, value) pairs
returned by https://api.alternative.me/fng/?limit=1000 (fetched 2026-08-24, UTC).
This script only reshapes that data - it never invents or interpolates values.
The source API itself is missing one day (2024-10-26); that gap is preserved.

Usage:  python -m fear_greed_strategy.build_cache
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "cache" / "fng_raw.txt"
OUT = HERE / "cache" / "fng.csv"


def classify(v: int) -> str:
    """Classification bands exactly as emitted by alternative.me (verified against
    every label in the API response)."""
    if v <= 25:
        return "Extreme Fear"
    if v <= 46:
        return "Fear"
    if v <= 54:
        return "Neutral"
    if v <= 75:
        return "Greed"
    return "Extreme Greed"


def main() -> None:
    rows = []
    for line in RAW.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ts, val = line.split(",")
        rows.append((int(ts), int(val)))

    # integrity checks (fail loudly rather than ship bad data)
    assert all(0 <= v <= 100 for _, v in rows), "value out of range"
    for (t0, _), (t1, _) in zip(rows, rows[1:]):
        assert t0 - t1 in (86400, 172800), f"bad step {t0} -> {t1}"
    assert len(rows) == len({t for t, _ in rows}), "duplicate timestamps"

    df = pd.DataFrame(rows, columns=["ts", "value"])
    df["timestamp"] = df["ts"].map(lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d"))
    df["classification"] = df["value"].map(classify)
    df[["timestamp", "value", "classification"]].to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} rows, {df['timestamp'].iloc[-1]} -> {df['timestamp'].iloc[0]}")


if __name__ == "__main__":
    main()
