"""
Data preparation for the algo-research-lab (Gen 1 v2 reset).

Builds all research datasets from the cached raw files and writes
normalised, gap-checked, hourly CSVs + a manifest documenting
provenance. Run once:

    python data/prepare_data.py

Raw sources (see MANIFEST for full provenance):

  Window A (2017-08-17 .. 2019-11-04) - Binance spot 1h OHLCV
      BTC/USDT, ETH/USDT
  Window B (2020-01-01 .. 2024-01-01) - Bitstamp BTC/USD 1m
      resampled to 1h (execution price series), plus Binance
      USDT-M BTC-USDT perpetual funding rates (8h events).

No-lookahead rules applied here:
  * hourly bars are indexed by bar OPEN time (UTC, tz-aware);
  * funding events are stamped at their settlement time and are
    merged onto bars with open_time >= funding_time, then the
    merged column is shifted by one bar before ANY signal uses it
    (enforced in strategy_genome.generator, not here);
  * the still-forming bar is dropped everywhere.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
BITSTAMP_1M = "/tmp/bitstamp-btcusd-minute-data-main/data/historical/btcusd_bitstamp_1min_2012-2025.csv.gz"
MANIFEST_PATH = os.path.join(CACHE, "MANIFEST.json")

PROVENANCE = {
    "btc_usdt_1h_2017_2019.csv": {
        "source": "Binance spot klines 1h (BTCUSDT), vendored CSV",
        "retrieved_via": "https://codeload.github.com/cryptobigbro/binance-BTCUSDT (master)",
        "original_columns": "open_timestamp_utc,close_timestamp_utc,open,high,low,close,volume",
    },
    "eth_usdt_1h_2017_2019.csv": {
        "source": "Binance spot klines 1h (ETHUSDT), vendored CSV",
        "retrieved_via": "https://codeload.github.com/cryptobigbro/binance-ETHUSDT (master)",
    },
    "funding_rates_BTC.csv": {
        "source": "Binance USDT-M perpetual BTCUSDT funding rate history, 8h settlements",
        "retrieved_via": "https://codeload.github.com/supervik/historical-funding-rates-fetcher (main)",
        "columns": "Symbol,Date,Funding Rate (decimal fraction, e.g. -0.00012359 = -0.012359%)",
    },
    "funding_rates_ETH.csv": {
        "source": "Binance USDT-M perpetual ETHUSDT funding rate history, 8h settlements",
        "retrieved_via": "https://codeload.github.com/supervik/historical-funding-rates-fetcher (main)",
    },
    "btc_usd_1h_2020_2023.csv": {
        "source": "Bitstamp BTC/USD 1-minute OHLCV, resampled to 1h",
        "retrieved_via": "https://codeload.github.com/ff137/bitstamp-btcusd-minute-data (main)",
        "note": "raw 1m csv.gz (95MB) kept outside the repo; only the 1h resample is cached",
    },
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_binance_1h(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # open_timestamp_utc is seconds since epoch, bar open time.
    # Build from raw values: aligning Series against a timestamp index
    # here would silently produce an all-NaN frame.
    idx = pd.to_datetime(df["open_timestamp_utc"], unit="s", utc=True)
    out = pd.DataFrame(
        {
            "open": df["open"].astype(float).values,
            "high": df["high"].astype(float).values,
            "low": df["low"].astype(float).values,
            "close": df["close"].astype(float).values,
            "volume": df["volume"].astype(float).values,
        },
        index=idx,
    )
    assert out["close"].notna().sum() > 0, f"all-NaN OHLCV loaded from {path}"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def reindex_hourly(df: pd.DataFrame, name: str, stats: dict) -> pd.DataFrame:
    """Reindex to a continuous hourly grid; forward-fill rare gaps."""
    full = pd.date_range(df.index[0], df.index[-1], freq="1h", tz="UTC")
    missing = full.difference(df.index)
    stats[name] = {
        "rows_raw": int(len(df)),
        "grid_rows": int(len(full)),
        "missing_hour_bars_filled": int(len(missing)),
        "start": str(full[0]),
        "end": str(full[-1]),
    }
    out = df.reindex(full)
    gap_mask = out["close"].isna()
    # fill gaps: carry last close, O=H=L=C of the fill, volume 0
    out["close"] = out["close"].ffill()
    out["open"] = np.where(gap_mask, out["close"], out["open"])
    out["high"] = np.where(gap_mask, out["close"], out["high"])
    out["low"] = np.where(gap_mask, out["close"], out["low"])
    out["volume"] = out["volume"].fillna(0.0)
    out["filled_gap"] = gap_mask
    return out


def resample_bitstamp() -> pd.DataFrame:
    print("Resampling Bitstamp 1m -> 1h (this reads ~7M rows) ...")
    df = pd.read_csv(BITSTAMP_1M, compression="gzip")
    idx = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df.index = idx
    agg = df.resample("1h").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    # keep only hours with at least one 1m bar
    agg = agg.dropna(subset=["close"])
    return agg


def load_funding(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    idx = pd.to_datetime(df["Date"], utc=True)
    # build from raw values (NOT aligned against idx) - duplicate timestamps
    # in the index would otherwise produce NaNs during alignment
    out = pd.DataFrame(
        {"funding_rate": df["Funding Rate"].astype(float).values}, index=idx
    )
    out = out[~out.index.duplicated(keep="last")].sort_index()
    assert out["funding_rate"].notna().all(), "NaN funding rates after load"
    return out


def main() -> None:
    stats: dict = {"windows": {}}

    # ---- Window A : Binance spot 1h 2017-2019 ------------------------------
    btc_a = load_binance_1h(os.path.join(CACHE, "btc_usdt_1h_2017_2019.csv"))
    eth_a = load_binance_1h(os.path.join(CACHE, "eth_usdt_1h_2017_2019.csv"))
    btc_a = reindex_hourly(btc_a, "btc_usdt_1h_A", stats["windows"])
    eth_a = reindex_hourly(eth_a, "eth_usdt_1h_A", stats["windows"])

    # intersect BTC and ETH onto the common grid (both Binance spot)
    common = btc_a.index.intersection(eth_a.index)
    btc_a = btc_a.loc[common]
    eth_a = eth_a.loc[common]
    stats["windows"]["window_A"] = {
        "rows": int(len(common)),
        "start": str(common[0]),
        "end": str(common[-1]),
        "venue": "Binance spot",
        "note": "BTC/USDT and ETH/USDT on identical hourly grid",
    }
    btc_a.to_parquet(os.path.join(CACHE, "btc_1h_A.parquet"))
    eth_a.to_parquet(os.path.join(CACHE, "eth_1h_A.parquet"))

    # ---- Bitstamp 1h (Window B price series) ------------------------------
    bs_path = os.path.join(CACHE, "btc_usd_1h_2020_2023.csv")
    if not os.path.exists(os.path.join(CACHE, "btc_usd_1h_2020_2023.parquet")):
        if not os.path.exists(bs_path):
            agg = resample_bitstamp()
            agg.to_csv(bs_path)
        else:
            agg = pd.read_csv(bs_path, index_col=0, parse_dates=True)
            agg.index = pd.to_datetime(agg.index, utc=True)
        # Window B grid: lead-in from 2019-12-01 so indicators warm up,
        # funding coverage 2020-01-01 .. 2024-01-01
        m = (agg.index >= "2019-12-01") & (agg.index < "2024-01-01")
        btc_b = agg.loc[m]
        btc_b = btc_b[~btc_b.index.duplicated(keep="last")].sort_index()
        btc_b = reindex_hourly(btc_b, "btc_usd_1h_B", stats["windows"])
        btc_b.to_parquet(os.path.join(CACHE, "btc_1h_B.parquet"))

    # ---- Funding ----------------------------------------------------------
    fund = load_funding(os.path.join(CACHE, "funding_rates_BTC.csv"))
    stats["windows"]["funding_BTC"] = {
        "rows": int(len(fund)),
        "start": str(fund.index[0]),
        "end": str(fund.index[-1]),
        "settlement_hours_utc": sorted({t.hour for t in fund.index}),
        "p5_pct": float(np.percentile(fund["funding_rate"], 5) * 100),
        "median_pct": float(np.percentile(fund["funding_rate"], 50) * 100),
        "p95_pct": float(np.percentile(fund["funding_rate"], 95) * 100),
    }
    fund.to_parquet(os.path.join(CACHE, "funding_BTC.parquet"))

    # ---- Manifest ----------------------------------------------------------
    manifest = {
        "provenance": PROVENANCE,
        "stats": stats,
        "sha16": {
            f: sha256(os.path.join(CACHE, f))
            for f in sorted(os.listdir(CACHE))
            if f.endswith(".csv")
        },
        "generated_by": "algo-research-lab/data/prepare_data.py",
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(stats, indent=2))
    print("\nManifest written:", MANIFEST_PATH)


if __name__ == "__main__":
    sys.exit(main())
