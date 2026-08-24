"""Data layer for the Fear & Greed Contrarian strategy.

- Fear & Greed index:  https://api.alternative.me/fng/   (free, no auth)
- BTC dominance:       https://api.coingecko.com/api/v3/ (free, no auth; HISTORY
                       for 730d is NOT available on the free tier - only the
                       current snapshot from /global. If a full daily series
                       cannot be built we return None and the strategy runs
                       WITHOUT the dominance filter, per spec.)
- Prices:              read-only 1h CSVs in sandbox/pattern_research/cache/

No data is ever fabricated: every fallback either uses a cached copy of a real
API response or fails loudly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
CACHE_DIR = PACKAGE_DIR / "cache"
PRICE_CACHE = REPO_ROOT / "sandbox" / "pattern_research" / "cache"

FNG_URL = "https://api.alternative.me/fng/"
COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"
COINGECKO_BTC_MCAP = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

FNG_CACHE = CACHE_DIR / "fng.csv"
DOMINANCE_CACHE = CACHE_DIR / "dominance.csv"


# --------------------------------------------------------------------------- #
# Fear & Greed
# --------------------------------------------------------------------------- #
def fetch_fear_greed(limit: int = 500, timeout: int = 15) -> pd.DataFrame:
    """Return DataFrame(timestamp, value 0-100, classification), newest last.

    Tries the live API first and caches the response. If the API is unreachable
    it falls back to the cached copy. Raises if neither is available.
    """
    import requests

    try:
        resp = requests.get(FNG_URL, params={"limit": limit}, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        data = payload["data"]
        df = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp(int(row["timestamp"]), unit="s", tz="UTC").strftime("%Y-%m-%d")
                    for row in data
                ],
                "value": [int(row["value"]) for row in data],
                "classification": [row["value_classification"] for row in data],
            }
        )
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(FNG_CACHE, index=False)
        print(f"[fng] fetched {len(df)} days from API (cached to {FNG_CACHE})")
        return df
    except Exception as exc:  # network blocked, timeout, bad payload...
        print(f"[fng] API unavailable ({type(exc).__name__}: {exc}); trying cache {FNG_CACHE}")
        if FNG_CACHE.exists():
            df = pd.read_csv(FNG_CACHE)
            print(f"[fng] using cached copy: {len(df)} days "
                  f"({df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]})")
            return df
        raise RuntimeError(
            "Fear & Greed API failed AND no cached copy exists. "
            "Refusing to fabricate data - aborting."
        ) from exc


# --------------------------------------------------------------------------- #
# BTC dominance
# --------------------------------------------------------------------------- #
def fetch_btc_dominance_history(days: int = 730, timeout: int = 15):
    """Try to build a daily BTC-dominance history (timestamp, dominance_pct).

    Strategy:
      1. /global/market_cap_chart needs a demo/paid key -> only attempt as
         opportunistic query; skip on failure.
      2. Otherwise we would need total-market-cap history (paid) to divide the
         BTC market-cap history by. Without it a *historical* dominance series
         cannot be computed from free endpoints.
    Returns None when no real history is obtainable (caller then runs the
    strategy WITHOUT the dominance filter and notes it in the report). A real
    cached dominance.csv is used if present.
    """
    import requests

    if DOMINANCE_CACHE.exists():
        df = pd.read_csv(DOMINANCE_CACHE, parse_dates=["timestamp"])
        print(f"[dominance] using cached history: {len(df)} days")
        return df

    try:
        g = requests.get(COINGECKO_GLOBAL, timeout=timeout)
        g.raise_for_status()
        now_pct = g.json()["data"]["market_cap_percentage"]["btc"]
        print(f"[dominance] /global reachable - current BTC dominance = {now_pct:.2f}% "
              "(point-in-time only)")
    except Exception as exc:
        print(f"[dominance] API unavailable ({type(exc).__name__}: {exc})")

    try:
        r = requests.get(
            COINGECKO_BTC_MCAP,
            params={"vs_currency": "usd", "days": days},
            timeout=timeout,
        )
        r.raise_for_status()
        btc_mcap = pd.DataFrame(
            r.json()["market_caps"], columns=["ms", "btc_mcap"]
        )
        # total market-cap *history* endpoint (demo key required)
        t = requests.get(
            "https://api.coingecko.com/api/v3/global/market_cap_chart",
            params={"days": days},
            timeout=timeout,
        )
        t.raise_for_status()
        total = pd.DataFrame(t.json()["market_cap"], columns=["ms", "total_mcap"])
        df = btc_mcap.merge(total, on="ms")
        df["timestamp"] = pd.to_datetime(df["ms"], unit="ms", utc=True)
        df["dominance_pct"] = df["btc_mcap"] / df["total_mcap"] * 100.0
        df = df[["timestamp", "dominance_pct"]].set_index("timestamp").resample("1D").last().dropna().reset_index()
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DOMINANCE_CACHE, index=False)
        print(f"[dominance] built {len(df)}-day history and cached it")
        return df
    except Exception as exc:
        print(f"[dominance] no free historical dominance available "
              f"({type(exc).__name__}: {exc}) -> running WITHOUT dominance filter")
        return None


# --------------------------------------------------------------------------- #
# Prices (READ ONLY)
# --------------------------------------------------------------------------- #
def load_price_data(asset: str = "BTC") -> pd.DataFrame:
    """Load the 1h price CSV from sandbox/pattern_research/cache (read-only)."""
    asset = asset.upper()
    path = PRICE_CACHE / f"{asset}USDT_1h_730d.csv"
    if not path.exists():
        raise FileNotFoundError(f"price cache not found: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    return df[["open", "high", "low", "close", "volume"]]


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #
def merge_all_data(
    prices: pd.DataFrame,
    fng: pd.DataFrame,
    dominance: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Align daily F&G (and dominance) onto the 1h price bars.

    No-lookahead handling: the F&G value for calendar day D is published at the
    END of day D, so it is only made available to bars from day D+1 00:00
    onwards.  The merged frame carries `fng_value` already lagged this way;
    signal_generator additionally applies the spec's .shift(1) on the signal.
    """
    df = prices.copy()

    fng = fng.copy()
    fng["avail_from"] = (
        pd.to_datetime(fng["timestamp"]) + pd.Timedelta(days=1)
    ).dt.tz_localize(df.index.tz)  # value of day D usable from D+1 00:00
    fng = fng.set_index("avail_from").sort_index()
    df["fng_value"] = fng["value"].reindex(df.index, method="ffill")
    df["fng_classification"] = fng["classification"].reindex(df.index, method="ffill")

    if dominance is not None and len(dominance):
        dom = dominance.copy()
        dom["avail_from"] = (
            pd.to_datetime(dom["timestamp"]) + pd.Timedelta(days=1)
        ).dt.tz_localize(df.index.tz)
        dom = dom.set_index("avail_from").sort_index()
        df["dominance_pct"] = dom["dominance_pct"].reindex(df.index, method="ffill")
    else:
        df["dominance_pct"] = float("nan")

    # leading NaNs (bars before first F&G availability) are dropped
    df = df.dropna(subset=["fng_value"])
    return df
