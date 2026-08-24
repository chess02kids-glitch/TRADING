"""Signal generation for the Fear & Greed Contrarian strategy.

Entry (ALL must hold, evaluated on information available at the bar):
    1. fng_value < fear_threshold          (extreme fear)
    2. close > close.shift(1)              (momentum: not still falling)
    3. dominance filter, when dominance data exists:
         dominance rising  -> prefer BTC long
         dominance falling -> prefer ETH long
       (no dominance data -> filter skipped)

Exit (handled statefully in the backtester):
    - fng_value > greed_threshold          (extreme greed)
    - held longer than max_hold_days
    - stop loss hit

Position sizing (scaled so the spec's 10/15/20/25 bands are the exact
fear_threshold=25 case):
    fng < 0.40*T -> 1.00   (T=25: <10)
    fng < 0.60*T -> 0.85   (T=25: <15)
    fng < 0.80*T -> 0.70   (T=25: <20)
    fng < T      -> 0.50   (T=25: <25)
    else         -> 0.0
"""
from __future__ import annotations

import pandas as pd


def position_size_for_fear(fng: float, fear_threshold: float) -> float:
    """More fearful -> bigger position. Bands scale with the fear threshold."""
    t = float(fear_threshold)
    if fng < 0.40 * t:
        return 1.00
    if fng < 0.60 * t:
        return 0.85
    if fng < 0.80 * t:
        return 0.70
    if fng < t:
        return 0.50
    return 0.0


def generate_signals(
    df: pd.DataFrame,
    asset: str = "BTC",
    fear_threshold: float = 25,
    greed_threshold: float = 75,
    use_dominance: bool = True,
) -> pd.DataFrame:
    """Return df with signal columns.

    Columns added:
        fng_signal       +1 buy zone / -1 sell zone / 0 flat
        dominance_rising bool (NaN-safe: False when no dominance data)
        final_signal     +1 where ALL entry conditions hold
        position_size    0..1 sized off the fear level
        force_exit       True where fng > greed_threshold
        position         final_signal.shift(1)   (spec: no lookahead)
        size             position_size.shift(1)  (spec: no lookahead)
    """
    out = df.copy()
    fng = out["fng_value"]

    out["fng_signal"] = 0
    out.loc[fng < fear_threshold, "fng_signal"] = 1
    out.loc[fng > greed_threshold, "fng_signal"] = -1

    has_dom = use_dominance and out["dominance_pct"].notna().any()
    if has_dom:
        # 24h change of the (daily, forward-filled) dominance series
        out["dominance_rising"] = out["dominance_pct"] > out["dominance_pct"].shift(24)
    else:
        out["dominance_rising"] = False

    momentum_ok = out["close"] > out["close"].shift(1)

    dom_ok = pd.Series(True, index=out.index)
    if has_dom:
        if asset.upper() == "ETH":
            dom_ok = ~out["dominance_rising"]   # falling dominance -> alts (ETH)
        else:
            dom_ok = out["dominance_rising"]    # rising dominance -> BTC

    out["final_signal"] = (
        (out["fng_signal"] == 1) & momentum_ok & dom_ok
    ).astype(int)

    out["position_size"] = fng.map(lambda v: position_size_for_fear(v, fear_threshold))
    out.loc[out["final_signal"] != 1, "position_size"] = 0.0

    out["force_exit"] = fng > greed_threshold

    # spec: use yesterday's signal for today's position (no lookahead)
    out["position"] = out["final_signal"].shift(1).fillna(0).astype(int)
    out["size"] = out["position_size"].shift(1).fillna(0.0)

    return out
