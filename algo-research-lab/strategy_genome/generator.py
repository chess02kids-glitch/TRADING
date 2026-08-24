import pandas as pd
import numpy as np
import vectorbt as vbt
from backtesting.regimes import classify_regimes
from typing import Tuple

class StrategyGenerator:
    """
    Compiles an expanded Gen 2 strategy genome into VectorBT signals (entries, exits, sizes).
    """
    
    @staticmethod
    def compile_genome(df: pd.DataFrame, genome: dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df.get("volume", pd.Series(0, index=df.index))
        
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        sizes = pd.Series(1.0, index=df.index)
        
        # Helper for Multi-Timeframe (simulated 4h dynamically)
        # Using rolling windows to prevent lookahead instead of true resampling which can leak if not shifted correctly.
        # 4h = 4 periods of 1h
        close_4h = close
        high_4h = high.rolling(4).max()
        low_4h = low.rolling(4).min()
        
        # 1. Base Directional Signal
        direction = genome.get("direction", {})
        family = direction.get("family")
        indicator = direction.get("indicator")
        params = direction.get("params", {})
        
        if family == "trend" and indicator == "sma_crossover":
            fast_window = params.get("fast", 10)
            slow_window = params.get("slow", 50)
            fast_ma = close.rolling(fast_window).mean()
            slow_ma = close.rolling(slow_window).mean()
            entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
            exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
            
        elif family == "breakout":
            if indicator == "donchian":
                window = params.get("window", 20)
                rolling_high = high.shift(1).rolling(window).max()
                rolling_low = low.shift(1).rolling(window).min()
                entries = close > rolling_high
                exits = close < rolling_low
            elif indicator == "volatility_expansion":
                window = params.get("window", 20)
                atr = vbt.ATR.run(high, low, close, window=14).atr
                rolling_high = high.shift(1).rolling(window).max()
                # Breakout confirmed by ATR expansion
                atr_expanding = atr > atr.rolling(window).mean()
                entries = (close > rolling_high) & atr_expanding
                rolling_low = low.shift(1).rolling(window).min()
                exits = close < rolling_low
                
        elif family == "mean_reversion":
            if indicator == "rsi":
                window = params.get("window", 14)
                lower_bound = params.get("lower", 30)
                upper_bound = params.get("upper", 70)
                rsi = vbt.RSI.run(close, window=window).rsi
                entries = (rsi < lower_bound) & (rsi.shift(1) >= lower_bound)
                exits = (rsi > upper_bound) & (rsi.shift(1) <= upper_bound)
            elif indicator == "z_score":
                window = params.get("window", 24)
                threshold = params.get("threshold", -2.0)
                mean = close.rolling(window).mean()
                std = close.rolling(window).std()
                z = (close - mean) / std
                entries = (z < threshold) & (z.shift(1) >= threshold)
                exits = z > 0
            elif indicator == "atr_normalized":
                window = params.get("window", 24)
                threshold = params.get("threshold", -2.0)
                atr = vbt.ATR.run(high, low, close, window=14).atr
                mean = close.rolling(window).mean()
                dev = (close - mean) / atr
                entries = (dev < threshold) & (dev.shift(1) >= threshold)
                exits = dev > 0
                
        elif family == "momentum":
            if indicator == "roc":
                window = params.get("window", 14)
                threshold = params.get("threshold", 0)
                roc = close.pct_change(periods=window) * 100
                entries = (roc > threshold) & (roc.shift(1) <= threshold)
                exits = (roc < -threshold) & (roc.shift(1) >= -threshold)
            elif indicator == "acceleration":
                window = params.get("window", 14)
                roc1 = close.pct_change(periods=window)
                roc2 = roc1.shift(window)
                entries = (roc1 > roc2) & (roc1 > 0)
                exits = (roc1 < roc2) & (roc1 < 0)

        # 2. Multi-Timeframe Context
        mtf = genome.get("multi_timeframe", {})
        if mtf:
            mtf_type = mtf.get("type")
            if mtf_type == "trend":
                # 4h SMA
                sma4h = close_4h.rolling(50).mean()
                entries = entries & (close_4h > sma4h)

        # 3. Confirmation Logic (AND condition on entries)
        confirmation = genome.get("confirmation", {})
        if confirmation:
            conf_type = confirmation.get("type")
            if conf_type == "trend":
                sma200 = close.rolling(200 * 24).mean()
                entries = entries & (close > sma200)
            elif conf_type == "volatility":
                tr = (high - low) / close
                vol = tr.rolling(24).mean()
                entries = entries & (vol > vol.rolling(168).mean())

        # 4. Regime Filtering (Binary)
        regime = genome.get("regime", {})
        if regime:
            regimes_df = classify_regimes(df)
            if "allowed_vol_regimes" in regime:
                entries = entries & regimes_df["volatility_regime"].isin(regime["allowed_vol_regimes"])
            if "allowed_trend_regimes" in regime:
                entries = entries & regimes_df["trend_regime"].isin(regime["allowed_trend_regimes"])

        # 5. Continuous Dynamic Sizing
        sizing_logic = genome.get("sizing", {})
        if sizing_logic:
            size_type = sizing_logic.get("type")
            
            if size_type == "volatility_inverse":
                # Base size scaled inversely by normalized volatility
                tr = (high - low) / close
                vol = tr.rolling(168).mean()
                vol_90d = vol.rolling(2160).mean()
                # If vol is 2x the 90d mean, size is 0.5x
                # If vol is 0.5x the 90d mean, size is 2.0x (capped at 1.0 or whatever limit)
                # Since we don't want leverage over 1.0 for these base tests, we cap at 1.0
                ratio = (vol_90d / vol).fillna(1.0)
                sizes = np.clip(ratio, 0.1, 1.0)
                
            elif size_type == "trend_strength":
                # Scale by trend strength (0 to 1)
                sma20 = close.rolling(20 * 24).mean()
                sma50 = close.rolling(50 * 24).mean()
                diff_pct = abs(sma20 - sma50) / close
                diff_mean = diff_pct.rolling(2160).mean()
                diff_std = diff_pct.rolling(2160).std()
                z = (diff_pct - diff_mean) / diff_std
                # Map z-score (-2 to +2) to size (0 to 1)
                strength = (z + 2) / 4
                sizes = np.clip(strength, 0.1, 1.0)
                
            elif size_type == "breakout_strength":
                window = params.get("window", 20)
                rolling_high = high.shift(1).rolling(window).max()
                atr = vbt.ATR.run(high, low, close, window=14).atr
                dist = (close - rolling_high) / atr
                # 0 ATR = 0 size, 2 ATR = 1 size
                strength = dist / 2.0
                sizes = np.clip(strength, 0.1, 1.0)

        # 6. Alternative Exits (OR condition on exits)
        exit_logic = genome.get("exit", {})
        if exit_logic:
            exit_type = exit_logic.get("type")
            if exit_type == "time_based":
                hold_bars = exit_logic.get("hold_bars", 24)
                time_exits = entries.shift(hold_bars).fillna(False)
                exits = exits | time_exits
                
        return entries, exits, sizes


# ============================================================================
# GEN 1 v2 RESET - NEW SIGNAL GENERATOR (six genuinely-open signal types)
# ============================================================================
# Ban-list enforced: this generator can NEVER produce the closed signal
# families from Generation 1 (EMA/SMA crossover, RSI thresholds, Donchian
# breakout, ROC momentum, candlestick patterns, volume-spike direction,
# time-of-day filters, Bollinger bands). All six signal types below come
# from the "what has NOT been tested" list in the lab brief.
#
# Funding thresholds are in PERCENT units (e.g. -0.02 == -0.02%/8h) and are
# converted to decimals internally; raw funding data is decimal fraction.

import copy as _copy
import hashlib as _hashlib
import json as _json

SIGNAL_TYPES_V2 = [
    "funding_rate_contrarian",
    "spread_zscore",
    "har_regime_sized",
    "vol_regime_breakout",
    "multi_asset_momentum",
    "funding_trend",
]

BANNED_SIGNAL_FAMILIES = [
    "ema_crossover", "sma_crossover", "rsi_thresholds", "donchian_breakout",
    "roc_momentum", "candlestick_pattern", "volume_spike", "time_of_day",
    "bollinger_bands",
]

PARAM_SPACE_V2 = {
    "funding_rate_contrarian": {
        "long_threshold": ("uniform", -0.02, -0.005),   # percent per 8h
        "short_threshold": ("uniform", 0.005, 0.05),    # percent per 8h
        "holding_bars": ("randint", 1, 24),
        "size_pct": ("uniform", 0.1, 1.0),
        "exit_type": ("choice", ["fixed_hold", "funding_flip"]),
    },
    "spread_zscore": {
        "zscore_window": ("randint", 12, 168),
        "entry_zscore": ("uniform", 1.5, 3.0),
        "exit_zscore": ("uniform", 0.0, 0.5),
        "size_pct": ("uniform", 0.1, 1.0),
        "direction": ("choice", ["mean_revert", "momentum"]),
    },
    "har_regime_sized": {
        "base_signal": ("choice", ["breakout", "reversion"]),
        "entry_regime": ("choice", ["low", "medium"]),
        "har_window": ("choice", [5, 22]),
        "breakout_multiplier": ("uniform", 1.5, 3.0),
        "size_formula": ("choice", ["inverse_vol", "fixed"]),
        "holding_bars": ("randint", 1, 12),
    },
    "vol_regime_breakout": {
        "atr_window": ("randint", 10, 50),
        "expansion_factor": ("uniform", 1.2, 2.5),
        "breakout_lookback": ("randint", 5, 50),
        "holding_bars": ("randint", 2, 24),
        "size_pct": ("uniform", 0.1, 1.0),
        "regime_filter": ("choice", ["low_only", "any"]),
    },
    "multi_asset_momentum": {
        "primary_asset": ("choice", ["ETH/USDT", "BTC/USDT"]),
        "lookback_bars": ("randint", 12, 96),
        "momentum_threshold": ("uniform", 0.005, 0.05),
        "require_confirmation": ("boolean",),
        "holding_bars": ("randint", 2, 24),
        "size_pct": ("uniform", 0.1, 1.0),
    },
    "funding_trend": {
        "funding_ma_window": ("randint", 3, 24),
        "trend_threshold": ("uniform", 0.001, 0.02),   # percent per 8h
        "price_confirm": ("boolean",),
        "holding_bars": ("randint", 2, 24),
        "size_pct": ("uniform", 0.1, 1.0),
    },
}


def _sample(rng, spec):
    kind = spec[0]
    if kind == "uniform":
        return float(rng.uniform(spec[1], spec[2]))
    if kind == "randint":
        return int(rng.randint(spec[1], spec[2] + 1))
    if kind == "choice":
        return spec[1][int(rng.randint(len(spec[1])))]
    if kind == "boolean":
        return bool(rng.random() < 0.5)
    raise ValueError(f"unknown param spec {spec}")


def genome_id(genome: dict) -> str:
    payload = {k: v for k, v in genome.items() if k not in ("genome_id", "name", "parents", "generation")}
    return _hashlib.md5(_json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


class GenomeGeneratorV2:
    """Random genome sampling for the six open signal types."""

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate(self, n: int) -> list:
        genomes = []
        for i in range(n):
            st = SIGNAL_TYPES_V2[i % len(SIGNAL_TYPES_V2)]
            genomes.append(self.generate_one(st))
        return genomes

    def generate_many(self, signal_type: str, n: int) -> list:
        return [self.generate_one(signal_type) for _ in range(n)]

    def generate_many_focused(self, signal_type: str, n: int, space_overrides: dict) -> list:
        return [self.generate_one(signal_type, space_overrides=space_overrides) for _ in range(n)]

    def generate_one(self, signal_type: str, overrides: dict = None, space_overrides: dict = None) -> dict:
        g = {"signal_type": signal_type}
        if signal_type == "spread_zscore":
            g["asset_a"] = "BTC/USDT"
            g["asset_b"] = "ETH/USDT"
        space = dict(PARAM_SPACE_V2[signal_type])
        if space_overrides:
            space.update(space_overrides)
        for key, spec in space.items():
            g[key] = _sample(self.rng, spec)
        if signal_type == "multi_asset_momentum":
            g["confirmation_asset"] = "BTC/USDT" if g["primary_asset"] == "ETH/USDT" else "ETH/USDT"
        if signal_type == "spread_zscore":
            g["asset_a"] = "BTC/USDT"
            g["asset_b"] = "ETH/USDT"
        if overrides:
            g.update(overrides)
        g["genome_id"] = genome_id(g)
        g["name"] = f"{signal_type[:10]}_{g['genome_id']}"
        return g


def _hold_exits(entries: pd.Series, short_entries: pd.Series, holding: int):
    ex = entries.shift(holding).fillna(False)
    sx = short_entries.shift(holding).fillna(False)
    return ex.astype(bool), sx.astype(bool)


def _sequentialize(entries, exits, short_entries, short_exits, holding=None):
    """State machine guaranteeing alternating long/short trades.

    vectorbt raises 'SizeType.Percent does not support position reversal
    using signals' when an opposite entry fires while a position is open.
    Rule here (no lookahead): while LONG, a condition exit, an opposite
    (short) entry, or the expiry of `holding` bars counted from the ACTUAL
    entry bar closes the position. Entries only fire from flat.
    Simultaneous long+short signals on the same bar are skipped.

    `holding` is measured from the actual entry bar (not from when the
    raw level first became True), preventing the premature-exit artifact.
    """
    e = entries.to_numpy(dtype=bool)
    x = exits.to_numpy(dtype=bool) if exits is not None else np.zeros(len(e), dtype=bool)
    se = short_entries.to_numpy(dtype=bool)
    sx = short_exits.to_numpy(dtype=bool) if short_exits is not None else np.zeros(len(e), dtype=bool)
    n = len(e)
    ne = np.zeros(n, dtype=bool); nx = np.zeros(n, dtype=bool)
    nse = np.zeros(n, dtype=bool); nsx = np.zeros(n, dtype=bool)
    state = 0
    entry_bar = -1
    for i in range(n):
        if state == 1:
            due = (holding is not None) and (i - entry_bar >= holding)
            if x[i] or se[i] or due:
                nx[i] = True
                state = 0
        elif state == -1:
            due = (holding is not None) and (i - entry_bar >= holding)
            if sx[i] or e[i] or due:
                nsx[i] = True
                state = 0
        else:
            if e[i] and not se[i]:
                ne[i] = True
                state = 1
                entry_bar = i
            elif se[i] and not e[i]:
                nse[i] = True
                state = -1
                entry_bar = i
    idx = entries.index
    return (pd.Series(ne, index=idx), pd.Series(nx, index=idx),
            pd.Series(nse, index=idx), pd.Series(nsx, index=idx))


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=2).mean()


def compile_genome_v2(genome: dict, ctx: dict):
    """Compile a v2 genome into a SignalSpec for the 5-gate pipeline."""
    from research.pipeline import SignalSpec
    from agent.lab_context import get_funding_ma

    st = genome["signal_type"]
    F = pd.Series(False, index=None)
    size = float(genome.get("size_pct", 1.0))

    if st == "funding_rate_contrarian":
        px_df = ctx["btc_B"]
        price = px_df["close"]
        F = pd.Series(False, index=price.index)
        fr = ctx["funding_h_B"]
        long_thr = genome["long_threshold"] / 100.0
        short_thr = genome["short_threshold"] / 100.0
        entries = (fr < long_thr).fillna(False)
        short_entries = (fr > short_thr).fillna(False)
        if genome["exit_type"] == "funding_flip":
            exits = (fr > 0).fillna(False) & entries.cummax()
            short_exits = (fr < 0).fillna(False) & short_entries.cummax()
            holding = None
        else:
            exits = pd.Series(False, index=price.index)
            short_exits = pd.Series(False, index=price.index)
            holding = int(genome["holding_bars"])
        entries, exits, short_entries, short_exits = _sequentialize(entries.fillna(False), exits,
                                                                    short_entries.fillna(False), short_exits, holding=holding)
        return SignalSpec(price, entries, exits,
                          short_entries, short_exits, float(genome["size_pct"]),
                          asset="BTC/USDT", meta={"window": "B", "venue": "Bitstamp BTC/USD price + Binance funding"})

    if st == "funding_trend":
        px_df = ctx["btc_B"]
        price = px_df["close"]
        F = pd.Series(False, index=price.index)
        fma = get_funding_ma(ctx)(int(genome["funding_ma_window"]))
        thr = genome["trend_threshold"] / 100.0
        entries = (fma > thr).fillna(False)
        short_entries = (fma < -thr).fillna(False)
        if genome["price_confirm"]:
            ma24 = price.rolling(24, min_periods=2).mean()
            entries = entries & (price > ma24).fillna(False)
            short_entries = short_entries & (price < ma24).fillna(False)
        no_exit = pd.Series(False, index=price.index)
        entries, exits, short_entries, short_exits = _sequentialize(
            entries, no_exit, short_entries, no_exit, holding=int(genome["holding_bars"]))
        return SignalSpec(price, entries, exits,
                          short_entries, short_exits, float(genome["size_pct"]),
                          asset="BTC/USDT", meta={"window": "B"})

    if st == "spread_zscore":
        spread = ctx["spread_A"]
        w = int(genome["zscore_window"])
        mean = spread.rolling(w, min_periods=w // 2).mean()
        std = spread.rolling(w, min_periods=w // 2).std()
        z = (spread - mean) / std
        entry_z = float(genome["entry_zscore"])
        exit_z = float(genome["exit_zscore"])
        F = pd.Series(False, index=spread.index)
        z_f = z.fillna(0.0)
        if genome["direction"] == "mean_revert":
            entries = z_f < -entry_z
            short_entries = z_f > entry_z
        else:
            entries = z_f > entry_z
            short_entries = z_f < -entry_z
        exits = z_f.abs() < exit_z
        short_exits = exits.copy()
        entries, exits, short_entries, short_exits = _sequentialize(entries, exits, short_entries, short_exits,
                                                                    holding=None)
        return SignalSpec(spread, entries, exits,
                          short_entries, short_exits, float(genome["size_pct"]),
                          leg_multiplier=2.0, asset="BTC/USDT vs ETH/USDT",
                          meta={"window": "A", "instrument": "log(BTC)-log(ETH) spread, 2 legs"})

    if st == "har_regime_sized":
        px_df = ctx["btc_A"]
        price = px_df["close"]
        variant = str(genome["har_window"])
        pred_range = ctx["har_range_A"][variant]
        regime = ctx["har_regime_A"][variant]
        in_regime = (regime == genome["entry_regime"]).fillna(False)
        move = price.diff()
        mult = float(genome["breakout_multiplier"])
        big_move_up = (move > mult * pred_range).fillna(False)
        big_move_down = (move < -mult * pred_range).fillna(False)
        if genome["base_signal"] == "breakout":
            entries = big_move_up & in_regime
            short_entries = big_move_down & in_regime
        else:
            entries = big_move_down & in_regime
            short_entries = big_move_up & in_regime
        no_exit = pd.Series(False, index=price.index)
        entries, exits, short_entries, short_exits = _sequentialize(
            entries.fillna(False), no_exit, short_entries.fillna(False), no_exit,
            holding=int(genome["holding_bars"]))
        if genome["size_formula"] == "inverse_vol":
            target = pred_range.rolling(90 * 24, min_periods=30 * 24).median()
            sizes = (target / pred_range).clip(0.1, 1.0).fillna(0.5)
        else:
            sizes = 1.0
        return SignalSpec(price, entries, exits,
                          short_entries, short_exits, sizes,
                          asset="BTC/USDT", meta={"window": "A"})

    if st == "vol_regime_breakout":
        px_df = ctx["btc_A"]
        price = px_df["close"]
        atr = _atr(px_df, int(genome["atr_window"]))
        atr_ma = atr.rolling(200, min_periods=50).mean()
        expanding = (atr > float(genome["expansion_factor"]) * atr_ma).fillna(False)
        lb = int(genome["breakout_lookback"])
        hh = px_df["high"].shift(1).rolling(lb, min_periods=2).max()
        ll = px_df["low"].shift(1).rolling(lb, min_periods=2).min()
        entries = (price > hh).fillna(False) & expanding
        short_entries = (price < ll).fillna(False) & expanding
        if genome["regime_filter"] == "low_only":
            low_vol = (atr < atr.rolling(200, min_periods=50).median()).fillna(False)
            entries = entries & low_vol
            short_entries = short_entries & low_vol
        exits, short_exits = _hold_exits(entries, short_entries, int(genome["holding_bars"]))
        if genome.get("har_regime_filter"):
            reg = ctx["har_regime_A"]["22"]
            allowed = (reg == genome["har_regime_filter"]).fillna(False)
            entries = entries & allowed
            short_entries = short_entries & allowed
        no_exit = pd.Series(False, index=price.index)
        entries, exits, short_entries, short_exits = _sequentialize(
            entries.fillna(False), no_exit, short_entries.fillna(False), no_exit,
            holding=int(genome["holding_bars"]))
        return SignalSpec(price, entries, exits,
                          short_entries, short_exits, float(genome["size_pct"]),
                          asset="BTC/USDT", meta={"window": "A"})

    if st == "multi_asset_momentum":
        primary = ctx["eth_A"] if genome["primary_asset"] == "ETH/USDT" else ctx["btc_A"]
        conf = ctx["btc_A"] if genome["primary_asset"] == "ETH/USDT" else ctx["eth_A"]
        price = primary["close"]
        lb = int(genome["lookback_bars"])
        thr = float(genome["momentum_threshold"])
        mom = price / price.shift(lb) - 1.0
        entries = (mom > thr).fillna(False)
        short_entries = (mom < -thr).fillna(False)
        if genome["require_confirmation"]:
            conf_mom = conf["close"] / conf["close"].shift(lb) - 1.0
            entries = entries & (conf_mom > 0).fillna(False)
            short_entries = short_entries & (conf_mom < 0).fillna(False)
        if genome.get("har_regime_filter"):
            reg = ctx["har_regime_A"]["22"]
            allowed = (reg == genome["har_regime_filter"]).fillna(False)
            entries = entries & allowed
            short_entries = short_entries & allowed
        no_exit = pd.Series(False, index=price.index)
        entries, exits, short_entries, short_exits = _sequentialize(
            entries, no_exit, short_entries, no_exit, holding=int(genome["holding_bars"]))
        return SignalSpec(price, entries, exits,
                          short_entries, short_exits, float(genome["size_pct"]),
                          asset=genome["primary_asset"], meta={"window": "A"})

    raise ValueError(f"unknown signal_type {st}")
