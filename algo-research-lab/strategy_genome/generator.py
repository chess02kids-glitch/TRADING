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
