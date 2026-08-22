"""
Dashboard configuration constants.
All tunable values in one place.
"""

# Assets to display
ASSETS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"

# Calibration period
CALIBRATION_TOTAL_DAYS = 30

# Auto-refresh interval (seconds)
AUTO_REFRESH_SECONDS = 300  # 5 minutes

# How many predictions to show in table
RECENT_PREDICTIONS_ROWS = 50

# How many breakouts to show
BREAKOUT_TABLE_ROWS = 20

# Minimum predictions for calibration stats
MIN_PREDICTIONS_FOR_STATS = 24

# Breakout threshold (must match bot)
BREAKOUT_THRESHOLD = 2.0

# Regime colors
REGIME_COLORS = {
    "low": "#00ff88",
    "medium": "#ffaa00",
    "high": "#ff4444",
    "unknown": "#888888",
}

# Theme colors
BACKGROUND_COLOR = "#0e1117"
CARD_COLOR = "#1e2130"
POSITIVE_COLOR = "#00ff88"
NEGATIVE_COLOR = "#ff4444"
NEUTRAL_COLOR = "#888888"
ACCENT_COLOR = "#4f8ef7"
