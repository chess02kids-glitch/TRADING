"""
NautilusTrader HAR Volatility Targeting
Configuration constants.
"""

# Assets
ASSETS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"

# Volatility targeting
TARGET_VOL_PER_BAR = 0.02  # 2% per bar
MIN_ALLOCATION = 0.05       # 5% minimum
MAX_ALLOCATION = 1.00       # 100% maximum

# Rebalancing
REBALANCE_THRESHOLD = 0.05  # 5% drift

# Portfolio
INITIAL_CAPITAL = 10_000.0  # USD
FEES_BPS = 10               # 0.1% = 10bps

# Backtest timerange (pre-registered)
BACKTEST_START = "2024-01-01"
BACKTEST_END   = "2026-01-01"

# Stability periods
STABILITY_PERIODS = [
    ("2024-01-01", "2024-09-01"),
    ("2024-09-01", "2025-05-01"),
    ("2025-05-01", "2026-01-01"),
]

# Minimum bars for HAR prediction
HAR_MIN_TRAIN = 24
