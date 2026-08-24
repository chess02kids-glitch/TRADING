"""Fear & Greed Contrarian + BTC Dominance Filter strategy package.

Modules:
    data_fetcher      - fetch/cache F&G index, BTC dominance, load price CSVs
    signal_generator  - build entry/exit signals (no lookahead)
    backtester        - bar-by-bar backtest with stop-loss + walk-forward
    grader            - grade the strategy (A-F, score 0-100, recommendation)
    run_backtest      - CLI entry point with parameter sweep
"""

__version__ = "1.0.0"
