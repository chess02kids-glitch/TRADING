# Repository Audit

## What Already Exists
1. **Core Data Structures**: 
   - Supabase setup with `ohlcv_raw` (BTC/ETH 1h/4h data) and migration logic (`001_initial_schema.sql`).
   - Cached CSV data in `data/` and `sandbox/pattern_research/cache/`.
2. **Strategy Infrastructures**:
   - `freqtrade_har/`: A Freqtrade deployment incorporating the HAR volatility model.
   - `vectorbt_har/`: VectorBT components integrating HAR logic.
   - `nautilus_har/`: NautilusTrader experiments/modules.
   - `sandbox/pattern_research/`: Recently executed standalone pattern-discovery testbed demonstrating disciplined chronological testing and 6-gate validation (DM tests, temporal stability).
3. **Execution & Live Monitoring**:
   - `kronos_trading/`: Live alerting and DB logging.
   - `execution/`: Execution routing and configurations.
   - `dashboard/`: Analytics UI.
4. **Tooling & Packages**:
   - Backtesting engines: VectorBT, Freqtrade (installed via Pip/Venv). Jesse is NOT installed locally.
   - Data / Stats: Pandas, NumPy, SciPy, CCXT, yfinance, PostgreSQL tools.

## What Can Be Reused
- **VectorBT**: Excellent for massive vectorized parameter sweeps during the Fast Screening stage.
- **Freqtrade**: Reliable for realistic step-by-step strategy simulation and eventual paper-trading validation (accounting for fees, slippage, real-world execution).
- **Supabase OHLCV DB**: Can be queried for clean historical data, eliminating the need to re-download.
- **Pattern Research Sandbox Validation Logic**: The strict G1-G6 validation gates (Diebold-Mariano tests, temporal degradation checks) built for Phase 9 can be generalized and integrated into the new lab.
- **HAR Volatility Model**: Can be imported as a frozen regime filter or sizing overlay for the ablation studies.

## What Should Remain Untouched
- Existing production execution and live-trading alert code (`kronos_trading/`, `execution/`).
- The internal mechanics of the `HAR` volatility model (treat as a frozen API/module).
- The `sandbox/pattern_research` directory (which holds finalized, static historical research).
- User's existing Freqtrade configuration logic that is currently live.

## What Must Be Built
- **Algo Research Lab (`algo-research-lab/`)**: A highly modular environment specifically tailored to hypothesis generation, fast screening, and robustness checks.
- **Supabase Research Schema**: New tables to track runs, strategy variants, statistical results, and rejections (`research_runs`, `strategy_hypotheses`, `backtests`, etc.).
- **Strategy Genome**: A declarative configuration system defining features, entry signals, exits, regimes, and risk limits.
- **Agentic Loop Coordinator**: A script/system managing the OODA loop (Hypothesize -> Implement -> Test -> Diagnose -> Persist).
- **Cost & Stability Stress Testers**: Utilities to perturb parameters, inflate slippage/fees, and block-bootstrap the time series.

## Important Risks
- **Look-Ahead Bias**: Reusing Pandas dataframes aggressively without `.shift(1)` strictness can easily leak future data. Automated leakage checks will be required.
- **Overfitting & Multiple Testing**: Testing hundreds of generated permutations dramatically increases the likelihood of finding a false positive. We must rely heavily on Deflated Sharpe Ratio, walk-forward OOS stability, and the final locked holdout.
- **Data Integrity**: Funding rates and order book depth data are currently not clearly present in the standard `ohlcv_raw` schema, limiting the immediate viability of Statistical Arbitrage and Carry strategies without new data pipelines.

## Data Availability
- **Primary Sets**: BTC/USDT and ETH/USDT (1h and 4h timeframes) are readily available.
- **Coverage**: Full coverage up to recent timestamps (proven by recent 730-day fetch runs in the sandbox).

## Available Tooling
- **VectorBT**: Core engine for Strategy Generation + Fast Screening.
- **Freqtrade**: Core engine for Paper Trading + Realistic Execution Costing.
- **CCXT**: For pulling any missing historical candles if we need to expand the universe.
- **SciPy**: For robust statistical analysis (DM test, Bootstrap).
