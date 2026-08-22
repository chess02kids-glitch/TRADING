# Freqtrade HAR Research Experiment

## What This Is

A paper trading research experiment testing
whether HAR volatility regime classification
improves RSI strategy performance on crypto.

## Status

PAPER TRADING ONLY.
No real money. No live orders.
`dry_run: true` in all configs.
This is a research experiment.

## Research Hypothesis

High volatility periods (HAR regime = HIGH)
are worse for RSI mean-reversion entries.
Filtering them out should improve
risk-adjusted returns.

## Three Strategies

| Strategy | Class | HAR Filter | Purpose |
|---|---|---|---|
| A | RSIBaseline | None | Baseline |
| B | RSIHARFiltered | Skip HIGH | Experimental |
| C | RSIHARInverse | Only HIGH | Control |

## Pre-Registered Gate Criteria

Must pass ALL 6 before declaring success:

C1: Strategy B Sharpe > Strategy A Sharpe
C2: Strategy B max drawdown < Strategy A
C3: Strategy B total trades >= 30
C4: Strategy C Sharpe < Strategy A Sharpe
C5: Results hold on BTC AND ETH
C6: Results stable across time periods

## Installation

```bash
pip install freqtrade
pip install psycopg[binary] python-dotenv
```

## Download Data (Run Once)

```bash
freqtrade download-data \
  --config freqtrade_har/config/config_backtest.json \
  --userdir freqtrade_har/user_data \
  --pairs BTC/USDT ETH/USDT \
  --timeframes 1h \
  --days 730
```

## Run Backtests

Strategy A (baseline):
```bash
freqtrade backtesting \
  --strategy RSIBaseline \
  --strategy-path freqtrade_har/strategies \
  --config freqtrade_har/config/config_backtest.json \
  --userdir freqtrade_har/user_data \
  --timerange 20240101-20260101
```

Strategy B (experimental):
```bash
freqtrade backtesting \
  --strategy RSIHARFiltered \
  --strategy-path freqtrade_har/strategies \
  --config freqtrade_har/config/config_backtest.json \
  --userdir freqtrade_har/user_data \
  --timerange 20240101-20260101
```

Strategy C (control):
```bash
freqtrade backtesting \
  --strategy RSIHARInverse \
  --strategy-path freqtrade_har/strategies \
  --config freqtrade_har/config/config_backtest.json \
  --userdir freqtrade_har/user_data \
  --timerange 20240101-20260101
```

## Run Paper Trading (Strategy B)

```bash
export SUPABASE_DB_URL=postgresql://...

freqtrade trade \
  --strategy RSIHARFiltered \
  --strategy-path freqtrade_har/strategies \
  --config freqtrade_har/config/config_paper.json \
  --userdir freqtrade_har/user_data
```

## Required Environment Variable

```
SUPABASE_DB_URL=postgresql://your_connection_string
```

## Evaluating Results (Day 30)

After 30 days of paper trading:
1. Record Strategy B paper trading metrics
2. Compare against Strategy A backtest
3. Apply C1-C6 gate criteria
4. Document findings in research log

## Important Notes

- Backtest of Strategy B = Strategy A
  (HAR filter needs live DB, unavailable
  in historical backtest)
- Live paper trading of Strategy B includes
  real HAR regime from Supabase
- Minimum 30 trades before any conclusions
- This experiment does not modify or
  affect the HAR alert bot in any way

## WARNING

RSI has not been proven to generate
consistent crypto trading profits.
HAR regime filtering has not been proven
to improve RSI strategy performance.
These are hypotheses under investigation.
Results may be negative.
Negative results are equally valid
scientific outcomes.
