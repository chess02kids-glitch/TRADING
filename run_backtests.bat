@echo off
set PYTHONPATH=.
echo Downloading data for freqtrade backtests...
.\.venv\Scripts\freqtrade download-data --config freqtrade_har/config/config_backtest.json --userdir freqtrade_har/user_data --timerange 20240101-20260101 -t 1h > freqtrade_har\download.log 2>&1

echo Running HARStopBaseline
.\.venv\Scripts\freqtrade backtesting --strategy HARStopBaseline --strategy-path freqtrade_har/strategies --config freqtrade_har/config/config_backtest.json --userdir freqtrade_har/user_data --timerange 20240101-20260101 --export trades > freqtrade_har\baseline.log 2>&1
echo Done baseline

echo Running HARStopDynamic
.\.venv\Scripts\freqtrade backtesting --strategy HARStopDynamic --strategy-path freqtrade_har/strategies --config freqtrade_har/config/config_backtest.json --userdir freqtrade_har/user_data --timerange 20240101-20260101 --export trades > freqtrade_har\dynamic.log 2>&1
echo Done dynamic

echo Running HARStopInverse
.\.venv\Scripts\freqtrade backtesting --strategy HARStopInverse --strategy-path freqtrade_har/strategies --config freqtrade_har/config/config_backtest.json --userdir freqtrade_har/user_data --timerange 20240101-20260101 --export trades > freqtrade_har\inverse.log 2>&1
echo Done inverse

echo Running HARStopDynamic 20240101-20240901
.\.venv\Scripts\freqtrade backtesting --strategy HARStopDynamic --strategy-path freqtrade_har/strategies --config freqtrade_har/config/config_backtest.json --userdir freqtrade_har/user_data --timerange 20240101-20240901 --export trades > freqtrade_har\dynamic_p1.log 2>&1
echo Done P1

echo Running HARStopDynamic 20240901-20250501
.\.venv\Scripts\freqtrade backtesting --strategy HARStopDynamic --strategy-path freqtrade_har/strategies --config freqtrade_har/config/config_backtest.json --userdir freqtrade_har/user_data --timerange 20240901-20250501 --export trades > freqtrade_har\dynamic_p2.log 2>&1
echo Done P2

echo Running HARStopDynamic 20250501-20260101
.\.venv\Scripts\freqtrade backtesting --strategy HARStopDynamic --strategy-path freqtrade_har/strategies --config freqtrade_har/config/config_backtest.json --userdir freqtrade_har/user_data --timerange 20250501-20260101 --export trades > freqtrade_har\dynamic_p3.log 2>&1
echo Done P3
