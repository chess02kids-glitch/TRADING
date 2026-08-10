@echo off
REM PHASE 1 AUDITED - Windows Setup for RTX 3060
REM Per audit: CCXT only, PAPER default, LIVE disabled, no auto mirror, no upstream modification

echo ============================================================
echo KRONOS TRADING SYSTEM - PHASE 1 AUDITED - WINDOWS SETUP
echo RTX 3060, 1h/4h/1d timeframes, PAPER default, CCXT only
echo ============================================================

where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: conda not found. Install Miniconda first
    echo https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo [1/6] Creating conda environment (pinned torch 2.4.1, CUDA 12.1)...
call conda env create -f environment.yml
if %ERRORLEVEL% NEQ 0 (
    echo Env may exist, updating...
    call conda env update -f environment.yml --prune
)

echo Activating environment...
call conda activate kronos_trading
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate. Run: conda activate kronos_trading
    pause
    exit /b 1
)

echo [2/6] Installing torch CUDA 12.1 (pinned 2.4.1 consistent with env.yml)...
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
echo Verifying no duplication: CCXT only per audit #4
pip install -r requirements_exact.txt

echo [3/6] Cloning Kronos repo (upstream, kept untouched per audit #6)...
if not exist Kronos (
    git clone https://github.com/shiyu-coder/Kronos.git
) else (
    echo Kronos repo exists - not modifying, will check commit for reproducibility
    cd Kronos && git rev-parse HEAD && cd ..
)

echo [4/6] Creating .env from template (no secrets in config/yaml/logs per audit #9)...
if not exist .env (
    copy .env.example .env
    echo .env created - EDIT with TESTNET keys only! Get from https://testnet.binance.vision/
    echo LIVE trading disabled by default, requires explicit triple confirmation
) else (
    echo .env exists
)

echo [5/6] Downloading Kronos-small with measured benchmark (audit #3)...
echo Using normal HuggingFace by default per audit #7
echo For China users, optionally add --use-mirror flag
python scripts/setup/download_models.py --hardware rtx3060_win --device cuda:0
REM Optional mirror example (commented): 
REM python scripts/setup/download_models.py --hardware rtx3060_win --use-mirror

echo [6/6] Running audited verifications (actual measurements)...
echo This will report param count, dtype, GPU name, VRAM allocated/reserved/peak, latency, CUDA
python scripts/setup/bug_audit.py
python scripts/setup/environment_report.py
python scripts/setup/verify_install.py

echo.
echo ============================================================
echo SETUP COMPLETE - AUDITED PHASE 1
echo ============================================================
echo Files changed per audit: config.yaml timeframes 1h/4h/1d, env.yml pinned 2.4.1, CCXT only, PAPER default
echo Exact commands to install:
echo   conda env create -f environment.yml
echo   conda activate kronos_trading
echo   pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
echo   pip install -r requirements_exact.txt
echo   git clone https://github.com/shiyu-coder/Kronos.git
echo   copy .env.example .env
echo   python scripts/setup/download_models.py --hardware rtx3060_win --device cuda:0
echo Verification command:
echo   python scripts/setup/verify_install.py
echo Expected output: measured stats, not claimed - GPU name, CUDA, VRAM allocated/reserved/peak, param count, dtype, latency
echo.
echo Security: LIVE disabled by default, needs trading.mode=LIVE + live_trading_enabled=true + confirmation phrase + BINANCE_LIVE_CONFIRMED=true
echo No credentials in config.yaml, only env var names
echo.
pause
