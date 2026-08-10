#!/bin/bash
# PHASE 1 AUDITED - Linux Setup for RTX 3060
# Per audit: CCXT only, PAPER default, LIVE disabled, no auto mirror, no upstream modification
set -e

echo "============================================================"
echo "KRONOS TRADING SYSTEM - PHASE 1 AUDITED - LINUX SETUP"
echo "RTX 3060, 1h/4h/1d, PAPER default, CCXT only"
echo "============================================================"

if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found"
    exit 1
fi

echo "[1/6] Creating conda environment (pinned torch 2.4.1 CUDA 12.1)..."
if conda env list | grep -q "kronos_trading"; then
    echo "Env exists, updating..."
    conda env update -f environment.yml --prune
else
    conda env create -f environment.yml
fi

eval "$(conda shell.bash hook)"
conda activate kronos_trading

echo "[2/6] Installing torch CUDA 12.1 pinned 2.4.1 (consistent with env.yml and requirements_exact.txt)..."
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_exact.txt
echo "CCXT only per audit #4 - python-binance removed to avoid duplication"

echo "[3/6] Cloning Kronos repo (upstream untouched per audit #6)..."
if [ ! -d "Kronos" ]; then
    git clone https://github.com/shiyu-coder/Kronos.git
else
    echo "Kronos exists - not modifying, checking commit"
    cd Kronos && git rev-parse HEAD && cd ..
fi

echo "[4/6] Creating .env (no secrets in code/YAML per audit #9)..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ".env created - edit with TESTNET keys! LIVE disabled by default"
else
    echo ".env exists"
fi

echo "[5/6] Downloading model - normal HF default, mirror optional per audit #7..."
echo "Default: normal HF https://huggingface.co"
echo "Optional mirror: --use-mirror flag"
python scripts/setup/download_models.py --hardware rtx3060_win --device cuda:0
# Example optional: python scripts/setup/download_models.py --hardware rtx3060_win --use-mirror --device cuda:0

echo "[6/6] Running audited verifications with actual measurements..."
python scripts/setup/bug_audit.py
python scripts/setup/environment_report.py
python scripts/setup/verify_install.py

echo ""
echo "============================================================"
echo "SETUP COMPLETE - AUDITED PHASE 1"
echo "============================================================"
echo "Files changed: config.yaml timeframes, env.yml pinned 2.4.1, CCXT only, trading modes, no upstream modification"
echo "Install commands:"
echo "  conda env create -f environment.yml"
echo "  conda activate kronos_trading"
echo "  pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121"
echo "  pip install -r requirements_exact.txt"
echo "Verification:"
echo "  python scripts/setup/verify_install.py"
echo "Expected: measured GPU name, CUDA, VRAM allocated/reserved/peak, param count, dtype, latency"
