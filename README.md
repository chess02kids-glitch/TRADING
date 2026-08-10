# Kronos Trading System - $100 Crypto | BTC/ETH | Binance | RTX 3060

Production-ready automated trading system using Tsinghua's Kronos foundation model (36.3k stars), personalized for your setup.

## Your Personalized Config
- **Capital:** $100 USDT (Binance)
- **Market:** Crypto (BTC/USDT, ETH/USDT)
- **Broker:** Binance (Testnet first!)
- **Timeframes:** Multi (15m, 1h, 1d) - Primary 1h swing
- **Hardware:** RTX 3060 Windows - Kronos-small (24.7M params, ~4GB VRAM)
- **Risk:** Conservative 1% per trade, 3% daily loss limit
- **Goal:** Slow growth + validation via paper trading

## Project Structure
```
kronos_trading_system/
├── config/
│   └── config.yaml          # Master config (tune without editing code)
├── data/
│   ├── raw/                 # Raw OHLCV from Binance
│   ├── processed/           # Cleaned for Kronos
│   └── db/                  # SQLite + trade logs
├── logs/
├── models/                  # Kronos downloaded models
├── scripts/
│   ├── setup/               # PHASE 1 - Environment & Model Download
│   │   ├── download_models.py
│   │   ├── verify_install.py
│   │   └── bug_patches.py
│   ├── data/                # PHASE 2 - Data Pipeline (next)
│   ├── prediction/          # PHASE 3 - Kronos Inference
│   ├── risk/                # PHASE 4 - Risk Management
│   ├── strategy/            # PHASE 5 - Trading Logic
│   ├── backtest/            # PHASE 6 - Backtesting
│   ├── broker/              # PHASE 7 - Binance Execution
│   ├── automation/          # PHASE 8 - Scheduler, Alerts, Dashboard
│   ├── finetune/            # PHASE 9 - Fine-tuning
│   └── go_live/             # PHASE 10 - Go-live checklist
├── environment.yml          # Conda env (RTX 3060 CUDA 12.1)
├── requirements_exact.txt   # Pinned pip deps
├── .env.example             # Template for API keys
├── setup_windows.bat        # One-click Windows setup
└── setup_linux.sh           # One-click Linux setup
```

## PHASE 1: Environment Setup (Current Phase)

### Quick Start (Windows RTX 3060)

**Option A - One-Click (Recommended):**
```bat
# Double-click or run in terminal:
setup_windows.bat
```

**Option B - Manual Conda:**
```bash
# 1. Create env
conda env create -f environment.yml
conda activate kronos_trading

# 2. Install torch for CUDA 12.1 (RTX 3060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install rest
pip install -r requirements_exact.txt

# 4. Clone Kronos
git clone https://github.com/shiyu-coder/Kronos.git

# 5. Create .env
copy .env.example .env
# Edit .env with your BINANCE TESTNET keys! Get from https://testnet.binance.vision/

# 6. Download model (RTX 3060 optimal = small)
python scripts/setup/download_models.py --hardware rtx3060_win

# 7. Verify everything
python scripts/setup/verify_install.py

# 8. Patch known bugs
python scripts/setup/bug_patches.py
```

**Linux/WSL2:**
```bash
chmod +x setup_linux.sh
./setup_linux.sh
```

### Model Selection Guide (Personalized for You)

| Model | Params | VRAM | Context | Best For | Your Use |
|-------|--------|------|---------|----------|----------|
| Kronos-mini | 4.1M | ~2GB | 2048 | Laptop 8GB no GPU, 15m fast | Fallback if VRAM low |
| **Kronos-small** | **24.7M** | **~4GB** | **512** | **RTX 3060, multi-timeframe** | **✅ YOUR PRIMARY** |
| Kronos-base | 102M | ~8GB | 512 | RTX 4090, max accuracy | Optional for daily |

**Your RTX 3060 (12GB):** Use **Kronos-small** for 15m/1h realtime, switch to base for daily swing research.

### After Phase 1 - Test Commands

```bash
# Verify installation
python scripts/setup/verify_install.py

# Test model download
python scripts/setup/download_models.py --hardware rtx3060_win --device cuda:0 --skip-verify false

# Check bug patches
python scripts/setup/bug_patches.py

# Should see: ✅ ALL CHECKS PASSED
```

### Security Checklist

- [ ] `.env` file contains `BINANCE_TESTNET=true` (not live!)
- [ ] `.env` is in `.gitignore` (verified by verify script)
- [ ] No hardcoded API keys in config.yaml (uses env vars)
- [ ] Start with TESTNET: https://testnet.binance.vision/
- [ ] Only switch to live after 2+ weeks profitable paper trading

### Troubleshooting

**CUDA not found on RTX 3060:**
```bash
# Check driver
nvidia-smi
# Should show CUDA 12.1, if not update: https://developer.nvidia.com/cuda-downloads

# Reinstall torch
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**China/HF download slow:**
```bash
# Use mirror
export HF_ENDPOINT=https://hf-mirror.com
# Windows:
set HF_ENDPOINT=https://hf-mirror.com
python scripts/setup/download_models.py --use-mirror
```

**Import error Kronos:**
```bash
cd Kronos && git pull origin master
cd .. && python scripts/setup/bug_patches.py
```

## Next Phases (Coming After Your Confirmation)

- **Phase 2:** Data fetcher for BTC/ETH Binance, SQLite storage, real-time streaming
- **Phase 3:** Kronos inference wrapper, batch prediction, confidence scoring
- **Phase 4:** Position sizing (Kelly 1%), stop-loss, circuit breaker, exposure limits
- **Phase 5:** Signal generation from Kronos output
- **Phase 6:** Backtesting engine, Sharpe/Sortino, equity curve
- **Phase 7:** Binance Testnet paper trading
- **Phase 8:** Telegram alerts + Streamlit dashboard
- **Phase 9:** Fine-tune Kronos on your BTC/ETH data
- **Phase 10:** Go-live checklist + emergency shutdown

---

**Rules Followed:**
- Complete copy-paste-ready code, error handling, logging
- Config files for tuning without code edit
- Security: no hardcoded keys
- After each phase: test command

**Your Setup Summary File:** `config/config.yaml`
