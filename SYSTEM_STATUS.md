# Kronos Trading System — Current Status

**Date:** 2026-08-14

## Overall Status

- Phase 1: PASS
- Phase 2: PASS
- Database verification: PASS
- Supabase migration: COMPLETE
- SQLite ↔ Supabase parity: PASS
- Phase 3: IMPLEMENTED — real GPU inference **pending execution on the target machine**

## Environment

- Windows (target machine), Python 3.10.13, PyTorch 2.4.1, CUDA 12.1
- NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM
- Conda environment: `kronos_trading`
- Kronos upstream pinned to `67b630e67f6a`

## Database

Verified SQLite (target machine, not committed to Git):

`data/db/kronos_trading_verified.db`

Verified dataset: BTC/USDT & ETH/USDT at 1h/4h/1d — 45,500 OHLCV rows,
SQLite ↔ Supabase canonical SHA-256 `7e362e05…c03a527`, parity PASS.

## Supabase

PostgreSQL connection verified (target machine). Migrated tables `ohlcv_raw`,
`fetch_metadata`, `validation_reports`. Phase 3 reads the verified data source
and does **not** migrate or rewrite any data.

---

## Phase 3 — Real Kronos Inference

### What was built

The real (non-mock) Kronos inference path is now implemented end-to-end and
unit-tested:

- `kronos_trading/model.py`
  - `ModelManager` loads the **real** upstream `KronosTokenizer` +
    `Kronos` model via the upstream `from_pretrained` API and wraps them in the
    upstream `KronosPredictor`.
  - CUDA when available, explicit CPU fallback, **no silent mock fallback**.
  - Measured parameter count and dtype; device, model/tokenizer revision, and
    clear `ModelUnavailableError` reporting.
  - `KronosRealPredictor` feeds **raw** OHLCV (Kronos z-scores internally) and
    returns exactly the columns Kronos emits: `open, high, low, close, volume,
    amount`.
  - `DeterministicMockPredictor` remains for offline tests and is selected
    **only** by the explicit `--mock` flag.
- `kronos_trading/preprocess.py`
  - `closed()` (forming-candle exclusion), `validate_context()` (sorted,
    gap-free, duplicate-free, finite, OHLC-valid), `to_kronos_frame()` and
    `future_timestamps()`. Gaps are reported, never filled.
- `kronos_trading/pipeline.py` — real vs mock branch sharing one validation
  gate; builds the structured `Prediction`.
- `kronos_trading/types.py` — `Prediction` extended with structured Phase 3
  fields (context length, prediction timestamps, predicted OHLCV, model name/
  revisions, dtype, peak VRAM).
- `kronos_trading/benchmark.py` + CLI `benchmark` — separates model-load time,
  first inference, and warmed latency (with CUDA synchronisation), reports
  median/mean/min/max latency and peak VRAM.
- `kronos_trading/cli.py` — `predict` / `backtest` / `benchmark` subcommands.

### Real model facts (read from upstream source at `67b630e67f6a`)

- Model: `NeoQuasar/Kronos-small` (24.7M params documented; measured at load)
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-base`
- Upstream-pinned revisions (from Kronos `tests/test_kronos_regression.py`):
  model `901c26c1332695a2a8f243eb2f37243a37bea320`,
  tokenizer `0e0117387f39004a9016484a186a908917e22426`
- Context length: `max_context = 512`
- Input features: `open, high, low, close, volume` (`amount` derived as
  volume × mean price) + minute/hour/weekday/day/month timestamp features
- Normalisation: per-feature z-score + clip to `[-5, 5]` (done by upstream)
- Prediction API: `KronosPredictor.predict(df, x_timestamp, y_timestamp,
  pred_len, T, top_k, top_p, sample_count, verbose)` → autoregressive token
  decoding; outputs `open, high, low, close, volume, amount` per horizon step
- Generative/sampling model: **nondeterministic by default**
  (`torch.multinomial`); deterministic recipe = fixed seed + `eval()` +
  `top_k=1, top_p=1.0, sample_count=1` (the upstream regression-test recipe)

### Verification status in this environment

The implementation was validated in the **Arena sandbox**, which differs from
the target machine in three material ways:

1. **No NVIDIA GPU** — CUDA is unavailable, so GPU inference/VRAM numbers
   cannot be produced here (CPU fallback path only).
2. **Hugging Face egress blocked** — `NeoQuasar/Kronos-small` weights and
   tokenizer cannot be downloaded here (GitHub/PyPI are reachable; HF is not).
3. **Verified DB not in the Git checkout** — `data/db/kronos_trading_verified.db`
   is gitignored and lives on the target machine.

As a result, **real Kronos inference has not yet been executed against the
verified BTC/ETH dataset**, and no GPU latency/VRAM figures are claimed.

What *was* verified here:

- Upstream Kronos package imports cleanly (torch 2.4.1 installed).
- The real loader fails **explicitly** (`ModelUnavailableError`) when weights
  are absent/HF unreachable — never fakes success, never swaps to mock.
- Closed-candle exclusion, gap / duplicate / NaN / inf / invalid-OHLC /
  unsupported-timeframe / insufficient-context handling.
- CPU fallback and invalid-device handling.
- Seed/determinism helpers; mock determinism.
- Adapter wiring against the upstream `predict` contract (stub upstream).
- Full test suite: **63 passed, 2 skipped, 1 warning** (the 2 skips are the
  real-weight tests, which require the model to be present).

### To complete Phase 3 verification on the target machine (RTX 3050)

```bash
# 1. Ensure submodule + weights are present
git submodule update --init
python scripts/setup/download_models.py --hardware rtx3060_win

# 2. Real prediction (BTC/USDT 1h, next candle)
python -m kronos_trading.cli predict \
  --db data\db\kronos_trading_verified.db --symbol BTC/USDT --timeframe 1h

# 3. ETH/USDT
python -m kronos_trading.cli predict \
  --db data\db\kronos_trading_verified.db --symbol ETH/USDT --timeframe 1h

# 4. Deterministic repeatability
python -m kronos_trading.cli predict --db data\db\kronos_trading_verified.db \
  --symbol BTC/USDT --timeframe 1h --seed 0 --deterministic

# 5. Full benchmark (load / first / warmed latency / peak VRAM)
python -m kronos_trading.cli benchmark \
  --db data\db\kronos_trading_verified.db --symbol BTC/USDT --timeframe 1h

# 6. Tests (real-weight tests un-skip when the model is present)
pytest -q
```

## Testing

- Full suite: `63 passed, 2 skipped, 1 warning`
- Phase 2 audit: `7 passed`
- Offline system: `3 passed`
- Historical-range regression: `14 passed` (stale hardcoded span assertion
  fixed to track genesis→today without weakening its semantics)
- Phase 3: `22 passed, 2 skipped` (skips are the real-weight tests)

## Safety

- Trading mode: PAPER
- Live trading: DISABLED (`live_trading_enabled=false`)
- Live exchange order creation: disabled (no CCXT in `kronos_trading/`)
- No API credentials committed
- Kronos upstream source untouched (pinned `67b630e67f6a`)

## Next Phase

Phase 4+ remain blocked until real Kronos inference is executed and verified
against the verified dataset on the target machine (the commands above).
