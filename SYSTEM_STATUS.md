# Kronos Trading System — Current Status

**Date:** 2026-08-14

## Overall Status

- Phase 1: PASS
- Phase 2: PASS
- Database verification: PASS
- Supabase migration: COMPLETE
- SQLite ↔ Supabase parity: PASS
- Phase 3: PASS — real Kronos inference verified on the RTX 3050
- Phase 4: IMPLEMENTED — chronological evaluator built and tested; real-data
  evaluation **pending execution on the target machine**

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
`fetch_metadata`, `validation_reports`. Phases 3–4 read the verified data
source and do **not** migrate or rewrite any data.

---

## Phase 3 — Real Kronos Inference (PASS)

Real `NeoQuasar/Kronos-small` inference was verified on the RTX 3050
(`cuda:0`), model revision `901c26c1332695a2a8f243eb2f37243a37bea320`,
tokenizer revision `0e0117387f39004a9016484a186a908917e22426`, 512-candle
context, deterministic argmax recipe (`seed=0`, `top_k=1`, `top_p=1.0`).
Verified examples: BTC/USDT 1h and ETH/USDT 1h. Phase 3 tests: 66 passed,
2 warnings (a `PytestReturnNotNoneWarning` and a PyTorch flash-attention
warning — neither blocks later phases).

Key components: `kronos_trading/model.py` (`ModelManager`,
`KronosRealPredictor`), `preprocess.py`, `pipeline.py`, `types.py`,
`benchmark.py`, CLI `predict` / `backtest` / `benchmark`.

---

## Phase 4 — Chronological No-Lookahead Evaluation

### Status: IMPLEMENTED (real-data run pending)

The evaluator is fully implemented and unit-tested. It has **not yet been run
against the verified dataset on the target machine** (the Arena sandbox has no
GPU, no model weights, and no verified DB), so no Phase 4 metrics are claimed
yet. Phase 4 will be marked **PASS** only after the evaluation actually runs
on the verified dataset.

### What was built

- `kronos_trading/evaluation.py`
  - `EvaluationConfig` — context length, horizon, deterministic seed/sampling
    config, direction threshold, evaluation window, max predictions.
  - `PredictionEvaluator` — strict chronological walk; loads the model once and
    reuses it for every prediction.
  - `EvaluationRow` — one prediction-vs-actual record with all fields required
    (predicted/actual OHLCV, returns, errors, direction, model/tokenizer
    revision, latency, device, deterministic config).
  - `compute_metrics` — price / direction / return metrics with safe handling
    of empty and zero-variance cases (no fabricated numbers).
- CLI `evaluate` — `python -m kronos_trading.cli evaluate ...` with `--start`,
  `--end`, `--context`, `--horizon`, `--max-predictions`,
  `--direction-threshold`, `--seed`, `--no-deterministic`, `--output`,
  `--include-rows`. Defaults are safe and reproducible.
- `tests/test_phase4.py` — 16 tests (no-lookahead, chronology, determinism,
  metrics, window selection, skip reasons, CLI gating).

### Evaluation methodology

At each historical prediction time `T` (the open time of a closed candle):

1. Load only candles with open time `< T` (context ends at the latest closed
   candle strictly before `T`).
2. Exclude the currently-forming candle (the newest candle in the dataset).
3. Validate the context (contiguous, gap-free, finite, valid OHLC).
4. Take the allowed context window (up to 512).
5. Run **real** Kronos inference (deterministic: `seed=0`, `top_k=1`,
   `top_p=1.0`, `sample_count=1`).
6. Predict the next candle(s) (`T + horizon`, default `horizon=1`).
7. **After** inference, retrieve the actual candle(s) and compare.
8. Record the row and move forward chronologically.

No random train/test split, no shuffling, no future normalisation, no
future-derived feature leakage. The actual future candle is never supplied to
Kronos input.

### Holdout window

- Warm-up/context region: the `context_length` candles before the first target
  (default 512).
- Default evaluation window: the most recent `max_predictions` (default 1000)
  closed target candles; exact `evaluation_start_ms` / `evaluation_end_ms` /
  `warmup_*` timestamps are computed and reported per run.
- `--start` / `--end` select an explicit reproducible window (ISO-8601 UTC or
  epoch ms).

### Direction definition

`predicted_return` and `actual_return` are both measured against the same
baseline (the context's last close). A return is **flat** when its absolute
value `<= direction_threshold` (default `0.0005`, i.e. 0.05%).

- `directional_correct` = predicted direction is non-flat **and** equals the
  actual direction.
- `directional_accuracy` = correct / (candles with non-flat actual return);
  near-zero actual candles are excluded from all direction denominators.
- `bullish_accuracy` / `bearish_accuracy` are conditioned on actual up / down.

### Metrics computed

Price: MAE close / RMSE close / MAE open / MAE high / MAE low / MAE volume /
MAPE (only where `|actual_close| > 0`). Direction: directional accuracy,
bullish/bearish accuracy, counts of positive/negative/near-zero actual returns.
Return: mean predicted/actual return, return MAE/RMSE, Pearson correlation
(undefined → `null`). Always reported: predictions, skipped, skip reasons.

### Skip policy (never fabricated)

`context_invalid` (gap/duplicate/NaN/inf/OHLC in context), `target_gap`
(gap between context and target or within target window), `invalid_target`
(invalid actual candle), `empty_prediction` (model returned no steps).

### Determinism

Evaluation uses the deterministic Phase 3 recipe by default. A repeatability
test asserts identical input → identical output. `--no-deterministic` is a
loud opt-out (prints a warning); random sampling is never used silently.

### Efficiency

Model + tokenizer are loaded once and reused; the DB is read once into memory;
predictions run sequentially (chronology preserved, no cross-timestamp
batching that could leak); inference uses the upstream `torch.no_grad` path.

### Verification status in this environment (Arena sandbox)

The evaluator logic was validated here with a deterministic test double
(identical interface to `KronosRealPredictor`; never presented as real output):

- future candle never in model input;
- removing future candles leaves the context identical;
- strictly forward movement through time;
- gap → skip (never fabricated);
- forming candle excluded;
- target read only after inference;
- deterministic repeatability;
- exact metric values;
- empty / zero-variance safety;
- window selection + documented timestamps;
- CLI requires the real model (no mock fallback).

Full suite: **80 passed, 3 skipped, 1 warning** (3 skips = real-weight tests
that require the model; warning = pre-existing `PytestReturnNotNoneWarning`).

### To run the real-data evaluation on the target machine

```bash
# BTC/USDT 1h (default window: most recent 1000 closed targets)
python -m kronos_trading.cli evaluate \
  --db data\db\kronos_trading_verified.db --symbol BTC/USDT --timeframe 1h

# ETH/USDT 1h
python -m kronos_trading.cli evaluate \
  --db data\db\kronos_trading_verified.db --symbol ETH/USDT --timeframe 1h

# Optional: 4h and 1d series
python -m kronos_trading.cli evaluate \
  --db data\db\kronos_trading_verified.db --symbol BTC/USDT --timeframe 4h
python -m kronos_trading.cli evaluate \
  --db data\db\kronos_trading_verified.db --symbol ETH/USDT --timeframe 1d

# Tests (real-weight tests un-skip when the model is present)
pytest -q
```

Results are printed and saved as machine-readable JSON under `data/eval/`.

## Testing

- Full suite: `80 passed, 3 skipped, 1 warning`
- Phase 2 audit: `7 passed`
- Offline system: `3 passed`
- Historical-range regression: `14 passed`
- Phase 3: `22 passed, 2 skipped`
- Phase 4: `16 passed, 1 skipped` (skip = real-weight test)

## Safety

- Trading mode: PAPER
- Live trading: DISABLED (`live_trading_enabled=false`)
- Live exchange order creation: disabled (no CCXT in `kronos_trading/`)
- No API credentials committed
- Kronos upstream source untouched (pinned `67b630e67f6a`)
- Verified SQLite database and Supabase data unchanged (read-only access)

## Next Phase

Phase 5+ (strategy/signals, paper research) remain blocked until the Phase 4
chronological evaluation has actually run on the verified dataset and its
metrics are recorded (the commands above). Phase 4 evaluates the model only —
no strategy, no trading thresholds, no profitability claims.
