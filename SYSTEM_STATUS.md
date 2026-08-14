# Kronos Trading System — Current Status

**Date:** 2026-08-14

## Overall Status

- Phase 1: PASS
- Phase 2: PASS
- Database verification: PASS
- Supabase migration: COMPLETE
- SQLite ↔ Supabase parity: PASS
- Phase 3: PASS — real Kronos inference verified on the RTX 3050
- Phase 4: COMPLETE — verdict **C**; Kronos did not beat persistence on price/
  return across the 18-window robustness evaluation
- Phase 5: COMPLETE — multi-period return and volatility-normalized return
  failed vs persistence; range/volatility re-examined in Phase 5b
- Reference validation: COMPLETE — revision defaults pinned; OHLCV
  preprocessing/timestamps/normalization match upstream and the official
  fixture comparison passes for the OHLCV channels; only `amount` differs
  (derived proxy, no true turnover in the verified data)
- Phase 5b (volatility research): **VERDICT B (weak/ambiguous)** — Kronos beats
  previous-range but not serious baselines (pooled DM vs EWMA p=0.576, vs HAR
  p=0.00587); part of the MAE edge is forecast shrinkage.
- Phase 5c (classical volatility benchmark): **audited** — c6 gate + DM
  winner-label bugs fixed; corrected verdict **A (classical volatility
  predictability established)**; Kronos adds no incremental value over HAR.
- Phase 6 (ML vs HAR): COMPLETE — verdict **C**; fixed LightGBM did not
  robustly beat HAR (pooled DM ≈ −0.1186, p ≈ 0.9056).
- **OHLCV-only model-complexity branch: CLOSED** — HAR is the frozen champion;
  next branch is NEW INFORMATION (cross-asset), see
  `docs/NEXT_RESEARCH_BRANCH.md`.
- Phase 7 (cross-asset information): **REAL-RUN VERIFIED — verdict B (weak /
  ambiguous)**. Corrected run (commit `ae3dd2f`): pooled cross-vs-HAR normalized
  DM statistic = 2.2262, p = 0.0260 (HAR wins); C1/C2/C3/C7 true, C4/C5/C6
  false. Cross-asset family **CLOSED** per pre-registered STOP rule.
- Phase 8 (derivatives positioning): IMPLEMENTED — funding (last settled),
  open-interest 22-bar log change, basis (mark/spot − 1) as a linear HAR
  extension vs frozen HAR, point-in-time alignment with staleness bounds and
  no forward-fill, pre-registered C1–C7 gate; real run pending derivative data
  availability on the target machine (`docs/DERIVATIVES_METHODOLOGY.md`)

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

### Status: COMPLETE — verdict C (see "Phase 4 verdict (final)" below)

The evaluator ran on the real verified dataset for BTC/USDT 1h and ETH/USDT 1h
with the naive baselines. The robustness/generalization phase (4h, 1d, three
chronological windows, and paired statistical tests) is fully implemented and
unit-tested but has **not yet been executed** on the target machine, so the
final generalization verdict is not yet determined.

### Real-data results so far (target machine)

**BTC/USDT 1h** (1000 predictions):

| metric | Kronos | persistence | previous-direction |
|---|---|---|---|
| MAE close | 164.0275 | **135.2848** | 200.1951 |
| RMSE close | 226.5340 | *(lower)* | *(higher)* |
| MAPE close | 0.002564 | *(lower)* | *(higher)* |
| directional accuracy | 37.86% | 0% (flat by policy) | **38.24%** |
| return correlation | 0.0268 | — | — |

**ETH/USDT 1h** (1000 predictions):

| metric | Kronos | persistence | previous-direction |
|---|---|---|---|
| MAE close | 6.5228 | **5.2629** | 7.8907 |
| RMSE close | 9.1811 | *(lower)* | *(higher)* |
| MAPE close | 0.003511 | *(lower)* | *(higher)* |
| directional accuracy | **41.03%** | 0% (flat by policy) | 38.93% |
| return correlation | −0.0068 | — | — |

**Reading (preliminary, 1h only):** Kronos loses to persistence on close
MAE/RMSE/MAPE and return-error metrics on both assets. Kronos beats
previous-direction on most ETH error/direction metrics but does not beat it on
BTC direction. The 1h evidence does **not** establish useful incremental
predictive value — but this is one timeframe; generalization to 4h/1d and
across windows is exactly what the robustness phase (below) is for.

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

### Naive baselines (for fair comparison)

Added `kronos_trading/baselines.py` with two deterministic, no-model baselines
computed on exactly the same prediction timestamps as Kronos:

1. **Persistence / random-walk** — `predicted_close = last observed close`,
   `predicted_return = 0`. A flat prediction never scores direction, so its
   directional accuracy is 0 by policy.
2. **Previous-direction** — `previous_return = close[-1] / close[-2] - 1`;
   predicts the same return (so `predicted_close = last_close × (1 +
   previous_return)`) and predicts direction using the **same** flatness
   threshold as Kronos (0.0005).

Both baselines use only candles strictly before the prediction timestamp;
open/high/low/volume are left undefined and excluded from the comparison.

The report is extended with:

- `baseline_results.persistence` / `baseline_results.previous_direction` — full
  metric dictionaries computed with the *same* `compute_metrics` definitions;
- `model_comparison` — per-metric deltas (`kronos − baseline`) and an explicit
  `*_winner` label (`kronos` / `baseline` / `tie` / `null` when undefined):
  - `kronos_vs_persistence`: `mae_close_delta`, `rmse_close_delta`,
    `directional_accuracy_delta` (+ MAPE, return MAE/RMSE, correlation).
  - `kronos_vs_previous_direction`: `directional_accuracy_delta`,
    `return_correlation_delta` (+ MAE/RMSE/MAPE/return deltas).
  - `prediction_count.same_timestamps` asserts all systems used identical
    prediction timestamps.
- No statistical significance test is performed; `*_winner` is descriptive only
  (documented in the report's `note` field).

`tests/test_phase4_baselines.py` — 11 tests proving: persistence uses only the
last close; previous-direction uses only the previous/current closed candles;
baselines never inspect future candles; identical timestamps across systems;
identical flat-direction threshold; persistence never scores direction; safe
empty/zero-variance handling; explicit deltas and winner labels.

### Robustness / generalization (multi-window + statistics)

Added to support the generalization phase **without changing** the model,
revision, context (512), deterministic recipe, threshold (0.0005), baselines,
or no-lookahead rules:

- `kronos_trading/evaluation.py`
  - `define_windows()` — fixed chronological, non-overlapping windows
    (`older` / `middle` / `recent`) placed purely as a function of the data,
    never selected on performance. When data volume is insufficient, each
    window is shrunk to `n_targets // 3` (reported via
    `reduced_due_to_volume`); with very few targets, fewer windows are
    produced (`windows_omitted`).
  - `PredictionEvaluator.evaluate_windows()` — runs the *same* evaluator over
    each window; the recent window is identical to the single `evaluate()` run.
  - Each report now also carries `window` and `statistical_comparison`.
- `kronos_trading/statistics_compare.py` — self-contained paired statistics
  (numpy only, deterministic seed):
  - paired absolute close error: mean/median diff, Cohen's d_z, percentile
    bootstrap 95% CI, and a Wilcoxon signed-rank test (normal approximation
    with tie correction; zero differences dropped; caveat noted for n < 20);
  - paired directional outcome: accuracy delta, bootstrap 95% CI, and
    McNemar's test (exact binomial for < 25 discordant pairs, else
    continuity-corrected chi-square);
  - every test is paired on identical `prediction_timestamp`s; statistical
    significance is explicitly labelled as *not* trading profitability.
- `kronos_trading/robustness.py` — `run_robustness()` + consolidated report:
  per series × window metrics, `baseline_results`, `model_comparison`, and
  `statistical_comparison`; a `summary` of where Kronos wins/loses each metric
  across windows (with a `consistent_across_windows` flag); and an
  `across_all_series` roll-up.
- CLI `robustness` — `python -m kronos_trading.cli robustness --db
  data\db\kronos_trading_verified.db --assets BTC/USDT ETH/USDT --timeframes
  1h 4h 1d --window-size 1000` loads the model once and produces
  `data/eval/robustness_report.json`.

`tests/test_phase4_robustness.py` — 20 tests: window placement/non-overlap,
volume-based window shrinking, few-target fallback, recent-window equivalence,
chronological disjointness, identical timestamps per system, paired-test
correctness (bootstrap/wilcoxon/McNemar), identical-observation alignment,
empty/zero-variance safety, and deterministic repeatability of windows + stats.

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

Full suite: **236 passed, 3 skipped, 1 warning** (3 skips = real-weight tests
that require the model; warning = pre-existing `PytestReturnNotNoneWarning`).

### To run the real-data robustness matrix on the target machine

```bash
# 4h + 1d + multi-window + paired statistics in one consolidated report.
# Uses the same model/revision/context/threshold/deterministic recipe as the
# 1h evaluation - nothing is tuned.
python -m kronos_trading.cli robustness \
  --db data\db\kronos_trading_verified.db \
  --assets BTC/USDT ETH/USDT --timeframes 1h 4h 1d --window-size 1000

# Single-series windowed evaluation (one symbol/timeframe)
python -m kronos_trading.cli evaluate \
  --db data\db\kronos_trading_verified.db --symbol BTC/USDT --timeframe 4h

# Tests (real-weight tests un-skip when the model is present)
pytest -q
```

Results are printed and saved as machine-readable JSON under `data/eval/`.

## Testing

- Full suite: `236 passed, 3 skipped, 1 warning`
- Phase 2 audit: `7 passed`
- Offline system: `3 passed`
- Historical-range regression: `14 passed`
- Phase 3: `22 passed, 2 skipped`
- Phase 4: `47 passed, 1 skipped` (skip = real-weight test)
- Phase 5: `16 passed` (research targets)
- Reference validation: `9 passed`
- Phase 5b (volatility research): `24 passed`
- Phase 5c (classical volatility): `13 passed`
- Phase 5c audit (DM labels + c6 gate): `11 passed`
- Phase 6 (ML vs HAR): `16 passed`
- Phase 7 (cross-asset): `18 passed`
- Phase 8 (derivatives): `18 passed`

## Safety

- Trading mode: PAPER
- Live trading: DISABLED (`live_trading_enabled=false`)
- Live exchange order creation: disabled (no CCXT in `kronos_trading/`)
- No API credentials committed
- Kronos upstream source untouched (pinned `67b630e67f6a`)
- Verified SQLite database and Supabase data unchanged (read-only access)

## Phase 4 verdict (final)

**C. Robust evidence Kronos does NOT add predictive value beyond simple
baselines** — for the original OHLCV close target (horizon=1).

The robustness evaluation completed on the real verified dataset (6 series × 3
windows = 18 evaluations, same model/revision/context/threshold/deterministic
recipe/no-lookahead/baselines):

- Kronos lost to persistence on **all 18 windows** for close MAE, close RMSE,
  MAPE, return MAE and return RMSE.
- Kronos beat persistence on directional accuracy in all 18 windows, but
  persistence predicts zero return, so this is not a meaningful directional
  benchmark.
- Against previous-direction, Kronos won 16/18 windows on close MAE/RMSE/MAPE
  and return MAE/RMSE, but directional accuracy was mixed.
- 1h/4h patterns are stable across older/middle/recent windows; daily windows
  (~73 samples/window) are lower-confidence.

Conclusion: the present Kronos-small configuration has **not** demonstrated
incremental price-prediction value over persistence. This is the frozen
baseline experiment (`docs/phase4_baseline_experiment.json`, SHA-256 locked).

---

## Phase 5 — Model/Target Research (does the poor result come from the TARGET?)

### Status: COMPLETE — multi-period return and volatility-normalized return failed; range → Phase 5b

Phase 5 investigates whether Phase 4's negative result is a property of the
*target formulation* (absolute OHLC close, horizon=1) rather than of Kronos
itself. It changes only the derived target — never the model, revision,
tokenizer, context (512), deterministic recipe, threshold (0.0005), baselines,
or no-lookahead rules. This is research, not optimization; no trading strategy
is built.

### Architecture check (upstream facts, cited)

Kronos is an autoregressive sequence-to-sequence forecaster of the
6-dimensional (open, high, low, close, volume, amount) vector
(`Kronos/model/kronos.py`: `price_cols` at :489, z-scoring/clip at :544-547,
OHLCV reconstruction at :556-558, autoregressive generation at :389). It has no
classification head. Therefore target reformulations must preserve the
raw-OHLCV-in / OHLCV-out contract, and a direction-classification target
(candidate D) is rejected as an unjustified architectural change; direction is
derived from predicted returns instead.

### Selected targets (2–3 defensible, non-redundant)

1. **Multi-period return (candidate E, horizon=4)** — `predicted_close[T+4]/close[T]-1`
   vs the actual 4-candle return. Justified: Kronos is multi-step
   autoregressive, but the frozen experiment used only horizon=1 where 1h
   returns are noise-dominated. Contract preserved (native `pred_len`).
   Reversible to price. Baselines: persistence (0) and a horizon-aware
   previous-direction (`close[-1]/close[-1-4]-1`).
2. **Next-candle range/volatility (candidate F, horizon=1)** —
   `predicted_high - predicted_low` vs actual range. Justified: high/low are
   native Kronos outputs the frozen experiment never scored; isolates
   volatility structure from direction/level. Baseline: persistence (last
   observed range); previous-direction is N/A for a non-directional target.
3. **Volatility-normalized return (candidate B, horizon=1)** — next-candle
   return divided by a past-only scale `std(context returns) × √horizon`.
   Justified: crypto returns are heteroskedastic; normalizing yields a more
   stationary target and down-weights high-volatility regimes. Contract
   preserved (normalization applied outside the model on the derived target).
   Reversible. Baselines: persistence (0) and previous-direction, both on the
   same scale.

Candidates A (next-period return) and C (residual over persistence) are
subsumed by the frozen experiment's return metrics up to a price scale and were
therefore not duplicated as separate targets.

### Fair experiment design

Every target reuses the identical chronological windows (older/middle/recent),
identical closed-candle contexts, identical timestamps, and identical naive
baselines as the frozen experiment; the model is loaded once. Success is NOT
"any metric improved": a target is promising only if it is justified, beats the
relevant naive baseline on untouched chronological data across more than one
window, and does not depend on tuning.

### Implementation

- `kronos_trading/research_targets.py` — frozen-baseline loader + SHA-256 lock,
  `ARCHITECTURE_CHECK`, `TargetSpec` + `TARGET_SPECS` for the three targets,
  `compute_target_metrics()` (derives each target's metrics/baselines/comparison
  from the same rows), and `run_research_experiment()` (reuses the base
  horizon=1 pass for range + normalized return and a single extra horizon=4
  pass for multi-period return).
- `docs/phase4_baseline_experiment.json` — the immutable frozen Phase 4 record.
- CLI `research-targets` — `python -m kronos_trading.cli research-targets --db
  data\db\kronos_trading_verified.db --assets BTC/USDT ETH/USDT --timeframes
  1h 4h 1d --window-size 1000` → `data/eval/research_targets_report.json`.
- `tests/test_phase5_research_targets.py` — 16 tests: frozen hash lock,
  architecture check, target-spec completeness, no-lookahead per target
  (multi-period horizon-aware baseline, range persistence, normalized-return
  scale + zero-vol skip), identical timestamps, empty/zero-variance safety,
  deterministic repeatability, and report structure.

### To run on the target machine

```bash
python -m kronos_trading.cli research-targets \
  --db data\db\kronos_trading_verified.db \
  --assets BTC/USDT ETH/USDT --timeframes 1h 4h 1d --window-size 1000

pytest -q
```

## Phase 5 verdict (final)

The two return-formulation targets (multi-period return, volatility-normalized
return) did **not** robustly beat persistence in the well-powered 1h/4h
windows. The range/volatility target was the only promising branch and was
re-examined under stronger baselines in Phase 5b, where the apparent Kronos
advantage was shown to be largely forecast shrinkage.

---

## Reference-pipeline validation (before further research)

Strict validation of our pipeline against the upstream Kronos reference
(regression test + fixtures at `67b630e67f6a`). See
`docs/REFERENCE_VALIDATION.md` and `kronos_trading/reference_validation.py`
(CLI: `python -m kronos_trading.cli validate-reference`).

**Verdict: B — pipeline mismatch found (fixed).**

Findings:

1. **FIXED** — model/tokenizer revision defaults were unpinned; `ModelManager`
   and the CLI now pin `901c26c1…` / `0e011738…` by default.
2. **Documented (unfixable without data changes)** — the `amount` feature is an
   upstream-derived proxy (`volume × mean(price)`) because the verified Binance
   dataset has no turnover column; the reference feeds real turnover.
3. Everything else — OHLCV columns, timestamp features, per-sequence
   z-score + clip[-5,5] normalization, prediction API, and the deterministic
   argmax recipe (top_k=1/top_p=1.0 + eval) — matches the reference
   **bit-for-bit** on the OHLCV channels (verified offline with the real
   upstream `predict()` preprocessing).
4. **Decoding:** upstream's intended/default decoding is **probabilistic
   nucleus sampling** (`top_k=0`, `top_p=0.9`, `torch.multinomial`); the
   deterministic argmax is the upstream regression-test variant. No production
   default was changed.
5. Output-level equality (vs `regression_output_512.csv`): verified on the
   target machine — the official fixture comparison passes for the OHLCV
   channels; only `amount` differs (derived proxy).

---

## Phase 5b — Volatility / Range Research (is the range advantage genuine?)

### Status: EXECUTED — verdict B (weak/ambiguous volatility signal)

The Phase 5 raw-range target showed Kronos beating the weak "previous range"
persistence baseline in all 12 non-daily windows. This experiment tests whether
that advantage survives **serious** volatility baselines, price-level
normalization, shrinkage analysis, regime analysis, and paired statistical
tests. Full methodology (all constants fixed a priori): see
`docs/VOLATILITY_RESEARCH_METHODOLOGY.md`.

### What was added

- `kronos_trading/volatility_baselines.py` — fixed, past-only baseline family:
  A previous-range · B rolling-mean range (5 & 22) · C EWMA range (span=22) ·
  D HAR-style range (expanding past-only OLS, min 24 rows) + past-only
  tercile regime assignment.
- `kronos_trading/statistics_compare.py` — added Spearman rank correlation,
  Diebold-Mariano (Newey-West HAC, appropriate for serially correlated
  forecast errors), and circular block bootstrap.
- `kronos_trading/evaluation.py` — additive `VolatilityRow` per prediction
  (computed from the same closed-candle context; no extra inference, no
  future data).
- `kronos_trading/volatility_research.py` — full error analysis (MAE/RMSE,
  normalized MAE/RMSE, Spearman/Pearson, bias, dispersion & bias ratios),
  per-regime analysis, paired statistical tests, an 8-criterion pre-registered
  success gate, and the A/B/C verdict.
- CLI `volatility-research` →
  `data/eval/volatility_research_report.json`.
- `tests/test_volatility_research.py` — 24 tests (no-lookahead, past-only HAR
  fitting, identical timestamps, determinism, fixed baseline definitions,
  normalized-target correctness, regime assignment, statistical alignment,
  empty/short-window safety, gate classification A/B/C).

### Success gate (fixed a priori; all 8 must hold for verdict A)

1. beats previous-range on raw-range MAE · 2. beats EWMA or HAR · 3. ≥2 of 4
series · 4. ≥2 of 3 windows · 5. survives normalization · 6. pooled DM supports
improvement · 7. not solely shrinkage (dispersion ≥ 0.7) · 8. ≥2 of 3 regimes.
Verdict: A = genuine signal · B = weak/ambiguous · C = false positive.

### Verification status in this environment

Validated offline with a deterministic test double: the gate correctly flags a
**shrunken** constant-range predictor as **B** (beats baselines on MAE but
`dispersion_ratio = 0`), proving the shrinkage diagnostic works. The real
experiment then ran on the target machine (result below).

### REAL RESULT (executed on the verified dataset, target machine)

**VERDICT = B — weak / ambiguous volatility signal.**

Key findings (from `data/eval/volatility_research_report.json`, not committed):

- Kronos beats previous-range persistence.
- Kronos does **not** robustly beat serious volatility baselines.
- Pooled DM Kronos vs EWMA: `p = 0.576` (no evidence Kronos is better).
- Pooled DM Kronos vs HAR: `p = 0.00587` (HAR wins).
- Normalized-range criterion failed.
- Many Kronos dispersion ratios are substantially below 0.7 — Kronos often
  produces lower-dispersion forecasts than realized volatility, so part of the
  MAE improvement is forecast **shrinkage**, not volatility-state forecasting.
- HAR vs Kronos MAE (examples): BTC 1h older/middle/recent ≈ 153.3/206.1/116.3
  vs Kronos 167.8/212.2/118.2; BTC 4h ≈ 664.5/526.1/365.2 vs Kronos
  698.1/542.2/367.0 — HAR slightly but consistently lower.

Interpretation: classical volatility models (HAR/EWMA) already explain most of
the predictable structure; Kronos adds no incremental volatility information.
This motivated Phase 5c (classical benchmark as the primary design).

### To run on the target machine

```bash
python -m kronos_trading.cli volatility-research \
  --db data\db\kronos_trading_verified.db \
  --assets BTC/USDT ETH/USDT --timeframes 1h 4h 1d --window-size 1000
```

No tuning, no threshold optimization, no trading strategy, no profitability
claims. The frozen Phase 4/5 results are unchanged.

---

## Phase 5c — Classical Volatility Benchmark (HAR as primary; Kronos as challenger)

### Status: EXECUTED + AUDITED — corrected verdict A (classical volatility predictability established)

The highest-value question is now **not** "does Kronos win" but: *is the crypto
candle-range problem itself predictable, and does the classical HAR model
already capture that structure?* Full methodology (all constants fixed a
priori): `docs/CLASSICAL_VOLATILITY_METHODOLOGY.md`.

### What was added (additive; reuses Phase 5b machinery)

- `kronos_trading/classical_volatility.py`
  - classical benchmark matrix (previous-range / rolling-5 / rolling-22 /
    EWMA / HAR), with HAR as the primary model;
  - improvement % vs previous-range (raw + normalized);
  - HAR-vs-{prev,EWMA,rolling5,rolling22} paired DM / block-bootstrap /
    Wilcoxon, per window + pooled (supplementary);
  - HAR shrinkage/adequacy diagnostics: dispersion ratio, bias ratio, and a
    direct regime-tracking test (does HAR distinguish low/med/high regimes);
  - pre-registered 8-criterion A/B/C decision for *classical* predictability;
  - Kronos-vs-HAR incremental-value comparison (conservative: majority of
    windows AND significant pooled DM).
- CLI `classical-volatility` →
  `data/eval/classical_volatility_benchmark_report.json`.
- `tests/test_classical_volatility.py` — 13 tests (no-lookahead, past-only HAR
  fitting, determinism, identical timestamps, normalized target, baseline
  consistency, short-context safety, statistical alignment, improvement %,
  regime tracking, gate A/B/C).

### Pre-registered decision (classical predictability)

1. HAR beats previous-range (raw MAE) · 2. HAR beats EWMA · 3. ≥2 of 4 series ·
4. ≥2 of 3 windows · 5. survives normalization · 6. pooled DM p<0.0125
(Bonferroni) · 7. not purely shrinkage (regime tracking) · 8. >1 regime.
Verdict: A = predictability established · B = weak/ambiguous · C = none.

### IMPLEMENTATION AUDIT (corrected)

Two bugs were found and fixed after the first real run:

1. **Genuine gate bug** — criterion c6 was computed as
   `dm['winner'] == 'har'`, but `diebold_mariano` returned the hardcoded labels
   `'kronos'/'baseline'`, so c6 was ALWAYS False regardless of the p-value.
   Fixed: c6 is now derived label-independently from `p < 0.0125 AND
   mean_loss_diff < 0` (see `_c6_from_dm`).
2. **Naming/reporting bug** — the classical HAR-vs-* DM winner labels reported
   `'kronos'` instead of `'har'/'baseline'`. `diebold_mariano` now takes
   `a_name`/`b_name`, and the classical path passes `'har'`/`'baseline'`. The
   DM statistic, p-value and mean loss difference were always correct.

The corrected c6 uses the reported pooled HAR-vs-previous-range DM
(`stat = -10.6305, p = 2.15e-26, mean loss diff = -27.53`): `p < 0.0125` and
`mean loss diff < 0` → **c6 = true**. With c1–c5 and c7–c8 already true, all
eight criteria pass → **classical verdict A**. The report can be re-derived
without re-running inference via
`python -m kronos_trading.cli recompute-classical-gate --report <path> --summary`.

### To run on the target machine

```bash
python -m kronos_trading.cli classical-volatility \
  --db data\db\kronos_trading_verified.db \
  --assets BTC/USDT ETH/USDT --timeframes 1h 4h 1d --window-size 1000
```

---

## Phase 6 — Supervised ML vs HAR (does nonlinear ML add value?)

### Status: EXECUTED — verdict C (ML fails to robustly beat HAR)

The Kronos investigation is closed (Kronos adds nothing over HAR). The
remaining highest-value question: does a simple supervised ML model extract
incremental OHLCV volatility information beyond the HAR champion? Full
methodology (all constants fixed a priori): `docs/ML_VS_HAR_METHODOLOGY.md`.

### What was added

- `kronos_trading/ml_volatility.py`
  - 33 fixed, past-only OHLCV features (`build_feature_matrix`, left-aligned
    rolling windows + `.shift(1)` ⇒ no look-ahead by construction).
  - LightGBM (XGBoost fallback) with a fixed configuration and deterministic
    seeds.
  - `run_walk`: expanding-window walk-forward, retrain every 100 predictions,
    with a `leaks` counter and feature-importance history.
  - `run_ml_vs_har`: reuses the evaluator's window machinery (identical
    timestamps/baselines/regimes to the frozen classical benchmark) and emits
    the full report: per-window ML/baseline metrics (normalized + raw), ML-vs-HAR
    DM/bootstrap/Wilcoxon, regime analysis, shrinkage (dispersion/bias ratios +
    regime tracking), feature-importance stability, and the 8-criterion
    success gate.
- CLI `ml-vs-har` → `data/eval/ml_vs_har_volatility_report.json` (CPU only; no
  GPU/Kronos required).
- `tests/test_phase6_ml.py` — 16 tests (no-lookahead features, rolling-feature
  correctness, training-cutoff, walk-forward future-invariance, determinism,
  target alignment, identical timestamps, statistical alignment,
  empty/short-history safety, gate A/B/C + leakage blocking).

### Pre-registered success gate (all 8 must hold for verdict A)

1. ML beats HAR on normalized MAE in ≥2/3 windows for ≥2/4 series ·
2. same for normalized RMSE · 3. pooled DM (ML vs HAR) p<0.05 and ML wins ·
4. survives raw-range · 5. survives regime analysis · 6. not purely shrinkage ·
7. ≥2/3 windows (not one isolated period) · 8. no look-ahead.
Verdict: A = ML adds genuine incremental value · B = weak/ambiguous ·
C = ML fails (HAR already captures the structure).

### REAL RESULT (executed on the verified dataset, target machine)

**VERDICT = C — ML fails to robustly beat HAR.**

- Pooled DM (ML vs HAR, normalized): statistic ≈ −0.1186, p ≈ 0.9056 → no
  statistical evidence that ML beats HAR.
- ML beat HAR in some BTC 4h windows but generally failed on ETH; gains were
  unstable across assets/windows; raw-range improvement did not robustly
  persist; pooled statistical support is absent.
- Feature importance was dominated by simple volatility-state variables
  (`nr_1`, `hour`, `nr_mean_5`, `park_5`, `dow`, `dist_ma22`) — ML largely
  rediscovered structure already represented by HAR.
- `leaks = 0` (no look-ahead).

Interpretation: adding model complexity to the same single-asset OHLCV
information set does not add demonstrated value beyond HAR.

## Next Phase

The OHLCV-only model-complexity branch is **CLOSED** (see
`docs/FINAL_OHLCV_RESEARCH_CONCLUSION.md`). HAR is the frozen champion for all
future research.

The next research question tests **new information, not new architectures**:

> Does information from other assets improve volatility forecasting beyond
> single-asset HAR?

Recommended first data experiment: cross-asset / market-wide crypto features
(the other asset's trailing realized volatility, normalized range and return
factors) added to a HAR-linear extension, evaluated under the frozen HAR
benchmark with strict timestamp alignment and no forward-fill. Ranking, design,
gate and STOP conditions: `docs/NEXT_RESEARCH_BRANCH.md`.

**Phase 7 (REAL-RUN VERIFIED, verdict B):** the corrected cross-asset
experiment ran on the verified dataset. Result — pooled cross-vs-HAR normalized
DM statistic = 2.2262, p = 0.0260 (HAR wins); extreme-trimmed DM p = 0.0773
(HAR wins); low regime cross wins, medium/high HAR wins; `leaks = 0`; gate
C1/C2/C3/C7 true, C4/C5/C6 false → **B**. There is no robust statistical
evidence that BTC↔ETH information improves next-candle volatility beyond the
frozen single-asset HAR. **The cross-asset family is closed** (no more cross
features, no more assets, no nonlinear ML, no tuning).

**Phase 8 — DERIVATIVES / POSITIONING (implemented):** funding rate (last
settled, `funding_time ≤ t`), open-interest 22-bar log change, and basis
(`mark/spot − 1`), as a linear HAR extension under the frozen HAR benchmark and
the pre-registered C1–C7 gate (PASS = all 7 · B = C1 only · C = C1 false).
Data: public Binance USD-M market-data endpoints only (no key, no trading).
Point-in-time alignment with staleness bounds (missing required observation →
skip; no forward-fill; no interpolation). See
`docs/DERIVATIVES_METHODOLOGY.md`. **Real run pending** derivative data
availability on the target machine; STOP rule pre-registered (B/C closes the
derivatives family).

No trading strategy, no profitability claims, no hyperparameter search, and no
tuning on the OOS windows are performed. The frozen Phase 4/5/5b/5c reports are
unchanged.
