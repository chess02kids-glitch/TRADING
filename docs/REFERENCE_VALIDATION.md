# Kronos Reference-Pipeline Validation

**Date:** 2026-08-14 · **Upstream commit:** `67b630e67f6a18c9e9be918d9b4337c960db1e9a`

This document validates our inference pipeline (`ModelManager` /
`KronosRealPredictor` / `to_kronos_frame`) against the upstream Kronos
reference path before running any further research experiments.

---

## 1. The upstream reference path

The authoritative reference is the upstream regression test and its fixture
generator (both pinned at `67b630e67f6a`):

- `Kronos/tests/test_kronos_regression.py`
- `Kronos/tests/data/generate_regression_output.py`
- fixtures `Kronos/tests/data/regression_input.csv`,
  `regression_output_{512,256}.csv`

Reference recipe (extracted verbatim from the test file):

| parameter | value |
|---|---|
| model | `NeoQuasar/Kronos-small` |
| tokenizer | `NeoQuasar/Kronos-Tokenizer-base` |
| model revision | `901c26c1332695a2a8f243eb2f37243a37bea320` |
| tokenizer revision | `0e0117387f39004a9016484a186a908917e22426` |
| feature columns | `[open, high, low, close, volume, amount]` |
| context | 512 · pred_len 8 · max_context 512 |
| device | cpu · seed 123 |
| mode | `tokenizer.eval(); model.eval(); torch.no_grad()` |
| prediction call | `predict(df, x_timestamp, y_timestamp, pred_len=8, T=1.0, top_k=1, top_p=1.0, verbose=False, sample_count=1)` |

The official example (`Kronos/examples/prediction_example.py`) differs from the
regression test in two ways: it does **not** call `eval()` and it uses
**stochastic nucleus sampling** (`top_p=0.9`, no `top_k`). The regression test
is the deterministic, revision-pinned reference, so it is the one we validate
against.

## 2. Reproducibility of the reference

`upstream_reference_constants()` reads the constants from the pinned test file
and cross-checks our pipeline's constants. Result: **all in sync**

- `model_revision`: ✅
- `tokenizer_revision`: ✅
- `max_context` (512): ✅
- `pred_len` (8): ✅
- `seed` (123): ✅

## 3. Contract-level comparison (executed offline)

Both paths were fed through the **real upstream `KronosPredictor.predict()`**
with a stubbed `generate()` (so 100% of upstream preprocessing executed) and
the resulting tensors compared element-wise on the same 512-row reference
input.

| item | reference | ours | match |
|---|---|---|---|
| input df shape | (512, 6) | (512, 5) | — |
| input columns | open,high,low,close,volume,**amount** | open,high,low,close,volume | amount omitted (derived) |
| normalized `x` tensor | (1, 512, 6) | (1, 512, 6) | ✅ |
| `x_stamp` (minute/hour/weekday/day/month) | (1, 512, 5) | (1, 512, 5) | ✅ identical |
| `y_stamp` | (1, 8, 5) | (1, 8, 5) | ✅ identical |
| API kwargs (T, top_k, top_p, sample_count, verbose) | 1.0, 1, 1.0, 1, False | same | ✅ identical |
| normalization | per-sequence z-score + clip[-5,5] (upstream) | same (upstream) | ✅ |
| timestamp handling | naive datetimes | tz-aware UTC | ✅ identical features |

**Per-channel max absolute difference** (on the normalized `x` tensor):

| channel | max abs diff |
|---|---|
| open | 0.0 |
| high | 0.0 |
| low | 0.0 |
| close | 0.0 |
| volume | 0.0 |
| **amount** | 0.008851 (differs) |

The OHLCV channels, timestamps, normalization and API call are **bit-identical**
to the reference.

## 4. Documented differences

### 4.1 `amount` feature (data limitation, not a code bug)

The reference feeds a real **turnover** column (`amount`). Our verified Binance
dataset has only `volume` (base-asset volume), so `to_kronos_frame` emits no
`amount` and upstream derives `amount = volume × mean(open, high, low, close)`
— upstream's own documented fallback for amount-less data
(`Kronos/model/kronos.py:531-532`).

On the reference fixture (A-share 5-minute bars, where `amount` is turnover in
yuan and `volume` is in lots), the derived proxy differs from real turnover by
~99% in relative terms (mean abs diff ≈ 1.22M). For crypto this proxy is closer
(quote volume ≈ volume × price) but still not the true turnover the model was
trained with.

**Implication:** the model was trained with real turnover as the 6th feature;
our data supplies a proxy. This is a candidate contributing factor to the
Phase 4 result and **cannot be fixed without changing the verified dataset**
(which is out of scope). It is flagged for a future data phase.

### 4.2 Model/tokenizer revision defaults (mismatch → FIXED)

`ModelManager` and the CLI defaulted to `revision=None` (latest), which does
**not** guarantee the documented pinned revisions. This was a genuine mismatch
against the reference. **Fixed:** `ModelManager`, the CLI defaults, and
`kronos_trading/__init__.py` now pin `901c26c1…` / `0e011738…` by default.

### 4.3 Decoding: probabilistic sampling vs deterministic argmax

Upstream's *intended/default* decoding is **probabilistic nucleus sampling**
(`top_k=0`, `top_p=0.9`, `sample_logits=True → torch.multinomial`), per the
official example. The deterministic **argmax** (`top_k=1`, `top_p=1.0`) is the
upstream *regression-test* variant used for reproducible fixtures. Our
evaluation uses the argmax recipe, which matches the regression test, not the
example default. **No production/default configuration was changed.**

## 5. Output-level comparison

Comparing predicted OHLCV against `regression_output_512.csv` requires the
model weights, which are **not available in this environment** (Hugging Face
egress is blocked and there is no local cache). This comparison is implemented
(`run_output_comparison`) and runs automatically on the target machine via
`python -m kronos_trading.cli validate-reference`.

## 6. Verdict

**B — Pipeline mismatch found (fixed).**

- Mismatch 1 (fixed): model/tokenizer revision defaults were unpinned.
- Mismatch 2 (documented, unfixable without data changes): the `amount`
  feature is an upstream-derived proxy rather than real turnover.

The inference contract (columns, timestamps, normalization, prediction API,
decoding recipe) otherwise matches the upstream reference bit-for-bit on the
OHLCV channels. Output-level equality remains to be demonstrated on the target
machine.

**Per task 7:** Phase 3 real-inference checks and all previous
prediction-evaluation results must be revalidated with the now-pinned
revisions before they are treated as final.
