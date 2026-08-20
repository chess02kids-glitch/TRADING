# Volatility / Range Research — Methodology (Phase 5b)

**Date:** 2026-08-14 · **Upstream:** Kronos pinned `67b630e67f6a`

This document records the *a priori* methodology for the Phase 5b experiment:
"does Kronos have genuine volatility/range forecasting skill, or is the Phase 5
range-MAE advantage over the weak persistence baseline an artifact?"

All constants below are FIXED before any results were looked at. Nothing is
tuned, cherry-picked, or optimized.

---

## 1. Baseline family (all past-only, fixed)

Input: the sequence of closed-candle ranges `range_t = high_t − low_t` strictly
before the prediction timestamp.

| ID | Baseline | Definition | Fixed constants |
|---|---|---|---|
| A | previous range | `range_{t-1}` | — |
| B | rolling mean range | mean of last `w` closed ranges | `w ∈ {5, 22}` |
| C | EWMA range | `e_t = α·range_t + (1−α)·e_{t-1}`, seeded on the first closed range | `span = 22`, `α = 2/(span+1) = 2/23` |
| D | HAR-style range | `β0 + β1·range_{t-1} + β2·mean5 + β3·mean22` fitted by OLS | expanding past-only window, refit per step, min 24 rows |

Baselines B/C/D are computed from the same validated closed-candle context as
the Kronos prediction, so they share identical timestamps and never see the
future. HAR is undefined (excluded) when the context has fewer than
`22 + 24` bars.

## 2. Targets

- **Raw range** (frozen, unchanged): `high_future − low_future`.
- **Normalized range** (new, scale-invariant):
  `(high_future − low_future) / close_current`, where `close_current` is the
  context's last closed close. Predicted and actual ranges use the same
  denominator, so this is a dimensionless percent-range.

## 3. Evaluation windows

Fixed chronological windows `older / middle / recent` (non-overlapping), on
BTC/USDT and ETH/USDT at 1h / 4h / 1d. Daily windows (~72–73 samples) are
low-power supplementary evidence and are **excluded** from the success gate;
only 1h and 4h (12 windows) decide the verdict. Windows with < 30 samples are
flagged `low_power` and excluded from the gate.

## 4. Error analysis

Per system (Kronos + baselines): MAE, RMSE, normalized MAE/RMSE, Spearman and
Pearson correlation, bias `mean(pred − actual)`, mean/std of predictions and
actuals, **dispersion ratio** `std(pred)/std(actual)`, and **bias ratio**
`mean(pred)/mean(actual)`.

Shrinkage question: *is Kronos's MAE advantage explained by lower forecast
variance / reduced extremes?* Evidence = dispersion ratio and bias ratio.

## 5. Regimes (past-only, fixed)

The regime measure is a rolling 22-bar mean range. Its value at the last closed
bar is compared against the **1/3 and 2/3 quantiles** of the measure's
expanding past-only series → `low / medium / high`. Regimes are assigned per
prediction from the context only; thresholds are never tuned.

## 6. Statistical tests (paired on identical timestamps)

- **Diebold-Mariano** (two-sided) with Newey-West HAC variance — appropriate
  for serially correlated forecast errors. Fixed lag rule
  `4·(n/100)^(2/9)`.
- **Circular block bootstrap** 95% CI on paired error differences — fixed block
  length `n^(1/3)`.
- **Wilcoxon signed-rank** — nonparametric robustness check (assumes
  exchangeability; noted for completeness).
- Multiple comparisons: raw p-values reported; Bonferroni note (two primary DM
  tests → α = 0.025).

## 7. Success gate (all 8 must hold for verdict A)

1. Kronos beats previous-range persistence on raw-range MAE in > half of
   non-daily windows.
2. Kronos beats EWMA **or** HAR on raw-range MAE in > half of non-daily windows.
3. The advantage vs a serious baseline appears in ≥ 2 of 4 series.
4. The effect appears in ≥ 2 of 3 chronological windows.
5. The normalized-range MAE advantage survives in > half of windows.
6. At least one statistically defensible paired test (pooled DM) supports the
   improvement.
7. The improvement is not explained solely by shrinkage
   (pooled dispersion ratio ≥ 0.7, fixed a priori).
8. The advantage is present in ≥ 2 of 3 regimes (not single-regime).

## 8. Verdict mapping

- **A — genuine volatility signal**: all 8 criteria hold.
- **B — weak / ambiguous**: beats the weak persistence baseline but fails at
  least one stronger criterion.
- **C — false positive / no useful signal**: does not beat persistence, or the
  advantage disappears under normalized / regime / statistical analysis.

## 9. Decision (post-experiment)

- **A** → recommend one follow-up volatility-only experiment.
- **B** → one carefully chosen follow-up, or abandon the zero-shot volatility
  hypothesis.
- **C** → STOP the zero-shot Kronos volatility path; move to the non-Kronos
  diagnostic baseline stage.

No trading strategy, no tuning, no profitability claims.
