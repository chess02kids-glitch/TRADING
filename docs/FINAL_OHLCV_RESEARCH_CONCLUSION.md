# Final OHLCV-Only Forecasting Research Conclusion

**Date:** 2026-08-14 · **Status:** FINAL (frozen) · **Branch:** `arena/01a00001-trading`

This document is the frozen scientific record of the OHLCV-only forecasting
research (Phases 4–6). It is precise and descriptive; it makes no motivational
or profitability claims.

---

## 1. Research question

Can any model in the evaluated family (zero-shot Kronos-small, or a fixed
supervised tree model) extract forecasting information from single-asset OHLCV
data beyond simple classical models, for (a) absolute price / return and
(b) candle-range volatility?

## 2. Dataset

- Verified SQLite: `data/db/kronos_trading_verified.db` (not committed).
- Assets: BTC/USDT, ETH/USDT. Timeframes: 1h, 4h, 1d. ≈ 45,500 OHLCV rows.
- Primary decision windows: BTC/ETH × 1h/4h × older/middle/recent = 12 windows.
- Daily windows (~73 samples) are low-power and supplementary; they never
  determined the primary conclusion.

## 3. Data validation

- Phase 2 audit PASS; SQLite ↔ Supabase canonical OHLCV parity PASS
  (SHA-256 `7e362e05…c03a527`).
- No gaps filled, no fabricated candles, no data modification.
- Known limitation: the verified Binance dataset has no true turnover
  (`amount`). The upstream Kronos fallback derives `amount = volume × mean
  price`. This was documented and does not invalidate the core OHLCV
  conclusions (all price/range channels are exact).

## 4. Kronos configuration (frozen)

- Model: `NeoQuasar/Kronos-small`, revision
  `901c26c1332695a2a8f243eb2f37243a37bea320`.
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-base`, revision
  `0e0117387f39004a9016484a186a908917e22426`.
- Upstream commit pinned `67b630e67f6a18c9e9be918d9b4337c960db1e9a`.
- Context 512; deterministic argmax (`seed=0, top_k=1, top_p=1.0,
  sample_count=1`); CUDA `cuda:0` on RTX 3050.

## 5. Evaluation methodology

- Strict chronological walk; only closed candles strictly before each
  prediction timestamp are used; the currently-forming candle is excluded.
- No shuffling, no random train/test split, no future normalization, no
  look-ahead.
- Windows fixed a priori (older/middle/recent); identical timestamps across
  all compared systems.
- Paired statistics: Diebold-Mariano (Newey-West HAC), circular block
  bootstrap, Wilcoxon signed-rank; Spearman/Pearson where defined; Bonferroni
  correction for multiple primary comparisons.

## 6. Phase 4 result (price / return)

- Kronos-small lost to persistence on close MAE/RMSE/MAPE and return MAE/RMSE
  across all 18 robustness windows.
- Return correlation ≈ 0 (BTC +0.0268, ETH −0.0068); directional accuracy not
  robust (≤ ~41%, vs persistence's 0% by policy — not a meaningful benchmark).
- **Verdict C:** Kronos does not add predictive value beyond persistence for
  absolute price / return.

## 7. Phase 5 result (target reformulation)

- Multi-period return (horizon=4): failed to beat persistence in well-powered
  1h/4h windows.
- Volatility-normalized return: failed to beat persistence.
- Range/volatility: initially beat previous-range persistence, but this was
  re-examined in Phase 5b (see §8).

## 8. Volatility / HAR result (Phases 5b, 5c)

- Volatility/range forecasting **does contain robust predictable structure**.
- The HAR-style model (`β0 + β1·range_{t-1} + β2·mean5 + β3·mean22`,
  expanding past-only OLS) robustly beat previous-range, EWMA and rolling
  baselines; it survived normalized-range and regime-tracking checks.
- Corrected classical benchmark verdict: **A — classical volatility
  predictability established.**
- HAR vs previous-range pooled DM: statistic ≈ −10.63, p ≈ 2.15e-26.

## 9. Kronos-vs-HAR result

- Kronos-small did **not** demonstrate incremental value over HAR.
- Pooled DM (Kronos vs HAR): p ≈ 0.00587, winner = HAR.
- Kronos's apparent raw-range advantage was substantially explained by forecast
  shrinkage (dispersion ratios often < 0.7), not volatility-state forecasting.

## 10. LightGBM-vs-HAR result (Phase 6)

- Target: normalized next-candle range `(high_{t+1} − low_{t+1}) / close_t`.
- Model: fixed LightGBM (300 estimators, learning rate 0.05, 31 leaves, fixed
  seeds, CPU), 33 past-only OHLCV features, expanding walk-forward, retrain
  every 100 predictions, `leaks = 0`.
- Pooled DM (ML vs HAR): statistic ≈ −0.1186, p ≈ 0.9056 → **no statistical
  evidence that ML beats HAR**.
- Success gate: **C — ML fails to robustly beat HAR.**
- ML won some BTC 4h windows but generally failed on ETH; gains were unstable
  across assets/windows; raw-range improvement did not robustly persist.
- Feature importance was dominated by simple volatility-state variables
  (`nr_1`, `hour`, `nr_mean_5`, `park_5`, `dow`, `dist_ma22`), i.e. ML was
  largely rediscovering structure already represented by HAR.

## 11. Statistical evidence (summary)

| Comparison | Pooled DM statistic | p-value | Winner |
|---|---|---|---|
| HAR vs previous-range | ≈ −10.63 | ≈ 2.15e-26 | HAR |
| Kronos vs HAR | — | ≈ 0.00587 | HAR |
| Kronos vs EWMA | — | ≈ 0.576 | none |
| ML (LightGBM) vs HAR | ≈ −0.1186 | ≈ 0.9056 | none |

## 12. What was rejected

1. Zero-shot Kronos-small absolute price / return forecasting.
2. Kronos multi-period return and volatility-normalized return.
3. Kronos as an incremental volatility forecaster over HAR.
4. A fixed supervised LightGBM model over single-asset OHLCV as an
   incremental volatility forecaster over HAR.

## 13. What was established

1. Crypto volatility/range forecasting contains robust predictable structure.
2. That structure is adequately captured by the classical HAR model on
   single-asset OHLCV data.
3. Adding model complexity (zero-shot foundation model, gradient-boosted
   trees) to the same OHLCV information set did not add demonstrated value.

## 14. Limitations

- Two assets (BTC, ETH), spot OHLCV only, three timeframes.
- No true turnover/amount column in the verified dataset.
- Daily windows are low-power and excluded from the primary decision.
- Results are specific to the frozen Kronos revision, deterministic decoding,
  and the fixed LightGBM configuration; they do not rule out future models or
  other data sources.

## 15. Final conclusion

On the verified BTC/ETH OHLCV dataset, crypto volatility/range contains robust
predictable structure, but that structure is adequately captured by a simple
classical HAR model. Zero-shot Kronos-small does not demonstrate incremental
volatility forecasting value, and a fixed supervised LightGBM model does not
robustly improve on HAR.

This does **not** mean "crypto volatility is impossible to predict". It means
"predictable OHLCV volatility structure exists, but current evidence does not
justify additional model complexity beyond HAR on this information set."

## 16. STOP conditions for the closed OHLCV-only branch

The OHLCV-only model-complexity branch is CLOSED. It may be reopened only if:

1. The frozen HAR benchmark is first reproduced exactly (same formula, windows,
   OLS procedure, target, normalization, OOS windows, statistics).
2. A concrete, pre-registered hypothesis identifies a specific deficiency of
   HAR that a specific new model class is expected to fix.
3. A new out-of-sample, leak-free evaluation is specified before fitting.
4. The new model beats the frozen HAR benchmark under the pre-registered gate
   (see the HAR frozen benchmark specification below).

Absent these conditions, no further model/architecture work is performed on
single-asset OHLCV-only data.

---

## Appendix A — HAR frozen benchmark specification

This is the champion baseline for all future research. It is not retuned.

- **Target (primary):** normalized next-candle range
  `y_t = (high_{t+1} − low_{t+1}) / close_t`.
- **Target (secondary):** raw range `high_{t+1} − low_{t+1}`.
- **Formula:**
  `forecast_t = β0 + β1·range_{t-1} + β2·mean(range_{t-5:t}) + β3·mean(range_{t-22:t})`,
  where `range_k = high_k − low_k`.
- **Windows:** 5 and 22 bars, fixed.
- **Fitting:** expanding past-only ordinary least squares; refit at each
  prediction step using only closed candles strictly before the target candle;
  minimum 24 training rows; forecasts may be negative (honest OLS output).
- **Normalization:** none for raw-range features; the normalized target divides
  by `close_t` (the last closed close) only.
- **OOS windows:** older / middle / recent, chronological, non-overlapping.
  Primary decision = BTC/ETH × 1h/4h = 12 windows; daily supplementary.
- **Statistics:** paired Diebold-Mariano (Newey-West HAC), circular block
  bootstrap 95% CI, Wilcoxon signed-rank; identical timestamps across systems;
  Bonferroni correction for multiple primary comparisons (α = 0.0125 for the
  four classical comparisons; α = 0.05 for a single ML-vs-HAR comparison).
- **Reference result:** HAR vs previous-range pooled DM statistic ≈ −10.63,
  p ≈ 2.15e-26. Any future model/data source must beat this frozen benchmark.

## Appendix B — Frozen artifacts

- `docs/phase4_baseline_experiment.json` (SHA-256 locked) — Phase 4 baseline.
- `docs/REFERENCE_VALIDATION.md` — upstream pipeline validation.
- `docs/VOLATILITY_RESEARCH_METHODOLOGY.md` — Phase 5b methodology.
- `docs/CLASSICAL_VOLATILITY_METHODOLOGY.md` — Phase 5c methodology + audit.
- `docs/ML_VS_HAR_METHODOLOGY.md` — Phase 6 methodology.
- Report JSONs on the target machine (gitignored, not committed):
  `data/eval/*.json`.
