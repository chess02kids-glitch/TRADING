# Next Research Branch — New Information, Not New Architectures

**Date:** 2026-08-14 · **Status:** DESIGNED → IMPLEMENTED (Phase 7 cross-asset experiment)

> Update: the cross-asset experiment designed below has been implemented as
> Phase 7 (`kronos_trading/cross_asset.py`, CLI `cross-asset`). The precise
> feature definitions, C1–C7 gate and STOP conditions are specified in
> `docs/CROSS_ASSET_METHODOLOGY.md`. The real-data run is pending on the
> verified dataset.

The OHLCV-only model-complexity branch is CLOSED (see
`docs/FINAL_OHLCV_RESEARCH_CONCLUSION.md`). The next scientific question is
**not** "can another model beat HAR?" but:

> What information beyond single-asset OHLCV is needed to improve volatility
> forecasts beyond HAR?

---

## 1. Candidate information classes (ranked)

Ranking dimensions: expected information gain, scientific plausibility, data
accessibility, look-ahead risk, implementation cost, relevance to the
next-candle-range target, and ability to run strict OOS.

| Rank | Class | Information gain | Plausibility | Accessibility | Look-ahead risk | Cost | Relevance to volatility | Strict OOS |
|---|---|---|---|---|---|---|---|---|
| 1 | **A. Cross-asset / market-wide crypto** | High | High (volatility is contagious across crypto; BTC/ETH strongly correlated) | **Already available** (ETH is in the verified DB) | Low (align other asset at time t) | Low | High (common volatility factor) | Fully supported |
| 2 | B. Derivatives / funding / open interest | High (funding predicts short-horizon vol) | High | New futures exchange data required | Medium (async between spot/futures) | Medium–High | High | Supported with care |
| 3 | C. On-chain variables | Medium | Medium | New infrastructure (indexers) | Medium | High | Medium (lower frequency vs 1h/4h) | Supported |
| 4 | D. Macro variables | Medium | Medium (mostly at 1d horizon) | Public but low-frequency | Medium (forward-fill/release timing) | Low–Medium | Low–Medium for 1h/4h | Marginal |
| 5 | E. Sentiment / news | Medium–High | Medium | High infra, noisy | **High** (publication timing) | High | Medium | Hard |

**Selection: A — cross-asset / market-wide crypto information.** It is the
cheapest scientifically clean extension: the second asset is already in the
verified dataset, requires no new data acquisition, has low asynchronous-leakage
risk, and directly tests the common-volatility-factor hypothesis.

## 2. Next research question

> Does information from other assets (the cross asset's realized volatility and
> range/return factors) improve volatility forecasting beyond single-asset HAR?

## 3. Exact experiment design (pre-registered, not yet implemented)

### Target
Normalized next-candle range `(high_{t+1} − low_{t+1}) / close_t` (unchanged).

### Baseline
The **frozen single-asset HAR** (see
`docs/FINAL_OHLCV_RESEARCH_CONCLUSION.md` Appendix A). Not retuned.

### New information (one explicit feature family)
For forecasting asset A at time t, augment HAR with the cross asset B's
past-only volatility/range/return factors:

- B's previous normalized range (`nr1_B`),
- B's trailing 22-bar mean normalized range,
- B's trailing 22-bar realized volatility (close-to-close),
- B's trailing 1-bar and 22-bar returns.

Exactly one feature family (cross-asset volatility/range/return). No indicators
beyond this family are added in the first experiment.

### Model (simplest suitable first)
**HAR + linear cross-asset extension** — the frozen HAR features plus the
cross-asset features, fitted with the same expanding past-only OLS and refit
cadence. A fixed small supervised model is used only if linearity is shown to be
inadequate; it is not used in the first experiment.

### Strict temporal alignment (mandatory)
For forecasting A's candle `t`:
- Cross-asset features are computed from B's candles with **close time ≤ t**
  (i.e., B's last closed candle strictly before A's target open time).
- Same-timeframe rule: use B's latest candle whose open time < t.
- If B has no closed candle at t (gap or B's candle is still forming), the
  prediction is skipped (recorded), never forward-filled.
- **No forward-fill policy**: a missing cross-asset value is a skip, not an
  imputation. This is chosen specifically to avoid hidden future information.
- Explicit tests: timestamp alignment, stale/missing handling, forward-fill
  prohibition, and a leakage test (perturbing B's future candles must not
  change forecasts at t).

### Evaluation windows
Identical frozen windows: BTC/ETH × 1h/4h × older/middle/recent = 12 primary
windows; daily supplementary and excluded from the gate.

### Statistics
Identical paired framework: Diebold-Mariano (Newey-West HAC), circular block
bootstrap, Wilcoxon signed-rank; identical timestamps; single primary
comparison (HAR+ vs HAR) → α = 0.05.

## 4. Success gate (the new information source passes ONLY if all hold)

1. HAR+cross-asset beats frozen HAR on normalized MAE.
2. The improvement appears across multiple primary windows.
3. It survives normalized **and** raw-range evaluation.
4. Paired statistical evidence supports the improvement.
5. It survives regime analysis (low/medium/high, past-only terciles).
6. No look-ahead / leakage (verified).
7. The benefit is not confined to one asset or one regime.

## 5. STOP condition

If HAR+cross-asset fails this gate, the cross-asset data family is closed —
do **not** keep adding more cross-asset features or move to a larger model on
the same family. The next step would then be the next-ranked information class
(derivatives/funding/open interest), evaluated under the same gate.

## 6. Implementation scope (IMPLEMENTED in Phase 7)

- `kronos_trading/cross_asset.py`: cross-asset feature builder with strict
  temporal alignment (`open time == T − step`, no forward-fill) +
  HAR-linear-extension expanding-OLS walk-forward evaluator.
- `kronos_trading/cli.py`: `cross-asset` subcommand
  (`data/eval/cross_asset_volatility_report.json`).
- `tests/test_phase7_cross_asset.py` (17 tests): timestamp alignment, no future
  information, no forward-fill, missing-history handling, deterministic
  coefficients/predictions, identical OOS timestamps, leakage counter,
  gate classification.
- `SYSTEM_STATUS.md`: Phase 7 recorded.

The real-data experiment remains pending execution on the verified dataset:
`python -m kronos_trading.cli cross-asset --db data\db\kronos_trading_verified.db
--assets BTC/USDT ETH/USDT --timeframes 1h 4h 1d --window-size 1000`.
