# Freqtrade Strategy Research Results

Generated: 2026-08-22
Data: BTC/USDT + ETH/USDT, 1h candles, 2022-05-25 → 2026-05-24
(Binance kline archive; proxy for KuCoin spot — see setup notes)
Backtest window: 20240101-20260101 (single pre-registered timerange)
Environment: freqtrade 2026.7, dry_run: true everywhere

---

## Candidates Evaluated

### Candidate 1: RSI + EMA 200 Trend Filter
`candidate_1.py` — class `RSITrendFilter` (user's Candidate 1)

**Hypothesis:** The RSI baseline failed because oversold entries during
downtrends fought the prevailing trend and hit the -5% stop. Restricting
entries to close > EMA 200 (standard institutional trend filter) should
cut the large losses while keeping most wins. Same ROI/stoploss structure
as the baseline so the entry filter is the only variable.

**Result: FAIL (eliminated)**

| Metric | Value |
|---|---|
| Trades | 38 (BTC 19 / ETH 19) |
| Win rate | 84.2% (32W / 0D / 6L) |
| Avg profit / trade | −0.20% |
| Total profit | −7.68 USDT (−0.77%) |
| Sharpe (closed) / Sortino | −0.10 / −0.12 |
| Profit factor | 0.71 |
| Max drawdown | 17.81 USDT (1.77%) |
| Mean profit p-value | 0.5357 |
| Best / worst trade | +1.23% / −5.19% |

The trend filter eliminated most losing trades (84.2% win rate) but also
most trades entirely (38 vs 414). The remaining edge is still negative:
even at 84.2% wins, small ROI-capped wins (+0.78% avg class) cannot
outweigh −5.19% stops. **Eliminated: Sharpe < 0, profit < 0, trades < 50.**

### Candidate 2: Bollinger Band Mean Reversion
`candidate_2.py` — class `BollingerReversion` (user's Candidate 2)

**Hypothesis:** Bollinger bands are volatility-normalized — the entry zone
scales with current volatility (same principle as HAR). Exiting at the
middle band gives a volatility-scaled target instead of the baseline's
fixed small ROI, addressing the win/loss asymmetry.

**Result: FAIL (eliminated)**

| Metric | Value |
|---|---|
| Trades | 785 (BTC 368 / ETH 417) |
| Win rate | 63.8% (501W / 0D / 284L) |
| Avg profit / trade | −0.13% |
| Total profit | −100.22 USDT (−10.02%) |
| Sharpe (closed) / Sortino | −1.08 / −1.23 |
| Profit factor | 0.86 |
| Max drawdown | 144.64 USDT (14.00%) |
| Mean profit p-value | 0.1396 |
| Best / worst trade | +8.39% / −5.19% |

Middle-band targets were larger (+8.39% best) but 2024-2026 1h crypto
produced enough lower-band breakdowns that −5.19% stops dominated.
**Eliminated: Sharpe < 0, profit < 0.**

### Candidate 3: Volatility Breakout (Trend Following)
`candidate_3.py` — class `VolatilityBreakout` (user's Candidate 6)

**Hypothesis:** The baseline's payoff structure (wins capped at +0.5-4%,
losses run to −5%) loses even at 79% win rate. Trend following inverts
the structure: small frequent losses, occasional large winners. In a
window where the market moved +68%, a long-only breakout system should
capture a fraction of the move. Range expansion is the same volatility
signal HAR models — the most natural HAR integration.

**IMPORTANT — IMPLEMENTATION BUG DISCOVERED AND FIXED (see below).**

**Result after bug fix: WINNER (marginal — see selection rationale)**

| Metric | Value |
|---|---|
| Trades | 364 (BTC 203 / ETH 161) |
| Win rate | 36.0% (131W / 0D / 233L) |
| Avg profit / trade | +0.04% |
| Total profit | +14.61 USDT (+1.46%) |
| Sharpe (closed) / Sortino | +0.10 / +0.29 |
| Profit factor | 1.04 |
| Max drawdown | 68.55 USDT (6.48%) |
| Mean profit p-value | 0.8374 |
| Best / worst trade | +29.83% / −5.19% |
| Avg duration | 1 day 2:24 |
| Exit reasons | 352 exit_signal / 1 force_exit / 11 stop_loss |

Per-pair: BTC/USDT 203 trades +0.01% (statistically flat), ETH/USDT
161 trades +1.45%. Exit signal now fires correctly (352 of 364 exits).

### Implementation Bug Discovery (research integrity)

The FIRST run of Candidate 3 showed **+11.21% with only 18 trades** and
looked like a strong winner. Exit-reason stats exposed the truth: the
`exit_signal` NEVER fired — 16 trades ended via stop_loss and **2 trades
were force-closed at timerange end** (BTC held 705 days, ETH 253 days).

Root cause: the exit condition `close < lowest_low` used
`low.rolling(N).min()` which INCLUDES the current bar. Since a bar's
close >= its own low, the rolling min is always <= the current close —
**the condition was structurally impossible**; the strategy could never
exit by signal. The "+11.21%" was an artifact of two positions riding
the entire bull window, not a working exit.

Fix (standard Donchian channel formulation, no lookahead — uses only
bars t-N..t-1): `low.rolling(N).min().shift(1)`.
Applied to `candidate_3.py`, `candidate_3b.py`, and all winner files.
Guarded by a regression test:
`test_winner_strategies.py::TestExitLogic::test_exit_signal_fires_on_channel_break`
(verified: fails on the buggy code, passes on the fix).

Corrected results are shown above and everywhere below. The buggy
artifact is disclosed here and NOT used for any conclusion.

### Candidate 3b: Volatility Breakout — Fast Rotation (OPTION B retest)
`candidate_3b.py` — class `VolatilityBreakoutFast`

**Pre-registered modification** (chosen BEFORE the retest, targeting the
pre-registered trade-count requirement, not profit): range multiplier
2.0x → 1.5x (wider entry funnel) and channel exit 10-bar → 5-bar low
(faster rotation). Motivation: the winner's binding constraint was
position rotation — 1-day average duration means few cycles per pair.

**Result: FAIL (modification rejected)**

| Metric | Value |
|---|---|
| Trades | 694 (BTC 367 / ETH 327) |
| Win rate | 33.1% (230W / 0D / 464L) |
| Avg profit / trade | −0.10% |
| Total profit | −71.05 USDT (−7.10%) |
| Sharpe (closed) / Sortino | −0.64 / −1.67 |
| Profit factor | 0.89 |
| Max drawdown | 121.12 USDT (11.53%) |
| Mean profit p-value | 0.3549 |
| Best / worst trade | +35.84% / −5.19% |

More trades ≠ better: the faster rotation traded noise, more whipsaw
stops, and negative expectancy. The modification hypothesis is
**rejected** — no further parameter changes were made (one modification
round per protocol).

### Eliminated without backtesting (Phase 1)

| User candidate | Reason for elimination |
|---|---|
| C3: EMA crossover | 1h EMA crossovers whipsaw heavily; with meaningful EMA pairs, expected trade count per asset is marginal. The volatility breakout (tested) is the more robust trend-following representative. |
| C4: RSI + volume | Volume confirmation only reduces trade count; it does not address the diagnosed win/loss asymmetry (the actual failure cause). |
| C5: Multi-timeframe RSI | Same mean-reversion payoff asymmetry as the baseline; adds complexity without fixing the core defect. |
| C7: EMA+RSI+volume | Superset of Candidate 1; if the trend filter cannot fix the asymmetry, an added volume filter cannot either. Candidate 1 tested as the family representative. |

---

## Winner Selected

**Strategy: VolatilityBreakout (`candidate_3.py` → `winner_baseline.py`)**

Selection criteria applied (MUST HAVE: Sharpe > 0, profit > 0,
trades >= 50, both assets positive, DD < 30%):

| Criterion | C1 | C2 | C3 | C3b |
|---|---|---|---|---|
| Sharpe > 0 | ✗ | ✗ | ✓ (+0.10) | ✗ |
| Total profit > 0 | ✗ | ✗ | ✓ (+1.46%) | ✗ |
| Trades >= 50 | ✗ | ✓ | ✓ (364) | ✓ |
| Both assets positive | ✗ | ✗ | ✓ (weak) | ✗ |
| Max DD < 30% | ✓ | ✓ | ✓ (6.48%) | ✓ |

Only C3 passes the MUST-HAVEs (C5 "both assets" is a weak pass: BTC is
statistically flat at +0.01%). Selected as winner — **with the explicit
caveat that the edge is marginal: Sharpe +0.10, PF 1.04, mean-profit
p-value 0.84 (indistinguishable from zero)**.

---

## Three Variant Results

HAR filter logic from `har_regime_filter.py` (Supabase reader) applied
identically to the previous experiment: B skips HIGH-regime entries,
C only trades HIGH-regime entries, unknown → medium fallback, failure →
allow (never blocks), exits never blocked.

**NOTE (expected and documented):** during backtesting the live Supabase
DB is unavailable, so the HAR filter is bypassed automatically
(`har_regime_filter not found. Winner running without HAR filter.` in the
log). B and C therefore equal A in backtest — this is by design; the HAR
effect is only observable in live paper trading.

### Strategy A — WinnerBaseline

| Metric | Value |
|---|---|
| Trades | 364 (BTC 203 +0.01% / ETH 161 +1.45%) |
| Win rate | 36.0% (131W / 233L) |
| Avg profit / trade | +0.04% |
| Total profit | +14.61 USDT (+1.46%) |
| Sharpe (closed) / Sortino / Calmar | +0.10 / +0.29 / +0.59 |
| SQN | 0.21 |
| Profit factor | 1.04 |
| Expectancy | +0.04% (+0.02 ratio) |
| Max drawdown | 68.55 USDT (6.48%) |
| Mean profit p-value | 0.8374 |
| Best / worst trade | +29.83% / −5.19% |
| Avg duration | 1 day 2:24 |
| Exit reasons | 352 exit_signal / 1 force_exit / 11 stop_loss |
| Max consecutive wins / losses | 5 / 15 |
| Market change (benchmark) | +68.27% |

### Strategy B — WinnerHARFiltered (skip HIGH regime)

| Metric | Value |
|---|---|
| Trades | 364 (BTC 203 / ETH 161) |
| Win rate | 36.0% |
| Avg profit / trade | +0.04% |
| Total profit | +14.61 USDT (+1.46%) |
| Sharpe (closed) | +0.10 |
| Profit factor | 1.04 |
| Max drawdown | 68.55 USDT (6.48%) |

(Identical to A in backtest — HAR bypassed, as documented above.)

### Strategy C — WinnerHARInverse (only HIGH regime)

| Metric | Value |
|---|---|
| Trades | 364 (BTC 203 / ETH 161) |
| Win rate | 36.0% |
| Avg profit / trade | +0.04% |
| Total profit | +14.61 USDT (+1.46%) |
| Sharpe (closed) | +0.10 |
| Profit factor | 1.04 |
| Max drawdown | 68.55 USDT (6.48%) |

(Identical to A in backtest — HAR bypassed, as documented above.)

---

## Time Stability Analysis

WinnerBaseline run on three 8-month periods (trade counts differ from
the full-window 364 because positions open across period boundaries are
closed and re-entered — each period is standalone):

| Metric | P1 20240101-20240901 | P2 20240901-20250501 | P3 20250501-20260101 |
|---|---|---|---|
| Trades | 116 | 123 | 126 |
| Total profit % | **+1.63%** | **−1.34%** | **+0.87%** |
| Sharpe (closed) | +0.38 | **−0.30** | +0.16 |
| Profit factor | 1.13 | 0.91 | 1.06 |
| Max drawdown | 6.05% | 5.35% | 4.11% |
| BTC/USDT | +0.59% (66) | +2.60% (67) | −3.48% (71) |
| ETH/USDT | +1.05% (50) | −3.94% (56) | +4.35% (55) |

**Conclusion: UNSTABLE.** The strategy is profitable in 2 of 3 periods
and negative in P2 (which contains the April 2025 tariff crash and
August 2025 crypto crash — both high-volatility regimes). Per-asset
signs flip between periods (BTC negative in P3, ETH negative in P2).
The edge, such as it is, does not persist consistently across time
or across pairs. Combined with p-value 0.84, this pattern is
consistent with a strategy whose true expectancy is zero.

---

## Gate Criteria Status

Pre-registered criteria (B = HAR-filtered variant):

| Criterion | Status | Evidence |
|---|---|---|
| C1: B Sharpe > A Sharpe | **PENDING** | B = A in backtest (HAR needs live DB). Only paper trading can evaluate. |
| C2: B max DD < A max DD | **PENDING** | Same as C1. |
| C3: B total trades >= 30 | **PASS** (backtest proxy) | 364 >= 30 in backtest (B = A). True filtered count only known in paper trading. |
| C4: C Sharpe < A Sharpe | **PENDING** | C = A in backtest. Only paper trading can evaluate. |
| C5: Results on both BTC and ETH | **WEAK PASS** | Both non-negative (BTC +0.01% flat, ETH +1.45%) — not robust. |
| C6: Stable across time periods | **FAIL** | P2 negative (Sharpe −0.30); per-pair signs flip between periods. |

Testable from backtest: 2 PASS (one weak), 1 FAIL, 3 PENDING.

---

## Recommendation

**OPTION C — ABANDON THIS APPROACH (do not start paper trading).**

**Reason (specific evidence):**

1. **The winner's edge is statistically indistinguishable from zero.**
   Sharpe +0.10, profit factor 1.04, expectancy +0.04%/trade, and
   critically **mean profit p-value = 0.8374** over 364 trades. There is
   no statistical evidence of positive expectancy.

2. **Gate C6 (time stability) FAILED.** The strategy lost money in the
   middle period and per-asset signs flip across periods — the hallmark
   of a zero-edge system, not a robust one.

3. **The modification retest (C3b) was rejected** (694 trades, −7.10%),
   showing the marginal positive result is not a frequency artifact that
   a parameter fix can rescue.

4. **The earlier "+11.21% winner" was an implementation artifact**
   (exit-signal bug), discovered and fixed mid-research. The corrected
   result (+1.46%) is the honest number, and it is not economically or
   statistically meaningful.

5. Per the research plan's own logic: testing the HAR regime filter on a
   base strategy with zero demonstrated expectancy is scientifically
   meaningless — the previous conclusion stands, now with a stronger
   evidence base.

**Alternative approach (what to try instead):**

- **Use HAR for position sizing / stop placement, not entry gating.**
  HAR's validated output is next-bar RANGE (not direction). A strategy
  that sets stoploss/ROI as multiples of `har_predicted_range` uses the
  model where it has proven skill. Entry gating (skip HIGH) is a weaker
  application of a range forecast.
- **Multi-asset trend following with trailing stops.** The breakout
  concept failed on 2 pairs with a channel exit; a trailing-stop variant
  (freqtrade `trailing_stop`) across a wider universe (SOL, XRP, BNB...)
  would increase both trade count and independence of the sample.
- **Higher timeframe (4h/1d) trend following.** Fewer, higher-quality
  signals; the 1h horizon in 2024-2026 was dominated by mean-reverting
  chop punctuated by crashes — hostile to both tested families.
- **Regime-conditional hybrid:** mean reversion in HAR low/medium
  regimes, momentum only in high regimes (rather than blocking high).

---

## Research Integrity Statement

All results generated from historical data in a walk-forward manner
(single pre-registered window 20240101-20260101 for all tests). No future
data used in strategy logic (the channel exit uses only bars t-N..t-1;
indicators are causal). No parameters were tuned on the test window —
all indicator periods are standard values chosen before any backtest;
the one modification round (C3b) targeted the pre-registered trade-count
requirement and was rejected by the data. Every candidate tested is
documented, including the discovered implementation bug and its
correction. Negative results are reported as findings, not hidden.
Paper trading only. No real orders placed. dry_run: true in all configs.
