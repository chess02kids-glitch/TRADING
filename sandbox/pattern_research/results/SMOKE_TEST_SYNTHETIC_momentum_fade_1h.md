> **WARNING — THIS IS NOT A RESEARCH RESULT.**
>
> This file is a *pipeline smoke test* produced from a seeded synthetic random walk
> (`tools/make_synthetic_candles.py`), because the machine it was generated on has no
> network egress to `api.kucoin.com`. The numbers below describe random data and say
> nothing about BTC or ETH. Re-run the CLI with live KuCoin data to get real results.
>
> The momentum_fade reading is the exact inverse of the "momentum: combined" signal,
> so on random data it lands near 50% with p > 0.05 and is reported CLOSED — a
> sub-50% continuation hit rate is not evidence of a tradable fade edge.

# Pattern Research Results — momentum_fade

_Generated:_ 2026-08-23T01:47:44Z  
_Source:_ local CSV file(s) — verify their provenance, 1h OHLCV, last 730 days  
_Assets:_ both (BTC/USDT + ETH/USDT)  
_Timeframe:_ 1h  
_Horizon:_ t+1 = 1 hour forward  
_Sandbox:_ `sandbox/pattern_research` — no DB, no secrets, no contact with the production system.

## Data

| Asset | Bars | Bar spacing | First bar (UTC) | Last bar (UTC) |
|---|---|---|---|---|
| BTC/USDT | 17520 | 1h | 2024-01-01 00:00:00+00:00 | 2025-12-30 23:00:00+00:00 |
| ETH/USDT | 17520 | 1h | 2024-01-01 00:00:00+00:00 | 2025-12-30 23:00:00+00:00 |

## Method

* Every detector is `.shift(1)`-ed: `signal[t]` reflects a pattern that **completed at bar t-1**, so it is known before bar `t` opens.
* `forward_return[t] = close[t+1]/close[t] - 1` — entry at the close of the signal bar, exit `horizon` bars later. No look-ahead. **A horizon is always counted in bars** of the selected timeframe.
* `correct = 1` when `sign(forward_return) == sign(signal)`.
* Diebold-Mariano: one-sided vs a 50/50 coin flip, Newey-West HAC with 3 lags (identical arithmetic to Phase 9A `dm_test.py`).
* Gates G1–G6 as pre-registered in Phase 9A; all-or-nothing verdict.
* Patterns with fewer than 50 occurrences are skipped, not tested (rule 6/7).

## Results

### momentum_fade: combined fade (inverse of momentum)

```
+----------------------------------------------------------+
| momentum_fade: combined fade (inverse of momentum)       |
+----------------------------------------------------------+
| Horizon:          t+1                                    |
| Occurrences:      2044                                   |
|   BTC/USDT     1030                                      |
|   ETH/USDT     1014                                      |
+----------------------------------------------------------+
| Hit rate (all):   51.1%                                  |
| Hit rate (BTC):   51.5%                                  |
| Hit rate (ETH):   50.7%                                  |
| Mean fwd return:  -0.0136%                               |
+----------------------------------------------------------+
| DM statistic:     0.988                                  |
| p-value:          0.1616                                 |
+----------------------------------------------------------+
| Temporal windows (pooled events):                        |
|   Older:   52.5%                                         |
|   Middle:  49.9%                                         |
|   Recent:  50.8%                                         |
+----------------------------------------------------------+
| GATES:                                                   |
| G1 (hit > 55%, both):   FAIL                             |
| G2 (DM p < 0.05):       FAIL                             |
| G3 (both assets > 50%): PASS                             |
| G4 (stability):         FAIL                             |
| G5 (no degradation):    PASS                             |
| G6 (n >= 30 per asset): PASS                             |
+----------------------------------------------------------+
| VERDICT: CLOSED                                          |
+----------------------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.162)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 51.9% | 50.4% | 52.1% | yes | no |
| ETH/USDT | 53.5% | 48.2% | 50.3% | no | no |

## Summary

| Signal | N | Hit rate | DM p | Verdict |
|---|---|---|---|---|
| momentum_fade: combined fade (inverse of momentum) | 2044 | 51.1% | 0.1616 | CLOSED |

**No signal cleared all six gates.** Every pattern tested above is reported as CLOSED. This is the expected outcome for simple public patterns on liquid 1h crypto data and is documented here rather than buried — negative results are results.
