# Pattern Research Results — momentum_fade

_Generated:_ 2026-08-23T01:53:46Z  
_Source:_ local CSV file(s) — verify their provenance, 1h OHLCV, last 730 days  
_Assets:_ both (BTC/USDT + ETH/USDT)  
_Timeframe:_ 1h  
_Horizon:_ t+1 = 1 hour forward  
_Sandbox:_ `sandbox/pattern_research` — no DB, no secrets, no contact with the production system.

## Data

| Asset | Bars | Bar spacing | First bar (UTC) | Last bar (UTC) |
|---|---|---|---|---|
| BTC/USDT | 17519 | 1h | 2024-08-23 02:00:00+00:00 | 2026-08-23 00:00:00+00:00 |
| ETH/USDT | 17519 | 1h | 2024-08-23 02:00:00+00:00 | 2026-08-23 00:00:00+00:00 |

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
| Occurrences:      4274                                   |
|   BTC/USDT     2256                                      |
|   ETH/USDT     2018                                      |
+----------------------------------------------------------+
| Hit rate (all):   51.5%                                  |
| Hit rate (BTC):   51.4%                                  |
| Hit rate (ETH):   51.7%                                  |
| Mean fwd return:  +0.0047%                               |
+----------------------------------------------------------+
| DM statistic:     1.880                                  |
| p-value:          0.03007                                |
+----------------------------------------------------------+
| Temporal windows (pooled events):                        |
|   Older:   51.6%                                         |
|   Middle:  51.4%                                         |
|   Recent:  51.6%                                         |
+----------------------------------------------------------+
| GATES:                                                   |
| G1 (hit > 55%, both):   FAIL                             |
| G2 (DM p < 0.05):       PASS                             |
| G3 (both assets > 50%): PASS                             |
| G4 (stability):         PASS                             |
| G5 (no degradation):    PASS                             |
| G6 (n >= 30 per asset): PASS                             |
+----------------------------------------------------------+
| VERDICT: CLOSED                                          |
+----------------------------------------------------------+
```

DM conclusion: SIGNAL SIGNIFICANTLY BETTER THAN RANDOM (p=0.0301 < 0.05)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 52.9% | 50.9% | 50.2% | yes | no |
| ETH/USDT | 50.8% | 51.8% | 52.6% | yes | no |

## Summary

| Signal | N | Hit rate | DM p | Verdict |
|---|---|---|---|---|
| momentum_fade: combined fade (inverse of momentum) | 4274 | 51.5% | 0.03007 | CLOSED |

**No signal cleared all six gates.** Every pattern tested above is reported as CLOSED. This is the expected outcome for simple public patterns on liquid 1h crypto data and is documented here rather than buried — negative results are results.
