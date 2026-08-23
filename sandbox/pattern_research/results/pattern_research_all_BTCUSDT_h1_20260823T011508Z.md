# Pattern Research Results — all

_Generated:_ 2026-08-23T01:15:08Z  
_Source:_ CCXT KuCoin public API (spot), 1h OHLCV, last 730 days  
_Assets:_ BTC/USDT  
_Horizon:_ t+1  
_Sandbox:_ `sandbox/pattern_research` — no DB, no secrets, no contact with the production system.

## Data

| Asset | Bars | First bar (UTC) | Last bar (UTC) |
|---|---|---|---|
| BTC/USDT | 17519 | 2024-08-23 02:00:00+00:00 | 2026-08-23 00:00:00+00:00 |

## Method

* Every detector is `.shift(1)`-ed: `signal[t]` reflects a pattern that **completed at bar t-1**, so it is known before bar `t` opens.
* `forward_return[t] = close[t+1]/close[t] - 1` — entry at the close of the signal bar, exit `horizon` bars later. No look-ahead.
* `correct = 1` when `sign(forward_return) == sign(signal)`.
* Diebold-Mariano: one-sided vs a 50/50 coin flip, Newey-West HAC with 3 lags (identical arithmetic to Phase 9A `dm_test.py`).
* Gates G1–G6 as pre-registered in Phase 9A; all-or-nothing verdict.
* Patterns with fewer than 50 occurrences are skipped, not tested (rule 6/7).

## Time-of-day context (descriptive)

### BTC/USDT — hourly bias (full sample, sorted by win rate)

| Hour (UTC) | Mean return | Win rate | N |
|---|---|---|---|
| 17 | +0.0164% | 54.1% | 730 |
| 11 | +0.0088% | 53.6% | 730 |
| 14 | +0.0072% | 52.5% | 730 |
| 21 | +0.0268% | 52.2% | 730 |
| 22 | +0.0255% | 51.2% | 730 |
| 08 | +0.0173% | 51.2% | 730 |
| 09 | +0.0125% | 51.2% | 730 |
| 19 | +0.0016% | 51.1% | 730 |
| 01 | +0.0124% | 51.0% | 729 |
| 15 | +0.0268% | 51.0% | 730 |
| 04 | +0.0058% | 51.0% | 730 |
| 20 | +0.0111% | 50.8% | 730 |
| 18 | -0.0152% | 50.7% | 730 |
| 06 | -0.0184% | 50.5% | 730 |
| 05 | -0.0009% | 49.7% | 730 |
| 10 | -0.0093% | 49.5% | 730 |
| 12 | -0.0037% | 49.3% | 730 |
| 03 | +0.0200% | 49.2% | 730 |
| 07 | +0.0034% | 48.9% | 730 |
| 00 | -0.0108% | 47.9% | 730 |
| 23 | -0.0290% | 47.8% | 730 |
| 16 | -0.0142% | 47.1% | 730 |
| 02 | +0.0029% | 47.1% | 729 |
| 13 | -0.0356% | 46.3% | 730 |

### BTC/USDT — day-of-week bias (full sample)

| Day | Mean return | Win rate | N |
|---|---|---|---|
| Mon | +0.0105% | 50.0% | 2496 |
| Tue | -0.0033% | 50.2% | 2496 |
| Wed | +0.0197% | 51.6% | 2496 |
| Thu | -0.0184% | 48.5% | 2496 |
| Fri | +0.0079% | 51.1% | 2517 |
| Sat | -0.0020% | 49.9% | 2520 |
| Sun | +0.0034% | 50.2% | 2497 |

Best hours learned on the training half (bars up to 2025-08-23 00:00:00+00:00, n=8759): `[17]` — evaluated only on the held-out bars from 2025-08-23 01:00:00+00:00 onwards (n=8760).

## Results

### momentum: higher_high_higher_low (+1)

```
+--------------------------------------------+
| momentum: higher_high_higher_low (+1)      |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      1158                     |
|   BTC/USDT     1158                        |
+--------------------------------------------+
| Hit rate (all):   47.6%                    |
| Hit rate (BTC):   47.6%                    |
| Mean fwd return:  +0.0030%                 |
+--------------------------------------------+
| DM statistic:     -1.660                   |
| p-value:          0.9515                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   47.4%                           |
|   Middle:  46.6%                           |
|   Recent:  48.7%                           |
+--------------------------------------------+
| GATES:                                     |
| G1 (hit > 55%, both):   FAIL               |
| G2 (DM p < 0.05):       FAIL               |
| G3 (both assets > 50%): FAIL               |
| G4 (stability):         FAIL               |
| G5 (no degradation):    PASS               |
| G6 (n >= 30 per asset): FAIL               |
+--------------------------------------------+
| VERDICT: CLOSED                            |
+--------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.952)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 46.4% | 46.2% | 51.1% | no | no |

> Note: Single-asset run: cross-asset gates G1/G3/G6 require both BTC/USDT and ETH/USDT and therefore cannot pass. Run --asset both for a verdict.

### momentum: lower_low_lower_high (-1)

```
+--------------------------------------------+
| momentum: lower_low_lower_high (-1)        |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      1098                     |
|   BTC/USDT     1098                        |
+--------------------------------------------+
| Hit rate (all):   49.7%                    |
| Hit rate (BTC):   49.7%                    |
| Mean fwd return:  +0.0145%                 |
+--------------------------------------------+
| DM statistic:     -0.186                   |
| p-value:          0.5736                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   47.5%                           |
|   Middle:  52.7%                           |
|   Recent:  48.9%                           |
+--------------------------------------------+
| GATES:                                     |
| G1 (hit > 55%, both):   FAIL               |
| G2 (DM p < 0.05):       FAIL               |
| G3 (both assets > 50%): FAIL               |
| G4 (stability):         FAIL               |
| G5 (no degradation):    PASS               |
| G6 (n >= 30 per asset): FAIL               |
+--------------------------------------------+
| VERDICT: CLOSED                            |
+--------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.574)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 48.0% | 52.1% | 48.6% | no | no |

> Note: Single-asset run: cross-asset gates G1/G3/G6 require both BTC/USDT and ETH/USDT and therefore cannot pass. Run --asset both for a verdict.

### momentum: combined HH/HL + LL/LH

```
+--------------------------------------------+
| momentum: combined HH/HL + LL/LH           |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      2256                     |
|   BTC/USDT     2256                        |
+--------------------------------------------+
| Hit rate (all):   48.6%                    |
| Hit rate (BTC):   48.6%                    |
| Mean fwd return:  +0.0086%                 |
+--------------------------------------------+
| DM statistic:     -1.322                   |
| p-value:          0.9069                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   47.3%                           |
|   Middle:  49.7%                           |
|   Recent:  48.8%                           |
+--------------------------------------------+
| GATES:                                     |
| G1 (hit > 55%, both):   FAIL               |
| G2 (DM p < 0.05):       FAIL               |
| G3 (both assets > 50%): FAIL               |
| G4 (stability):         FAIL               |
| G5 (no degradation):    PASS               |
| G6 (n >= 30 per asset): FAIL               |
+--------------------------------------------+
| VERDICT: CLOSED                            |
+--------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.907)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 47.1% | 49.1% | 49.8% | no | no |

> Note: Single-asset run: cross-asset gates G1/G3/G6 require both BTC/USDT and ETH/USDT and therefore cannot pass. Run --asset both for a verdict.

### candlestick: bullish_engulfing (+1)

```
+--------------------------------------------+
| candlestick: bullish_engulfing (+1)        |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      1298                     |
|   BTC/USDT     1298                        |
+--------------------------------------------+
| Hit rate (all):   47.9%                    |
| Hit rate (BTC):   47.9%                    |
| Mean fwd return:  +0.0093%                 |
+--------------------------------------------+
| DM statistic:     -1.542                   |
| p-value:          0.9384                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   50.3%                           |
|   Middle:  49.0%                           |
|   Recent:  44.4%                           |
+--------------------------------------------+
| GATES:                                     |
| G1 (hit > 55%, both):   FAIL               |
| G2 (DM p < 0.05):       FAIL               |
| G3 (both assets > 50%): FAIL               |
| G4 (stability):         FAIL               |
| G5 (no degradation):    PASS               |
| G6 (n >= 30 per asset): FAIL               |
+--------------------------------------------+
| VERDICT: CLOSED                            |
+--------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.938)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 50.8% | 48.5% | 44.1% | no | no |

> Note: Single-asset run: cross-asset gates G1/G3/G6 require both BTC/USDT and ETH/USDT and therefore cannot pass. Run --asset both for a verdict.

### candlestick: bearish_engulfing (-1)

```
+--------------------------------------------+
| candlestick: bearish_engulfing (-1)        |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      1687                     |
|   BTC/USDT     1687                        |
+--------------------------------------------+
| Hit rate (all):   45.9%                    |
| Hit rate (BTC):   45.9%                    |
| Mean fwd return:  +0.0256%                 |
+--------------------------------------------+
| DM statistic:     -3.389                   |
| p-value:          0.9996                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   43.0%                           |
|   Middle:  48.0%                           |
|   Recent:  46.6%                           |
+--------------------------------------------+
| GATES:                                     |
| G1 (hit > 55%, both):   FAIL               |
| G2 (DM p < 0.05):       FAIL               |
| G3 (both assets > 50%): FAIL               |
| G4 (stability):         FAIL               |
| G5 (no degradation):    PASS               |
| G6 (n >= 30 per asset): FAIL               |
+--------------------------------------------+
| VERDICT: CLOSED                            |
+--------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=1)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 42.5% | 48.0% | 46.6% | no | no |

> Note: Single-asset run: cross-asset gates G1/G3/G6 require both BTC/USDT and ETH/USDT and therefore cannot pass. Run --asset both for a verdict.

### candlestick: doji (+1, tested as long)

```
+--------------------------------------------+
| candlestick: doji (+1, tested as long)     |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      1753                     |
|   BTC/USDT     1753                        |
+--------------------------------------------+
| Hit rate (all):   51.1%                    |
| Hit rate (BTC):   51.1%                    |
| Mean fwd return:  +0.0041%                 |
+--------------------------------------------+
| DM statistic:     0.939                    |
| p-value:          0.174                    |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   50.1%                           |
|   Middle:  51.0%                           |
|   Recent:  52.2%                           |
+--------------------------------------------+
| GATES:                                     |
| G1 (hit > 55%, both):   FAIL               |
| G2 (DM p < 0.05):       FAIL               |
| G3 (both assets > 50%): FAIL               |
| G4 (stability):         PASS               |
| G5 (no degradation):    PASS               |
| G6 (n >= 30 per asset): FAIL               |
+--------------------------------------------+
| VERDICT: CLOSED                            |
+--------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.174)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 50.7% | 50.3% | 52.3% | yes | no |

> Note: Single-asset run: cross-asset gates G1/G3/G6 require both BTC/USDT and ETH/USDT and therefore cannot pass. Run --asset both for a verdict.

### candlestick: hammer (+1)

```
+--------------------------------------------+
| candlestick: hammer (+1)                   |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      294                      |
|   BTC/USDT     294                         |
+--------------------------------------------+
| Hit rate (all):   48.6%                    |
| Hit rate (BTC):   48.6%                    |
| Mean fwd return:  -0.0194%                 |
+--------------------------------------------+
| DM statistic:     -0.514                   |
| p-value:          0.6965                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   46.9%                           |
|   Middle:  49.0%                           |
|   Recent:  50.0%                           |
+--------------------------------------------+
| GATES:                                     |
| G1 (hit > 55%, both):   FAIL               |
| G2 (DM p < 0.05):       FAIL               |
| G3 (both assets > 50%): FAIL               |
| G4 (stability):         FAIL               |
| G5 (no degradation):    PASS               |
| G6 (n >= 30 per asset): FAIL               |
+--------------------------------------------+
| VERDICT: CLOSED                            |
+--------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.696)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 47.5% | 49.5% | 48.8% | no | no |

> Note: Single-asset run: cross-asset gates G1/G3/G6 require both BTC/USDT and ETH/USDT and therefore cannot pass. Run --asset both for a verdict.

### volume: volume_spike (+1/-1)

```
+--------------------------------------------+
| volume: volume_spike (+1/-1)               |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      1632                     |
|   BTC/USDT     1632                        |
+--------------------------------------------+
| Hit rate (all):   48.6%                    |
| Hit rate (BTC):   48.6%                    |
| Mean fwd return:  +0.0037%                 |
+--------------------------------------------+
| DM statistic:     -1.211                   |
| p-value:          0.8871                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   48.7%                           |
|   Middle:  49.3%                           |
|   Recent:  47.8%                           |
+--------------------------------------------+
| GATES:                                     |
| G1 (hit > 55%, both):   FAIL               |
| G2 (DM p < 0.05):       FAIL               |
| G3 (both assets > 50%): FAIL               |
| G4 (stability):         FAIL               |
| G5 (no degradation):    PASS               |
| G6 (n >= 30 per asset): FAIL               |
+--------------------------------------------+
| VERDICT: CLOSED                            |
+--------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.887)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 49.4% | 47.9% | 48.4% | no | no |

> Note: Single-asset run: cross-asset gates G1/G3/G6 require both BTC/USDT and ETH/USDT and therefore cannot pass. Run --asset both for a verdict.

### time_of_day: best-hours (out-of-sample)

```
+--------------------------------------------+
| time_of_day: best-hours (out-of-sample)    |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      365                      |
|   BTC/USDT     365                         |
+--------------------------------------------+
| Hit rate (all):   50.1%                    |
| Hit rate (BTC):   50.1%                    |
| Mean fwd return:  -0.0221%                 |
+--------------------------------------------+
| DM statistic:     0.048                    |
| p-value:          0.4808                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   53.3%                           |
|   Middle:  47.5%                           |
|   Recent:  49.6%                           |
+--------------------------------------------+
| GATES:                                     |
| G1 (hit > 55%, both):   FAIL               |
| G2 (DM p < 0.05):       FAIL               |
| G3 (both assets > 50%): FAIL               |
| G4 (stability):         FAIL               |
| G5 (no degradation):    PASS               |
| G6 (n >= 30 per asset): FAIL               |
+--------------------------------------------+
| VERDICT: CLOSED                            |
+--------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.481)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 53.7% | 47.9% | 49.2% | no | no |

> Note: Single-asset run: cross-asset gates G1/G3/G6 require both BTC/USDT and ETH/USDT and therefore cannot pass. Run --asset both for a verdict.

## Summary

| Signal | N | Hit rate | DM p | Verdict |
|---|---|---|---|---|
| momentum: higher_high_higher_low (+1) | 1158 | 47.6% | 0.9515 | CLOSED |
| momentum: lower_low_lower_high (-1) | 1098 | 49.7% | 0.5736 | CLOSED |
| momentum: combined HH/HL + LL/LH | 2256 | 48.6% | 0.9069 | CLOSED |
| candlestick: bullish_engulfing (+1) | 1298 | 47.9% | 0.9384 | CLOSED |
| candlestick: bearish_engulfing (-1) | 1687 | 45.9% | 0.9996 | CLOSED |
| candlestick: doji (+1, tested as long) | 1753 | 51.1% | 0.174 | CLOSED |
| candlestick: hammer (+1) | 294 | 48.6% | 0.6965 | CLOSED |
| volume: volume_spike (+1/-1) | 1632 | 48.6% | 0.8871 | CLOSED |
| time_of_day: best-hours (out-of-sample) | 365 | 50.1% | 0.4808 | CLOSED |

**No signal cleared all six gates.** Every pattern tested above is reported as CLOSED. This is the expected outcome for simple public patterns on liquid 1h crypto data and is documented here rather than buried — negative results are results.
