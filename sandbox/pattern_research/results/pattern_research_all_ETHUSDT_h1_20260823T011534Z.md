# Pattern Research Results — all

_Generated:_ 2026-08-23T01:15:34Z  
_Source:_ CCXT KuCoin public API (spot), 1h OHLCV, last 730 days  
_Assets:_ ETH/USDT  
_Horizon:_ t+1  
_Sandbox:_ `sandbox/pattern_research` — no DB, no secrets, no contact with the production system.

## Data

| Asset | Bars | First bar (UTC) | Last bar (UTC) |
|---|---|---|---|
| ETH/USDT | 17519 | 2024-08-23 02:00:00+00:00 | 2026-08-23 00:00:00+00:00 |

## Method

* Every detector is `.shift(1)`-ed: `signal[t]` reflects a pattern that **completed at bar t-1**, so it is known before bar `t` opens.
* `forward_return[t] = close[t+1]/close[t] - 1` — entry at the close of the signal bar, exit `horizon` bars later. No look-ahead.
* `correct = 1` when `sign(forward_return) == sign(signal)`.
* Diebold-Mariano: one-sided vs a 50/50 coin flip, Newey-West HAC with 3 lags (identical arithmetic to Phase 9A `dm_test.py`).
* Gates G1–G6 as pre-registered in Phase 9A; all-or-nothing verdict.
* Patterns with fewer than 50 occurrences are skipped, not tested (rule 6/7).

## Time-of-day context (descriptive)

### ETH/USDT — hourly bias (full sample, sorted by win rate)

| Hour (UTC) | Mean return | Win rate | N |
|---|---|---|---|
| 17 | +0.0200% | 54.7% | 730 |
| 21 | +0.0530% | 53.4% | 730 |
| 19 | +0.0120% | 53.3% | 730 |
| 04 | +0.0126% | 52.5% | 730 |
| 15 | +0.0147% | 52.3% | 730 |
| 01 | +0.0096% | 52.3% | 729 |
| 03 | +0.0410% | 51.9% | 730 |
| 11 | +0.0123% | 51.9% | 730 |
| 09 | +0.0269% | 51.2% | 730 |
| 08 | +0.0102% | 51.0% | 730 |
| 22 | +0.0451% | 50.8% | 730 |
| 14 | -0.0189% | 50.8% | 730 |
| 20 | +0.0038% | 50.7% | 730 |
| 07 | +0.0087% | 50.1% | 730 |
| 12 | +0.0002% | 50.1% | 730 |
| 05 | +0.0039% | 49.9% | 730 |
| 00 | -0.0214% | 49.5% | 730 |
| 06 | -0.0353% | 49.0% | 730 |
| 02 | +0.0099% | 49.0% | 729 |
| 10 | -0.0278% | 48.6% | 730 |
| 18 | -0.0444% | 48.5% | 730 |
| 16 | -0.0054% | 48.5% | 730 |
| 23 | -0.0343% | 47.4% | 730 |
| 13 | -0.0465% | 46.6% | 730 |

### ETH/USDT — day-of-week bias (full sample)

| Day | Mean return | Win rate | N |
|---|---|---|---|
| Mon | +0.0106% | 51.7% | 2496 |
| Tue | -0.0129% | 48.9% | 2496 |
| Wed | +0.0340% | 52.0% | 2496 |
| Thu | -0.0321% | 47.7% | 2496 |
| Fri | +0.0091% | 51.5% | 2517 |
| Sat | +0.0077% | 52.1% | 2520 |
| Sun | -0.0021% | 50.2% | 2497 |

Best hours learned on the training half (bars up to 2025-08-23 00:00:00+00:00, n=8759): `[17, 4]` — evaluated only on the held-out bars from 2025-08-23 01:00:00+00:00 onwards (n=8760).

## Results

### momentum: higher_high_higher_low (+1)

```
+--------------------------------------------+
| momentum: higher_high_higher_low (+1)      |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      1068                     |
|   ETH/USDT     1068                        |
+--------------------------------------------+
| Hit rate (all):   47.9%                    |
| Hit rate (ETH):   47.9%                    |
| Mean fwd return:  -0.0089%                 |
+--------------------------------------------+
| DM statistic:     -1.331                   |
| p-value:          0.9084                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   47.8%                           |
|   Middle:  46.1%                           |
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

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.908)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| ETH/USDT | 47.7% | 47.6% | 48.9% | no | no |

> Note: Single-asset run: cross-asset gates G1/G3/G6 require both BTC/USDT and ETH/USDT and therefore cannot pass. Run --asset both for a verdict.

### momentum: lower_low_lower_high (-1)

```
+--------------------------------------------+
| momentum: lower_low_lower_high (-1)        |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      950                      |
|   ETH/USDT     950                         |
+--------------------------------------------+
| Hit rate (all):   48.3%                    |
| Hit rate (ETH):   48.3%                    |
| Mean fwd return:  +0.0105%                 |
+--------------------------------------------+
| DM statistic:     -1.050                   |
| p-value:          0.8532                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   50.5%                           |
|   Middle:  49.2%                           |
|   Recent:  45.3%                           |
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

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.853)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| ETH/USDT | 50.2% | 49.0% | 45.7% | no | no |

> Note: Single-asset run: cross-asset gates G1/G3/G6 require both BTC/USDT and ETH/USDT and therefore cannot pass. Run --asset both for a verdict.

### momentum: combined HH/HL + LL/LH

```
+--------------------------------------------+
| momentum: combined HH/HL + LL/LH           |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      2018                     |
|   ETH/USDT     2018                        |
+--------------------------------------------+
| Hit rate (all):   48.1%                    |
| Hit rate (ETH):   48.1%                    |
| Mean fwd return:  +0.0002%                 |
+--------------------------------------------+
| DM statistic:     -1.698                   |
| p-value:          0.9553                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   49.3%                           |
|   Middle:  47.3%                           |
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

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.955)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| ETH/USDT | 48.9% | 48.2% | 47.2% | no | no |

> Note: Single-asset run: cross-asset gates G1/G3/G6 require both BTC/USDT and ETH/USDT and therefore cannot pass. Run --asset both for a verdict.

### candlestick: bullish_engulfing (+1)

```
+--------------------------------------------+
| candlestick: bullish_engulfing (+1)        |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      1279                     |
|   ETH/USDT     1279                        |
+--------------------------------------------+
| Hit rate (all):   47.5%                    |
| Hit rate (ETH):   47.5%                    |
| Mean fwd return:  -0.0301%                 |
+--------------------------------------------+
| DM statistic:     -1.712                   |
| p-value:          0.9566                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   49.9%                           |
|   Middle:  47.4%                           |
|   Recent:  45.3%                           |
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

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.957)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| ETH/USDT | 49.8% | 47.5% | 45.0% | no | no |

> Note: Single-asset run: cross-asset gates G1/G3/G6 require both BTC/USDT and ETH/USDT and therefore cannot pass. Run --asset both for a verdict.

### candlestick: bearish_engulfing (-1)

```
+--------------------------------------------+
| candlestick: bearish_engulfing (-1)        |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      1484                     |
|   ETH/USDT     1484                        |
+--------------------------------------------+
| Hit rate (all):   48.0%                    |
| Hit rate (ETH):   48.0%                    |
| Mean fwd return:  +0.0243%                 |
+--------------------------------------------+
| DM statistic:     -1.527                   |
| p-value:          0.9366                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   47.9%                           |
|   Middle:  47.1%                           |
|   Recent:  49.2%                           |
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

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.937)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| ETH/USDT | 48.1% | 46.8% | 49.2% | no | no |

> Note: Single-asset run: cross-asset gates G1/G3/G6 require both BTC/USDT and ETH/USDT and therefore cannot pass. Run --asset both for a verdict.

### candlestick: doji (+1, tested as long)

```
+--------------------------------------------+
| candlestick: doji (+1, tested as long)     |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      1910                     |
|   ETH/USDT     1910                        |
+--------------------------------------------+
| Hit rate (all):   49.7%                    |
| Hit rate (ETH):   49.7%                    |
| Mean fwd return:  -0.0132%                 |
+--------------------------------------------+
| DM statistic:     -0.276                   |
| p-value:          0.6086                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   50.1%                           |
|   Middle:  49.6%                           |
|   Recent:  49.4%                           |
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

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.609)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| ETH/USDT | 50.4% | 49.3% | 49.3% | no | no |

> Note: Single-asset run: cross-asset gates G1/G3/G6 require both BTC/USDT and ETH/USDT and therefore cannot pass. Run --asset both for a verdict.

### candlestick: hammer (+1)

```
+--------------------------------------------+
| candlestick: hammer (+1)                   |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      290                      |
|   ETH/USDT     290                         |
+--------------------------------------------+
| Hit rate (all):   49.7%                    |
| Hit rate (ETH):   49.7%                    |
| Mean fwd return:  -0.0211%                 |
+--------------------------------------------+
| DM statistic:     -0.113                   |
| p-value:          0.5449                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   48.5%                           |
|   Middle:  51.5%                           |
|   Recent:  49.0%                           |
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

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.545)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| ETH/USDT | 48.6% | 47.6% | 53.9% | no | no |

> Note: Single-asset run: cross-asset gates G1/G3/G6 require both BTC/USDT and ETH/USDT and therefore cannot pass. Run --asset both for a verdict.

### volume: volume_spike (+1/-1)

```
+--------------------------------------------+
| volume: volume_spike (+1/-1)               |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      1379                     |
|   ETH/USDT     1379                        |
+--------------------------------------------+
| Hit rate (all):   48.9%                    |
| Hit rate (ETH):   48.9%                    |
| Mean fwd return:  -0.0193%                 |
+--------------------------------------------+
| DM statistic:     -0.859                   |
| p-value:          0.8049                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   48.9%                           |
|   Middle:  47.0%                           |
|   Recent:  50.8%                           |
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

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.805)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| ETH/USDT | 51.0% | 42.9% | 51.2% | no | no |

> Note: Single-asset run: cross-asset gates G1/G3/G6 require both BTC/USDT and ETH/USDT and therefore cannot pass. Run --asset both for a verdict.

### time_of_day: best-hours (out-of-sample)

```
+--------------------------------------------+
| time_of_day: best-hours (out-of-sample)    |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      730                      |
|   ETH/USDT     730                         |
+--------------------------------------------+
| Hit rate (all):   50.8%                    |
| Hit rate (ETH):   50.8%                    |
| Mean fwd return:  +0.0024%                 |
+--------------------------------------------+
| DM statistic:     0.442                    |
| p-value:          0.3292                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   53.7%                           |
|   Middle:  46.5%                           |
|   Recent:  52.3%                           |
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

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.329)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| ETH/USDT | 53.9% | 46.5% | 52.3% | no | no |

> Note: Single-asset run: cross-asset gates G1/G3/G6 require both BTC/USDT and ETH/USDT and therefore cannot pass. Run --asset both for a verdict.

## Summary

| Signal | N | Hit rate | DM p | Verdict |
|---|---|---|---|---|
| momentum: higher_high_higher_low (+1) | 1068 | 47.9% | 0.9084 | CLOSED |
| momentum: lower_low_lower_high (-1) | 950 | 48.3% | 0.8532 | CLOSED |
| momentum: combined HH/HL + LL/LH | 2018 | 48.1% | 0.9553 | CLOSED |
| candlestick: bullish_engulfing (+1) | 1279 | 47.5% | 0.9566 | CLOSED |
| candlestick: bearish_engulfing (-1) | 1484 | 48.0% | 0.9366 | CLOSED |
| candlestick: doji (+1, tested as long) | 1910 | 49.7% | 0.6086 | CLOSED |
| candlestick: hammer (+1) | 290 | 49.7% | 0.5449 | CLOSED |
| volume: volume_spike (+1/-1) | 1379 | 48.9% | 0.8049 | CLOSED |
| time_of_day: best-hours (out-of-sample) | 730 | 50.8% | 0.3292 | CLOSED |

**No signal cleared all six gates.** Every pattern tested above is reported as CLOSED. This is the expected outcome for simple public patterns on liquid 1h crypto data and is documented here rather than buried — negative results are results.
