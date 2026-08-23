# Pattern Research Results — all

_Generated:_ 2026-08-23T01:54:15Z  
_Source:_ CCXT KuCoin public API (spot), 4h OHLCV, last 730 days  
_Assets:_ both (BTC/USDT + ETH/USDT)  
_Timeframe:_ 4h  
_Horizon:_ t+1 = 4 hours forward  
_Sandbox:_ `sandbox/pattern_research` — no DB, no secrets, no contact with the production system.

## Data

| Asset | Bars | Bar spacing | First bar (UTC) | Last bar (UTC) |
|---|---|---|---|---|
| BTC/USDT | 4379 | 4h | 2024-08-23 04:00:00+00:00 | 2026-08-22 20:00:00+00:00 |
| ETH/USDT | 4379 | 4h | 2024-08-23 04:00:00+00:00 | 2026-08-22 20:00:00+00:00 |

## Method

* Every detector is `.shift(1)`-ed: `signal[t]` reflects a pattern that **completed at bar t-1**, so it is known before bar `t` opens.
* `forward_return[t] = close[t+1]/close[t] - 1` — entry at the close of the signal bar, exit `horizon` bars later. No look-ahead. **A horizon is always counted in bars** of the selected timeframe.
* `correct = 1` when `sign(forward_return) == sign(signal)`.
* Diebold-Mariano: one-sided vs a 50/50 coin flip, Newey-West HAC with 3 lags (identical arithmetic to Phase 9A `dm_test.py`).
* Gates G1–G6 as pre-registered in Phase 9A; all-or-nothing verdict.
* Patterns with fewer than 50 occurrences are skipped, not tested (rule 6/7).

## Time-of-day context (descriptive)

### BTC/USDT — hourly bias (full sample, sorted by win rate)

| Hour (UTC) | Mean return | Win rate | N |
|---|---|---|---|
| 08 | +0.0294% | 52.3% | 730 |
| 20 | +0.0341% | 51.2% | 730 |
| 00 | +0.0237% | 51.0% | 729 |
| 16 | -0.0112% | 50.3% | 730 |
| 04 | -0.0115% | 49.7% | 729 |
| 12 | -0.0054% | 49.6% | 730 |
| 01 | +nan% | nan% | 0 |
| 02 | +nan% | nan% | 0 |
| 03 | +nan% | nan% | 0 |
| 05 | +nan% | nan% | 0 |
| 06 | +nan% | nan% | 0 |
| 07 | +nan% | nan% | 0 |
| 09 | +nan% | nan% | 0 |
| 10 | +nan% | nan% | 0 |
| 11 | +nan% | nan% | 0 |
| 13 | +nan% | nan% | 0 |
| 14 | +nan% | nan% | 0 |
| 15 | +nan% | nan% | 0 |
| 17 | +nan% | nan% | 0 |
| 18 | +nan% | nan% | 0 |
| 19 | +nan% | nan% | 0 |
| 21 | +nan% | nan% | 0 |
| 22 | +nan% | nan% | 0 |
| 23 | +nan% | nan% | 0 |

### BTC/USDT — day-of-week bias (full sample)

| Day | Mean return | Win rate | N |
|---|---|---|---|
| Mon | +0.0415% | 53.2% | 624 |
| Tue | -0.0134% | 48.7% | 624 |
| Wed | +0.0796% | 49.2% | 624 |
| Thu | -0.0737% | 48.1% | 624 |
| Fri | +0.0299% | 51.0% | 628 |
| Sat | -0.0082% | 51.3% | 630 |
| Sun | +0.0132% | 53.4% | 624 |

Best hours learned on the training half (bars up to 2025-08-22 20:00:00+00:00, n=2189): `[]` — evaluated only on the held-out bars from 2025-08-23 00:00:00+00:00 onwards (n=2190).

### ETH/USDT — hourly bias (full sample, sorted by win rate)

| Hour (UTC) | Mean return | Win rate | N |
|---|---|---|---|
| 20 | +0.0667% | 53.3% | 730 |
| 16 | -0.0170% | 53.2% | 730 |
| 00 | +0.0379% | 51.6% | 729 |
| 04 | -0.0124% | 48.6% | 729 |
| 08 | +0.0214% | 48.5% | 730 |
| 12 | -0.0497% | 47.7% | 730 |
| 01 | +nan% | nan% | 0 |
| 02 | +nan% | nan% | 0 |
| 03 | +nan% | nan% | 0 |
| 05 | +nan% | nan% | 0 |
| 06 | +nan% | nan% | 0 |
| 07 | +nan% | nan% | 0 |
| 09 | +nan% | nan% | 0 |
| 10 | +nan% | nan% | 0 |
| 11 | +nan% | nan% | 0 |
| 13 | +nan% | nan% | 0 |
| 14 | +nan% | nan% | 0 |
| 15 | +nan% | nan% | 0 |
| 17 | +nan% | nan% | 0 |
| 18 | +nan% | nan% | 0 |
| 19 | +nan% | nan% | 0 |
| 21 | +nan% | nan% | 0 |
| 22 | +nan% | nan% | 0 |
| 23 | +nan% | nan% | 0 |

### ETH/USDT — day-of-week bias (full sample)

| Day | Mean return | Win rate | N |
|---|---|---|---|
| Mon | +0.0424% | 52.7% | 624 |
| Tue | -0.0526% | 47.8% | 624 |
| Wed | +0.1369% | 51.3% | 624 |
| Thu | -0.1290% | 44.7% | 624 |
| Fri | +0.0342% | 52.7% | 628 |
| Sat | +0.0305% | 52.2% | 630 |
| Sun | -0.0082% | 51.8% | 624 |

Best hours learned on the training half (bars up to 2025-08-22 20:00:00+00:00, n=2189): `[20]` — evaluated only on the held-out bars from 2025-08-23 00:00:00+00:00 onwards (n=2190).

## Results

### momentum: higher_high_higher_low (+1)

```
+----------------------------------------------------------+
| momentum: higher_high_higher_low (+1)                    |
+----------------------------------------------------------+
| Horizon:          t+1                                    |
| Occurrences:      500                                    |
|   BTC/USDT     259                                       |
|   ETH/USDT     241                                       |
+----------------------------------------------------------+
| Hit rate (all):   49.0%                                  |
| Hit rate (BTC):   50.2%                                  |
| Hit rate (ETH):   47.7%                                  |
| Mean fwd return:  +0.0669%                               |
+----------------------------------------------------------+
| DM statistic:     -0.410                                 |
| p-value:          0.6589                                 |
+----------------------------------------------------------+
| Temporal windows (pooled events):                        |
|   Older:   52.1%                                         |
|   Middle:  46.7%                                         |
|   Recent:  48.2%                                         |
+----------------------------------------------------------+
| GATES:                                                   |
| G1 (hit > 55%, both):   FAIL                             |
| G2 (DM p < 0.05):       FAIL                             |
| G3 (both assets > 50%): FAIL                             |
| G4 (stability):         FAIL                             |
| G5 (no degradation):    PASS                             |
| G6 (n >= 30 per asset): PASS                             |
+----------------------------------------------------------+
| VERDICT: CLOSED                                          |
+----------------------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.659)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 56.3% | 40.5% | 52.8% | no | no |
| ETH/USDT | 43.9% | 47.8% | 53.0% | no | no |

### momentum: lower_low_lower_high (-1)

```
+----------------------------------------------------------+
| momentum: lower_low_lower_high (-1)                      |
+----------------------------------------------------------+
| Horizon:          t+1                                    |
| Occurrences:      436                                    |
|   BTC/USDT     203                                       |
|   ETH/USDT     233                                       |
+----------------------------------------------------------+
| Hit rate (all):   52.1%                                  |
| Hit rate (BTC):   52.7%                                  |
| Hit rate (ETH):   51.5%                                  |
| Mean fwd return:  -0.1647%                               |
+----------------------------------------------------------+
| DM statistic:     0.833                                  |
| p-value:          0.2025                                 |
+----------------------------------------------------------+
| Temporal windows (pooled events):                        |
|   Older:   53.4%                                         |
|   Middle:  49.0%                                         |
|   Recent:  53.8%                                         |
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

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.202)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 55.1% | 47.8% | 55.4% | no | no |
| ETH/USDT | 52.1% | 50.0% | 52.1% | no | no |

### momentum: combined HH/HL + LL/LH

```
+----------------------------------------------------------+
| momentum: combined HH/HL + LL/LH                         |
+----------------------------------------------------------+
| Horizon:          t+1                                    |
| Occurrences:      936                                    |
|   BTC/USDT     462                                       |
|   ETH/USDT     474                                       |
+----------------------------------------------------------+
| Hit rate (all):   50.4%                                  |
| Hit rate (BTC):   51.3%                                  |
| Hit rate (ETH):   49.6%                                  |
| Mean fwd return:  -0.0410%                               |
+----------------------------------------------------------+
| DM statistic:     0.238                                  |
| p-value:          0.4059                                 |
+----------------------------------------------------------+
| Temporal windows (pooled events):                        |
|   Older:   52.2%                                         |
|   Middle:  47.8%                                         |
|   Recent:  51.3%                                         |
+----------------------------------------------------------+
| GATES:                                                   |
| G1 (hit > 55%, both):   FAIL                             |
| G2 (DM p < 0.05):       FAIL                             |
| G3 (both assets > 50%): FAIL                             |
| G4 (stability):         FAIL                             |
| G5 (no degradation):    PASS                             |
| G6 (n >= 30 per asset): PASS                             |
+----------------------------------------------------------+
| VERDICT: CLOSED                                          |
+----------------------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.406)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 55.8% | 43.8% | 54.0% | no | no |
| ETH/USDT | 48.3% | 48.8% | 52.6% | no | no |

### candlestick: bullish_engulfing (+1)

```
+----------------------------------------------------------+
| candlestick: bullish_engulfing (+1)                      |
+----------------------------------------------------------+
| Horizon:          t+1                                    |
| Occurrences:      637                                    |
|   BTC/USDT     336                                       |
|   ETH/USDT     301                                       |
+----------------------------------------------------------+
| Hit rate (all):   48.0%                                  |
| Hit rate (BTC):   46.1%                                  |
| Hit rate (ETH):   50.2%                                  |
| Mean fwd return:  +0.0026%                               |
+----------------------------------------------------------+
| DM statistic:     -0.930                                 |
| p-value:          0.8238                                 |
+----------------------------------------------------------+
| Temporal windows (pooled events):                        |
|   Older:   53.1%                                         |
|   Middle:  43.4%                                         |
|   Recent:  47.6%                                         |
+----------------------------------------------------------+
| GATES:                                                   |
| G1 (hit > 55%, both):   FAIL                             |
| G2 (DM p < 0.05):       FAIL                             |
| G3 (both assets > 50%): FAIL                             |
| G4 (stability):         FAIL                             |
| G5 (no degradation):    PASS                             |
| G6 (n >= 30 per asset): PASS                             |
+----------------------------------------------------------+
| VERDICT: CLOSED                                          |
+----------------------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.824)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 52.3% | 40.0% | 46.5% | no | no |
| ETH/USDT | 53.7% | 50.4% | 46.1% | no | no |

### candlestick: bearish_engulfing (-1)

```
+----------------------------------------------------------+
| candlestick: bearish_engulfing (-1)                      |
+----------------------------------------------------------+
| Horizon:          t+1                                    |
| Occurrences:      825                                    |
|   BTC/USDT     445                                       |
|   ETH/USDT     380                                       |
+----------------------------------------------------------+
| Hit rate (all):   45.7%                                  |
| Hit rate (BTC):   44.7%                                  |
| Hit rate (ETH):   46.8%                                  |
| Mean fwd return:  +0.0374%                               |
+----------------------------------------------------------+
| DM statistic:     -2.324                                 |
| p-value:          0.9899                                 |
+----------------------------------------------------------+
| Temporal windows (pooled events):                        |
|   Older:   46.9%                                         |
|   Middle:  44.7%                                         |
|   Recent:  45.5%                                         |
+----------------------------------------------------------+
| GATES:                                                   |
| G1 (hit > 55%, both):   FAIL                             |
| G2 (DM p < 0.05):       FAIL                             |
| G3 (both assets > 50%): FAIL                             |
| G4 (stability):         FAIL                             |
| G5 (no degradation):    PASS                             |
| G6 (n >= 30 per asset): PASS                             |
+----------------------------------------------------------+
| VERDICT: CLOSED                                          |
+----------------------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.99)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 47.8% | 43.2% | 43.4% | no | no |
| ETH/USDT | 46.5% | 45.5% | 48.0% | no | no |

### candlestick: doji (+1, tested as long)

```
+----------------------------------------------------------+
| candlestick: doji (+1, tested as long)                   |
+----------------------------------------------------------+
| Horizon:          t+1                                    |
| Occurrences:      999                                    |
|   BTC/USDT     466                                       |
|   ETH/USDT     533                                       |
+----------------------------------------------------------+
| Hit rate (all):   47.6%                                  |
| Hit rate (BTC):   46.8%                                  |
| Hit rate (ETH):   48.4%                                  |
| Mean fwd return:  -0.0486%                               |
+----------------------------------------------------------+
| DM statistic:     -1.500                                 |
| p-value:          0.9331                                 |
+----------------------------------------------------------+
| Temporal windows (pooled events):                        |
|   Older:   47.7%                                         |
|   Middle:  50.2%                                         |
|   Recent:  45.0%                                         |
+----------------------------------------------------------+
| GATES:                                                   |
| G1 (hit > 55%, both):   FAIL                             |
| G2 (DM p < 0.05):       FAIL                             |
| G3 (both assets > 50%): FAIL                             |
| G4 (stability):         FAIL                             |
| G5 (no degradation):    PASS                             |
| G6 (n >= 30 per asset): PASS                             |
+----------------------------------------------------------+
| VERDICT: CLOSED                                          |
+----------------------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.933)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 48.5% | 45.6% | 46.1% | no | no |
| ETH/USDT | 48.3% | 52.7% | 44.8% | no | no |

### candlestick: hammer (+1)

```
+----------------------------------------------------------+
| candlestick: hammer (+1)                                 |
+----------------------------------------------------------+
| Horizon:          t+1                                    |
| Occurrences:      113                                    |
|   BTC/USDT     56                                        |
|   ETH/USDT     57                                        |
+----------------------------------------------------------+
| Hit rate (all):   39.8%                                  |
| Hit rate (BTC):   41.1%                                  |
| Hit rate (ETH):   38.6%                                  |
| Mean fwd return:  -0.1011%                               |
+----------------------------------------------------------+
| DM statistic:     -2.412                                 |
| p-value:          0.9921                                 |
+----------------------------------------------------------+
| Temporal windows (pooled events):                        |
|   Older:   36.8%                                         |
|   Middle:  52.6%                                         |
|   Recent:  29.7%                                         |
+----------------------------------------------------------+
| GATES:                                                   |
| G1 (hit > 55%, both):   FAIL                             |
| G2 (DM p < 0.05):       FAIL                             |
| G3 (both assets > 50%): FAIL                             |
| G4 (stability):         FAIL                             |
| G5 (no degradation):    PASS                             |
| G6 (n >= 30 per asset): PASS                             |
+----------------------------------------------------------+
| VERDICT: CLOSED                                          |
+----------------------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.992)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 50.0% | 46.4% | 25.0% | no | yes |
| ETH/USDT | 31.2% | 52.6% | 31.8% | no | no |

### volume: volume_spike (+1/-1)

```
+----------------------------------------------------------+
| volume: volume_spike (+1/-1)                             |
+----------------------------------------------------------+
| Horizon:          t+1                                    |
| Occurrences:      669                                    |
|   BTC/USDT     375                                       |
|   ETH/USDT     294                                       |
+----------------------------------------------------------+
| Hit rate (all):   49.0%                                  |
| Hit rate (BTC):   48.0%                                  |
| Hit rate (ETH):   50.3%                                  |
| Mean fwd return:  +0.0537%                               |
+----------------------------------------------------------+
| DM statistic:     -0.470                                 |
| p-value:          0.6807                                 |
+----------------------------------------------------------+
| Temporal windows (pooled events):                        |
|   Older:   52.0%                                         |
|   Middle:  46.6%                                         |
|   Recent:  48.4%                                         |
+----------------------------------------------------------+
| GATES:                                                   |
| G1 (hit > 55%, both):   FAIL                             |
| G2 (DM p < 0.05):       FAIL                             |
| G3 (both assets > 50%): FAIL                             |
| G4 (stability):         FAIL                             |
| G5 (no degradation):    PASS                             |
| G6 (n >= 30 per asset): PASS                             |
+----------------------------------------------------------+
| VERDICT: CLOSED                                          |
+----------------------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.681)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 53.2% | 45.0% | 45.5% | no | no |
| ETH/USDT | 51.7% | 48.1% | 50.8% | no | no |

### time_of_day: best-hours (out-of-sample)

```
+----------------------------------------------------------+
| time_of_day: best-hours (out-of-sample)                  |
+----------------------------------------------------------+
| Horizon:          t+1                                    |
| Occurrences:      0                                      |
|   BTC/USDT     0                                         |
|   ETH/USDT     0                                         |
+----------------------------------------------------------+
| STATUS: SKIPPED (< 50 occurrences)                       |
+----------------------------------------------------------+
```

**SKIPPED** — only 0 out-of-sample occurrences (< 50); no hour cleared the selection filter on the training half, or the test window is too short (sandbox rule 7)

## Summary

| Signal | N | Hit rate | DM p | Verdict |
|---|---|---|---|---|
| momentum: higher_high_higher_low (+1) | 500 | 49.0% | 0.6589 | CLOSED |
| momentum: lower_low_lower_high (-1) | 436 | 52.1% | 0.2025 | CLOSED |
| momentum: combined HH/HL + LL/LH | 936 | 50.4% | 0.4059 | CLOSED |
| candlestick: bullish_engulfing (+1) | 637 | 48.0% | 0.8238 | CLOSED |
| candlestick: bearish_engulfing (-1) | 825 | 45.7% | 0.9899 | CLOSED |
| candlestick: doji (+1, tested as long) | 999 | 47.6% | 0.9331 | CLOSED |
| candlestick: hammer (+1) | 113 | 39.8% | 0.9921 | CLOSED |
| volume: volume_spike (+1/-1) | 669 | 49.0% | 0.6807 | CLOSED |
| time_of_day: best-hours (out-of-sample) | 0 | — | — | SKIPPED (<50) |

**No signal cleared all six gates.** Every pattern tested above is reported as CLOSED. This is the expected outcome for simple public patterns on liquid 4h crypto data and is documented here rather than buried — negative results are results.
