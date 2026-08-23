> **WARNING — THIS IS NOT A RESEARCH RESULT.**
>
> This file is a *pipeline smoke test* produced from a seeded synthetic random walk
> (`tools/make_synthetic_candles.py`), because the machine it was generated on has no
> network egress to `api.kucoin.com`. The numbers below describe random data and say
> nothing about BTC or ETH. Re-run the CLI with live KuCoin data to get real results.

# Pattern Research Results — all

_Generated:_ 2026-08-23T01:03:34Z  
_Source:_ local CSV file(s) — verify their provenance, 1h OHLCV, last 730 days  
_Assets:_ both (BTC/USDT + ETH/USDT)  
_Horizon:_ t+1  
_Sandbox:_ `sandbox/pattern_research` — no DB, no secrets, no contact with the production system.

## Data

| Asset | Bars | First bar (UTC) | Last bar (UTC) |
|---|---|---|---|
| BTC/USDT | 17520 | 2024-01-01 00:00:00+00:00 | 2025-12-30 23:00:00+00:00 |
| ETH/USDT | 17520 | 2024-01-01 00:00:00+00:00 | 2025-12-30 23:00:00+00:00 |

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
| 16 | +0.0253% | 53.3% | 730 |
| 03 | +0.0039% | 52.7% | 730 |
| 12 | +0.0093% | 52.7% | 730 |
| 15 | +0.0299% | 51.8% | 730 |
| 18 | +0.0092% | 50.8% | 730 |
| 21 | +0.0066% | 50.8% | 730 |
| 17 | +0.0006% | 50.3% | 730 |
| 22 | +0.0053% | 50.3% | 730 |
| 23 | +0.0153% | 50.3% | 730 |
| 04 | +0.0053% | 50.1% | 730 |
| 01 | -0.0039% | 50.0% | 730 |
| 02 | +0.0114% | 49.3% | 730 |
| 09 | -0.0096% | 49.2% | 730 |
| 08 | -0.0094% | 49.2% | 730 |
| 13 | -0.0147% | 49.2% | 730 |
| 07 | -0.0272% | 48.9% | 730 |
| 11 | -0.0234% | 48.8% | 730 |
| 14 | -0.0218% | 47.7% | 730 |
| 05 | -0.0101% | 47.5% | 730 |
| 20 | -0.0268% | 47.0% | 730 |
| 19 | -0.0157% | 46.8% | 730 |
| 06 | -0.0092% | 46.7% | 730 |
| 10 | -0.0161% | 46.7% | 730 |
| 00 | -0.0177% | 46.6% | 729 |

### BTC/USDT — day-of-week bias (full sample)

| Day | Mean return | Win rate | N |
|---|---|---|---|
| Mon | +0.0123% | 50.3% | 2519 |
| Tue | -0.0117% | 49.6% | 2520 |
| Wed | -0.0078% | 48.8% | 2496 |
| Thu | -0.0115% | 47.4% | 2496 |
| Fri | -0.0119% | 49.4% | 2496 |
| Sat | +0.0064% | 50.6% | 2496 |
| Sun | -0.0001% | 50.1% | 2496 |

Best hours learned on the training half (bars up to 2024-12-30 23:00:00+00:00, n=8760): `[]` — evaluated only on the held-out bars from 2024-12-31 00:00:00+00:00 onwards (n=8760).

### ETH/USDT — hourly bias (full sample, sorted by win rate)

| Hour (UTC) | Mean return | Win rate | N |
|---|---|---|---|
| 22 | +0.0233% | 54.1% | 730 |
| 00 | +0.0388% | 52.9% | 729 |
| 04 | +0.0206% | 52.7% | 730 |
| 14 | +0.0105% | 52.5% | 730 |
| 08 | +0.0193% | 52.3% | 730 |
| 07 | +0.0177% | 52.2% | 730 |
| 03 | +0.0196% | 51.5% | 730 |
| 16 | +0.0029% | 51.2% | 730 |
| 15 | -0.0022% | 50.7% | 730 |
| 10 | -0.0004% | 50.7% | 730 |
| 06 | -0.0015% | 50.5% | 730 |
| 23 | -0.0042% | 50.4% | 730 |
| 17 | +0.0213% | 50.3% | 730 |
| 13 | +0.0145% | 50.3% | 730 |
| 05 | +0.0035% | 50.1% | 730 |
| 19 | +0.0013% | 50.0% | 730 |
| 09 | +0.0082% | 49.5% | 730 |
| 12 | -0.0049% | 49.2% | 730 |
| 20 | -0.0115% | 49.0% | 730 |
| 21 | -0.0025% | 48.9% | 730 |
| 11 | -0.0031% | 48.6% | 730 |
| 18 | -0.0170% | 48.4% | 730 |
| 01 | -0.0303% | 47.8% | 730 |
| 02 | -0.0190% | 46.8% | 730 |

### ETH/USDT — day-of-week bias (full sample)

| Day | Mean return | Win rate | N |
|---|---|---|---|
| Mon | -0.0017% | 49.5% | 2519 |
| Tue | +0.0040% | 49.6% | 2520 |
| Wed | +0.0070% | 51.6% | 2496 |
| Thu | -0.0014% | 50.3% | 2496 |
| Fri | -0.0027% | 50.1% | 2496 |
| Sat | +0.0087% | 50.4% | 2496 |
| Sun | +0.0169% | 51.6% | 2496 |

Best hours learned on the training half (bars up to 2024-12-30 23:00:00+00:00, n=8760): `[]` — evaluated only on the held-out bars from 2024-12-31 00:00:00+00:00 onwards (n=8760).

## Results

### momentum: higher_high_higher_low (+1)

```
+--------------------------------------------+
| momentum: higher_high_higher_low (+1)      |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      998                      |
|   BTC/USDT     487                         |
|   ETH/USDT     511                         |
+--------------------------------------------+
| Hit rate (all):   47.2%                    |
| Hit rate (BTC):   45.0%                    |
| Hit rate (ETH):   49.3%                    |
| Mean fwd return:  -0.0224%                 |
+--------------------------------------------+
| DM statistic:     -1.747                   |
| p-value:          0.9596                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   44.7%                           |
|   Middle:  50.5%                           |
|   Recent:  46.4%                           |
+--------------------------------------------+
| GATES:                                     |
| G1 (hit > 55%, both):   FAIL               |
| G2 (DM p < 0.05):       FAIL               |
| G3 (both assets > 50%): FAIL               |
| G4 (stability):         FAIL               |
| G5 (no degradation):    PASS               |
| G6 (n >= 30 per asset): PASS               |
+--------------------------------------------+
| VERDICT: CLOSED                            |
+--------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.96)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 42.9% | 47.3% | 44.7% | no | no |
| ETH/USDT | 46.7% | 52.9% | 48.3% | no | no |

### momentum: lower_low_lower_high (-1)

```
+--------------------------------------------+
| momentum: lower_low_lower_high (-1)        |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      1046                     |
|   BTC/USDT     543                         |
|   ETH/USDT     503                         |
+--------------------------------------------+
| Hit rate (all):   50.6%                    |
| Hit rate (BTC):   51.7%                    |
| Hit rate (ETH):   49.3%                    |
| Mean fwd return:  -0.0053%                 |
+--------------------------------------------+
| DM statistic:     0.378                    |
| p-value:          0.3528                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   50.4%                           |
|   Middle:  49.6%                           |
|   Recent:  51.7%                           |
+--------------------------------------------+
| GATES:                                     |
| G1 (hit > 55%, both):   FAIL               |
| G2 (DM p < 0.05):       FAIL               |
| G3 (both assets > 50%): FAIL               |
| G4 (stability):         FAIL               |
| G5 (no degradation):    PASS               |
| G6 (n >= 30 per asset): PASS               |
+--------------------------------------------+
| VERDICT: CLOSED                            |
+--------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.353)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 53.1% | 51.6% | 50.6% | yes | no |
| ETH/USDT | 46.3% | 50.6% | 51.2% | no | no |

### momentum: combined HH/HL + LL/LH

```
+--------------------------------------------+
| momentum: combined HH/HL + LL/LH           |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      2044                     |
|   BTC/USDT     1030                        |
|   ETH/USDT     1014                        |
+--------------------------------------------+
| Hit rate (all):   48.9%                    |
| Hit rate (BTC):   48.5%                    |
| Hit rate (ETH):   49.3%                    |
| Mean fwd return:  -0.0136%                 |
+--------------------------------------------+
| DM statistic:     -0.988                   |
| p-value:          0.8384                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   47.5%                           |
|   Middle:  50.1%                           |
|   Recent:  49.2%                           |
+--------------------------------------------+
| GATES:                                     |
| G1 (hit > 55%, both):   FAIL               |
| G2 (DM p < 0.05):       FAIL               |
| G3 (both assets > 50%): FAIL               |
| G4 (stability):         FAIL               |
| G5 (no degradation):    PASS               |
| G6 (n >= 30 per asset): PASS               |
+--------------------------------------------+
| VERDICT: CLOSED                            |
+--------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.838)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 48.1% | 49.6% | 47.9% | no | no |
| ETH/USDT | 46.5% | 51.8% | 49.7% | no | no |

### candlestick: bullish_engulfing (+1)

```
+--------------------------------------------+
| candlestick: bullish_engulfing (+1)        |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      4435                     |
|   BTC/USDT     2227                        |
|   ETH/USDT     2208                        |
+--------------------------------------------+
| Hit rate (all):   49.5%                    |
| Hit rate (BTC):   49.6%                    |
| Hit rate (ETH):   49.4%                    |
| Mean fwd return:  -0.0024%                 |
+--------------------------------------------+
| DM statistic:     -0.644                   |
| p-value:          0.7402                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   49.0%                           |
|   Middle:  51.0%                           |
|   Recent:  48.5%                           |
+--------------------------------------------+
| GATES:                                     |
| G1 (hit > 55%, both):   FAIL               |
| G2 (DM p < 0.05):       FAIL               |
| G3 (both assets > 50%): FAIL               |
| G4 (stability):         FAIL               |
| G5 (no degradation):    PASS               |
| G6 (n >= 30 per asset): PASS               |
+--------------------------------------------+
| VERDICT: CLOSED                            |
+--------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.74)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 48.8% | 51.6% | 48.5% | no | no |
| ETH/USDT | 49.3% | 50.6% | 48.4% | no | no |

### candlestick: bearish_engulfing (-1)

```
+--------------------------------------------+
| candlestick: bearish_engulfing (-1)        |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      4377                     |
|   BTC/USDT     2224                        |
|   ETH/USDT     2153                        |
+--------------------------------------------+
| Hit rate (all):   50.0%                    |
| Hit rate (BTC):   50.4%                    |
| Hit rate (ETH):   49.6%                    |
| Mean fwd return:  -0.0037%                 |
+--------------------------------------------+
| DM statistic:     0.047                    |
| p-value:          0.4813                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   50.4%                           |
|   Middle:  49.5%                           |
|   Recent:  50.2%                           |
+--------------------------------------------+
| GATES:                                     |
| G1 (hit > 55%, both):   FAIL               |
| G2 (DM p < 0.05):       FAIL               |
| G3 (both assets > 50%): FAIL               |
| G4 (stability):         FAIL               |
| G5 (no degradation):    PASS               |
| G6 (n >= 30 per asset): PASS               |
+--------------------------------------------+
| VERDICT: CLOSED                            |
+--------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.481)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 50.4% | 50.6% | 50.3% | yes | no |
| ETH/USDT | 50.0% | 48.9% | 50.0% | no | no |

### candlestick: doji (+1, tested as long)

```
+--------------------------------------------+
| candlestick: doji (+1, tested as long)     |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      4927                     |
|   BTC/USDT     2456                        |
|   ETH/USDT     2471                        |
+--------------------------------------------+
| Hit rate (all):   49.1%                    |
| Hit rate (BTC):   49.0%                    |
| Hit rate (ETH):   49.2%                    |
| Mean fwd return:  -0.0041%                 |
+--------------------------------------------+
| DM statistic:     -1.228                   |
| p-value:          0.8902                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   49.7%                           |
|   Middle:  50.2%                           |
|   Recent:  47.4%                           |
+--------------------------------------------+
| GATES:                                     |
| G1 (hit > 55%, both):   FAIL               |
| G2 (DM p < 0.05):       FAIL               |
| G3 (both assets > 50%): FAIL               |
| G4 (stability):         FAIL               |
| G5 (no degradation):    PASS               |
| G6 (n >= 30 per asset): PASS               |
+--------------------------------------------+
| VERDICT: CLOSED                            |
+--------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.89)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 49.4% | 49.6% | 48.1% | no | no |
| ETH/USDT | 49.8% | 50.6% | 47.2% | no | no |

### candlestick: hammer (+1)

```
+--------------------------------------------+
| candlestick: hammer (+1)                   |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      604                      |
|   BTC/USDT     314                         |
|   ETH/USDT     290                         |
+--------------------------------------------+
| Hit rate (all):   49.7%                    |
| Hit rate (BTC):   48.4%                    |
| Hit rate (ETH):   51.0%                    |
| Mean fwd return:  -0.0097%                 |
+--------------------------------------------+
| DM statistic:     -0.166                   |
| p-value:          0.5658                   |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   48.0%                           |
|   Middle:  48.3%                           |
|   Recent:  52.7%                           |
+--------------------------------------------+
| GATES:                                     |
| G1 (hit > 55%, both):   FAIL               |
| G2 (DM p < 0.05):       FAIL               |
| G3 (both assets > 50%): FAIL               |
| G4 (stability):         FAIL               |
| G5 (no degradation):    PASS               |
| G6 (n >= 30 per asset): PASS               |
+--------------------------------------------+
| VERDICT: CLOSED                            |
+--------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.566)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 45.1% | 50.0% | 50.5% | no | no |
| ETH/USDT | 50.5% | 47.4% | 55.2% | no | no |

### volume: volume_spike (+1/-1)

```
+--------------------------------------------+
| volume: volume_spike (+1/-1)               |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      2383                     |
|   BTC/USDT     1185                        |
|   ETH/USDT     1198                        |
+--------------------------------------------+
| Hit rate (all):   49.9%                    |
| Hit rate (BTC):   50.2%                    |
| Hit rate (ETH):   49.5%                    |
| Mean fwd return:  +0.0099%                 |
+--------------------------------------------+
| DM statistic:     -0.143                   |
| p-value:          0.557                    |
+--------------------------------------------+
| Temporal windows (pooled events):          |
|   Older:   50.7%                           |
|   Middle:  50.3%                           |
|   Recent:  48.6%                           |
+--------------------------------------------+
| GATES:                                     |
| G1 (hit > 55%, both):   FAIL               |
| G2 (DM p < 0.05):       FAIL               |
| G3 (both assets > 50%): FAIL               |
| G4 (stability):         FAIL               |
| G5 (no degradation):    PASS               |
| G6 (n >= 30 per asset): PASS               |
+--------------------------------------------+
| VERDICT: CLOSED                            |
+--------------------------------------------+
```

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.557)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 50.0% | 52.5% | 47.6% | no | no |
| ETH/USDT | 51.1% | 48.5% | 48.6% | no | no |

### time_of_day: best-hours (out-of-sample)

```
+--------------------------------------------+
| time_of_day: best-hours (out-of-sample)    |
+--------------------------------------------+
| Horizon:          t+1                      |
| Occurrences:      0                        |
|   BTC/USDT     0                           |
|   ETH/USDT     0                           |
+--------------------------------------------+
| STATUS: SKIPPED (< 50 occurrences)         |
+--------------------------------------------+
```

**SKIPPED** — only 0 out-of-sample occurrences (< 50); no hour cleared the selection filter on the training half, or the test window is too short (sandbox rule 7)

## Summary

| Signal | N | Hit rate | DM p | Verdict |
|---|---|---|---|---|
| momentum: higher_high_higher_low (+1) | 998 | 47.2% | 0.9596 | CLOSED |
| momentum: lower_low_lower_high (-1) | 1046 | 50.6% | 0.3528 | CLOSED |
| momentum: combined HH/HL + LL/LH | 2044 | 48.9% | 0.8384 | CLOSED |
| candlestick: bullish_engulfing (+1) | 4435 | 49.5% | 0.7402 | CLOSED |
| candlestick: bearish_engulfing (-1) | 4377 | 50.0% | 0.4813 | CLOSED |
| candlestick: doji (+1, tested as long) | 4927 | 49.1% | 0.8902 | CLOSED |
| candlestick: hammer (+1) | 604 | 49.7% | 0.5658 | CLOSED |
| volume: volume_spike (+1/-1) | 2383 | 49.9% | 0.557 | CLOSED |
| time_of_day: best-hours (out-of-sample) | 0 | — | — | SKIPPED (<50) |

**No signal cleared all six gates.** Every pattern tested above is reported as CLOSED. This is the expected outcome for simple public patterns on liquid 1h crypto data and is documented here rather than buried — negative results are results.
