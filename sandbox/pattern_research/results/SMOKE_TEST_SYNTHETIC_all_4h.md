> **WARNING — THIS IS NOT A RESEARCH RESULT.**
>
> This file is a *pipeline smoke test* produced from a seeded synthetic random walk
> (`tools/make_synthetic_candles.py`), because the machine it was generated on has no
> network egress to `api.kucoin.com`. The numbers below describe random data and say
> nothing about BTC or ETH. Re-run the CLI with live KuCoin data to get real results.
>
> The bars are genuine 4h synthetic bars (`--freq 4h`), so the detected bar spacing
> matches --timeframe 4h and no spacing warning appears. On 4h bars there are ~6x
> fewer bars than 1h, so rare patterns can fall under the 50-occurrence floor and
> get SKIPPED instead of tested.

# Pattern Research Results — all

_Generated:_ 2026-08-23T01:47:45Z  
_Source:_ local CSV file(s) — verify their provenance, 4h OHLCV, last 730 days  
_Assets:_ both (BTC/USDT + ETH/USDT)  
_Timeframe:_ 4h  
_Horizon:_ t+1 = 4 hours forward  
_Sandbox:_ `sandbox/pattern_research` — no DB, no secrets, no contact with the production system.

## Data

| Asset | Bars | Bar spacing | First bar (UTC) | Last bar (UTC) |
|---|---|---|---|---|
| BTC/USDT | 4380 | 4h | 2024-01-01 00:00:00+00:00 | 2025-12-30 20:00:00+00:00 |
| ETH/USDT | 4380 | 4h | 2024-01-01 00:00:00+00:00 | 2025-12-30 20:00:00+00:00 |

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
| 00 | +0.0119% | 52.3% | 729 |
| 12 | +0.0219% | 52.1% | 730 |
| 16 | +0.0060% | 52.1% | 730 |
| 08 | -0.0114% | 50.1% | 730 |
| 04 | +0.0097% | 48.2% | 730 |
| 20 | -0.0218% | 46.3% | 730 |
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
| Mon | -0.0206% | 49.4% | 629 |
| Tue | +0.0202% | 50.0% | 630 |
| Wed | +0.0068% | 52.6% | 624 |
| Thu | +0.0158% | 49.5% | 624 |
| Fri | +0.0043% | 51.9% | 624 |
| Sat | -0.0027% | 47.9% | 624 |
| Sun | -0.0047% | 49.8% | 624 |

Best hours learned on the training half (bars up to 2024-12-30 20:00:00+00:00, n=2190): `[]` — evaluated only on the held-out bars from 2024-12-31 00:00:00+00:00 onwards (n=2190).

### ETH/USDT — hourly bias (full sample, sorted by win rate)

| Hour (UTC) | Mean return | Win rate | N |
|---|---|---|---|
| 20 | -0.0039% | 53.3% | 730 |
| 16 | +0.0098% | 50.8% | 730 |
| 04 | -0.0109% | 50.0% | 730 |
| 00 | -0.0108% | 49.5% | 729 |
| 12 | +0.0096% | 48.9% | 730 |
| 08 | -0.0278% | 47.4% | 730 |
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
| Mon | -0.0169% | 47.9% | 629 |
| Tue | -0.0132% | 49.0% | 630 |
| Wed | +0.0089% | 51.6% | 624 |
| Thu | +0.0072% | 53.7% | 624 |
| Fri | +0.0200% | 52.4% | 624 |
| Sat | -0.0081% | 49.5% | 624 |
| Sun | -0.0373% | 45.8% | 624 |

Best hours learned on the training half (bars up to 2024-12-30 20:00:00+00:00, n=2190): `[]` — evaluated only on the held-out bars from 2024-12-31 00:00:00+00:00 onwards (n=2190).

## Results

### momentum: higher_high_higher_low (+1)

```
+----------------------------------------------------------+
| momentum: higher_high_higher_low (+1)                    |
+----------------------------------------------------------+
| Horizon:          t+1                                    |
| Occurrences:      234                                    |
|   BTC/USDT     111                                       |
|   ETH/USDT     123                                       |
+----------------------------------------------------------+
| Hit rate (all):   50.9%                                  |
| Hit rate (BTC):   52.3%                                  |
| Hit rate (ETH):   49.6%                                  |
| Mean fwd return:  +0.0060%                               |
+----------------------------------------------------------+
| DM statistic:     0.274                                  |
| p-value:          0.3919                                 |
+----------------------------------------------------------+
| Temporal windows (pooled events):                        |
|   Older:   50.0%                                         |
|   Middle:  53.8%                                         |
|   Recent:  48.7%                                         |
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

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.392)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 50.0% | 55.9% | 51.2% | no | no |
| ETH/USDT | 50.0% | 56.1% | 42.9% | no | no |

### momentum: lower_low_lower_high (-1)

```
+----------------------------------------------------------+
| momentum: lower_low_lower_high (-1)                      |
+----------------------------------------------------------+
| Horizon:          t+1                                    |
| Occurrences:      271                                    |
|   BTC/USDT     119                                       |
|   ETH/USDT     152                                       |
+----------------------------------------------------------+
| Hit rate (all):   51.3%                                  |
| Hit rate (BTC):   43.7%                                  |
| Hit rate (ETH):   57.2%                                  |
| Mean fwd return:  +0.0177%                               |
+----------------------------------------------------------+
| DM statistic:     0.418                                  |
| p-value:          0.3379                                 |
+----------------------------------------------------------+
| Temporal windows (pooled events):                        |
|   Older:   45.1%                                         |
|   Middle:  61.1%                                         |
|   Recent:  47.8%                                         |
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

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.338)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 30.0% | 51.2% | 50.0% | no | no |
| ETH/USDT | 58.9% | 63.9% | 42.9% | no | yes |

### momentum: combined HH/HL + LL/LH

```
+----------------------------------------------------------+
| momentum: combined HH/HL + LL/LH                         |
+----------------------------------------------------------+
| Horizon:          t+1                                    |
| Occurrences:      505                                    |
|   BTC/USDT     230                                       |
|   ETH/USDT     275                                       |
+----------------------------------------------------------+
| Hit rate (all):   51.1%                                  |
| Hit rate (BTC):   47.8%                                  |
| Hit rate (ETH):   53.8%                                  |
| Mean fwd return:  +0.0123%                               |
+----------------------------------------------------------+
| DM statistic:     0.476                                  |
| p-value:          0.3172                                 |
+----------------------------------------------------------+
| Temporal windows (pooled events):                        |
|   Older:   47.3%                                         |
|   Middle:  58.9%                                         |
|   Recent:  47.0%                                         |
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

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.317)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 39.5% | 53.2% | 50.6% | no | no |
| ETH/USDT | 55.2% | 60.8% | 42.9% | no | yes |

### candlestick: bullish_engulfing (+1)

```
+----------------------------------------------------------+
| candlestick: bullish_engulfing (+1)                      |
+----------------------------------------------------------+
| Horizon:          t+1                                    |
| Occurrences:      1040                                   |
|   BTC/USDT     528                                       |
|   ETH/USDT     512                                       |
+----------------------------------------------------------+
| Hit rate (all):   49.3%                                  |
| Hit rate (BTC):   48.5%                                  |
| Hit rate (ETH):   50.2%                                  |
| Mean fwd return:  -0.0106%                               |
+----------------------------------------------------------+
| DM statistic:     -0.443                                 |
| p-value:          0.6711                                 |
+----------------------------------------------------------+
| Temporal windows (pooled events):                        |
|   Older:   54.5%                                         |
|   Middle:  44.7%                                         |
|   Recent:  48.8%                                         |
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

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.671)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 50.0% | 43.8% | 51.5% | no | no |
| ETH/USDT | 59.3% | 43.5% | 47.8% | no | yes |

### candlestick: bearish_engulfing (-1)

```
+----------------------------------------------------------+
| candlestick: bearish_engulfing (-1)                      |
+----------------------------------------------------------+
| Horizon:          t+1                                    |
| Occurrences:      1089                                   |
|   BTC/USDT     550                                       |
|   ETH/USDT     539                                       |
+----------------------------------------------------------+
| Hit rate (all):   50.3%                                  |
| Hit rate (BTC):   52.2%                                  |
| Hit rate (ETH):   48.4%                                  |
| Mean fwd return:  +0.0001%                               |
+----------------------------------------------------------+
| DM statistic:     0.223                                  |
| p-value:          0.4117                                 |
+----------------------------------------------------------+
| Temporal windows (pooled events):                        |
|   Older:   50.4%                                         |
|   Middle:  49.9%                                         |
|   Recent:  50.7%                                         |
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

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.412)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 54.1% | 48.7% | 53.7% | no | no |
| ETH/USDT | 45.7% | 51.1% | 48.3% | no | no |

### candlestick: doji (+1, tested as long)

```
+----------------------------------------------------------+
| candlestick: doji (+1, tested as long)                   |
+----------------------------------------------------------+
| Horizon:          t+1                                    |
| Occurrences:      1237                                   |
|   BTC/USDT     614                                       |
|   ETH/USDT     623                                       |
+----------------------------------------------------------+
| Hit rate (all):   48.2%                                  |
| Hit rate (BTC):   50.7%                                  |
| Hit rate (ETH):   45.7%                                  |
| Mean fwd return:  -0.0092%                               |
+----------------------------------------------------------+
| DM statistic:     -1.233                                 |
| p-value:          0.8912                                 |
+----------------------------------------------------------+
| Temporal windows (pooled events):                        |
|   Older:   50.4%                                         |
|   Middle:  47.1%                                         |
|   Recent:  47.1%                                         |
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

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.891)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 54.9% | 47.1% | 50.2% | no | no |
| ETH/USDT | 47.0% | 46.1% | 44.2% | no | no |

### candlestick: hammer (+1)

```
+----------------------------------------------------------+
| candlestick: hammer (+1)                                 |
+----------------------------------------------------------+
| Horizon:          t+1                                    |
| Occurrences:      160                                    |
|   BTC/USDT     68                                        |
|   ETH/USDT     92                                        |
+----------------------------------------------------------+
| Hit rate (all):   49.4%                                  |
| Hit rate (BTC):   50.0%                                  |
| Hit rate (ETH):   48.9%                                  |
| Mean fwd return:  -0.0090%                               |
+----------------------------------------------------------+
| DM statistic:     -0.187                                 |
| p-value:          0.574                                  |
+----------------------------------------------------------+
| Temporal windows (pooled events):                        |
|   Older:   48.1%                                         |
|   Middle:  41.5%                                         |
|   Recent:  58.5%                                         |
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

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.574)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 50.0% | 40.0% | 59.3% | no | no |
| ETH/USDT | 48.3% | 44.7% | 56.0% | no | no |

### volume: volume_spike (+1/-1)

```
+----------------------------------------------------------+
| volume: volume_spike (+1/-1)                             |
+----------------------------------------------------------+
| Horizon:          t+1                                    |
| Occurrences:      649                                    |
|   BTC/USDT     318                                       |
|   ETH/USDT     331                                       |
+----------------------------------------------------------+
| Hit rate (all):   48.5%                                  |
| Hit rate (BTC):   48.4%                                  |
| Hit rate (ETH):   48.6%                                  |
| Mean fwd return:  -0.0108%                               |
+----------------------------------------------------------+
| DM statistic:     -0.755                                 |
| p-value:          0.775                                  |
+----------------------------------------------------------+
| Temporal windows (pooled events):                        |
|   Older:   50.7%                                         |
|   Middle:  47.7%                                         |
|   Recent:  47.2%                                         |
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

DM conclusion: NO SIGNIFICANT EDGE OVER RANDOM (p=0.775)

Walk-forward (signal re-run inside each chronological third):

| Asset | Older | Middle | Recent | Stable | Degrading |
|---|---|---|---|---|---|
| BTC/USDT | 49.5% | 48.5% | 47.3% | no | no |
| ETH/USDT | 51.8% | 46.8% | 48.2% | no | no |

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
| momentum: higher_high_higher_low (+1) | 234 | 50.9% | 0.3919 | CLOSED |
| momentum: lower_low_lower_high (-1) | 271 | 51.3% | 0.3379 | CLOSED |
| momentum: combined HH/HL + LL/LH | 505 | 51.1% | 0.3172 | CLOSED |
| candlestick: bullish_engulfing (+1) | 1040 | 49.3% | 0.6711 | CLOSED |
| candlestick: bearish_engulfing (-1) | 1089 | 50.3% | 0.4117 | CLOSED |
| candlestick: doji (+1, tested as long) | 1237 | 48.2% | 0.8912 | CLOSED |
| candlestick: hammer (+1) | 160 | 49.4% | 0.574 | CLOSED |
| volume: volume_spike (+1/-1) | 649 | 48.5% | 0.775 | CLOSED |
| time_of_day: best-hours (out-of-sample) | 0 | — | — | SKIPPED (<50) |

**No signal cleared all six gates.** Every pattern tested above is reported as CLOSED. This is the expected outcome for simple public patterns on liquid 4h crypto data and is documented here rather than buried — negative results are results.
