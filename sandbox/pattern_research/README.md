# Agent 1 — Pattern Research Sandbox

A **completely separate** research playground for testing simple, public trading
patterns on crypto data (1h / 4h / 1d bars).

* No database. No Supabase. No secrets. No `.env`.
* Data source: **CCXT KuCoin public API** only (same client + sanitising rules as
  `kronos_trading/alerts/har_forecaster.py`).
* Nothing in `kronos_trading/`, `execution/`, `phase9a/` or the dashboards imports
  anything from here, and nothing here imports the production system.
  (`tests/test_data_loader.py::test_sandbox_imports_no_production_or_db_modules`
  enforces that automatically.)

```
sandbox/pattern_research/
├── data_loader.py            # KuCoin public fetch (730d, 1h/4h/1d), CSV cache, offline CSV mode
├── patterns/
│   ├── momentum.py           # Phase 9B — HH/HL, LL/LH, fade inverse, compute_forward_return
│   ├── candlestick.py        # Phase 9C — engulfing, doji, hammer
│   ├── time_of_day.py        # Phase 9D — hourly / daily bias, find_best_hours
│   └── volume_spike.py       # Phase 9E — volume ratio + spike direction
├── validator.py              # DM test, G1–G6 gates, walk-forward
├── run_pattern_research.py   # CLI runner → markdown report
├── tools/make_synthetic_candles.py  # offline smoke-test data (NOT research data)
├── results/                  # generated reports
├── cache/                    # fetched OHLCV CSVs (git-ignored)
└── tests/                    # 92 unit tests, fully offline
```

---

## Quick start

```bash
pip install pandas numpy scipy ccxt pytest

# everything, both assets, t+1
python -m sandbox.pattern_research.run_pattern_research \
    --pattern all --asset both --horizon 1 \
    --output sandbox/pattern_research/results

# one family, one asset, longer horizon
python -m sandbox.pattern_research.run_pattern_research \
    --pattern candlestick --asset BTC/USDT --horizon 3

# fade (mean-reversion) reading of Pattern 1 — deliberately NOT in "all"
python -m sandbox.pattern_research.run_pattern_research \
    --pattern momentum_fade --asset both --horizon 1 --timeframe 4h
```

CLI flags: `--pattern {momentum,momentum_fade,candlestick,time,volume,all}`,
`--asset {BTC/USDT,ETH/USDT,both}`, `--horizon {1,2,3}`,
`--timeframe {1h,4h,1d}` (default `1h`), `--output DIR`,
plus `--days` (default 730), `--cache-dir`, `--no-cache`, `--min-win-rate`,
`--splits`, `--quiet`, and
`--csv "BTC/USDT=a.csv,ETH/USDT=b.csv"` for fully offline runs.

The first live run fetches ~17,520 hourly bars per asset (paginated, rate-limited)
and caches them in `cache/`; later runs are instant and hit no API.

Tests:

```bash
pytest sandbox/pattern_research/tests -q     # 92 passed, no network needed
```

---

## Timeframes

The loader and runner support **1h, 4h and 1d bars** (`--timeframe`). Things
worth knowing before comparing runs across timeframes:

* **Cache naming** — the timeframe is part of the cache filename
  (`BTCUSDT_1h_730d.csv`, `BTCUSDT_4h_730d.csv`, `BTCUSDT_1d_730d.csv`), so
  two timeframes of the same asset never share a cache file.
* **Horizons are counted in bars, not clock time** — `--timeframe 4h
  --horizon 1` means *4 hours forward* and `--timeframe 1d --horizon 1` means
  *1 day forward*. The report's `_Horizon:_` line spells this out
  (`t+2 = 8 hours forward`).
* **Fewer bars on 4h/1d** — the same 730-day window holds ~6× fewer 4h bars
  and ~24× fewer 1d bars than 1h bars. Rare patterns (hammer, engulfing,
  time-of-day selections) can therefore fall under the 50-occurrence floor and
  get **SKIPPED instead of tested** (rule 6/7). A skip on 4h/1d is a sample-size
  statement, not a verdict.
* **CSV spacing warning** — a `--csv` file can be saved at a different bar
  spacing than `--timeframe` claims (a live fetch cannot). The runner detects
  this via `infer_timeframe` (median bar spacing, 1% tolerance), logs a
  warning, and prints a `> **Warning:**` block in the report saying horizons
  are counted in **bars of the loaded data**. The synthetic-data tool gained a
  `--freq` flag so genuine 4h/1d bars can be generated offline.

---

## No look-ahead — how it is guaranteed

Every detector follows one convention:

```
pattern completes at bar t-1  →  signal[t] = ±1  →  entry at close[t]
                              →  exit at close[t + horizon]
```

1. The raw condition at bar `t` only reads bars `t, t-1, …` — never `t+1`.
2. The returned series is then `.shift(1)`-ed, so `signal[t]` is fully knowable
   *before bar `t` even opens* (rule 2: all patterns use `.shift(1)` minimum).
3. `compute_forward_return` measures `close[t+horizon]/close[t] - 1`, i.e. the
   trade starts one bar **after** the information that produced it. This is
   deliberately conservative — it throws away the move of the pattern bar itself
   rather than risk any leakage.
4. Two tests assert this empirically: truncating the future must not change any
   earlier signal, and multiplying all future bars by 1.5 must not change any
   past signal.

The time-of-day signal is even stricter: it contains **zero** market data (it is
pure calendar arithmetic — fire on the bar preceding a selected hour), and the
"best hours" are learned on the first 50% of the sample and only ever evaluated
on the held-out remainder. Learning hours in-sample and scoring them in-sample
would be data snooping and is not done.

`data_loader` also drops the still-forming candle (`open_time + bar_len > now`)
exactly like `har_forecaster.fetch_candles`, so the current partial bar can never
enter a feature.

---

## The patterns

### Pattern 1 — Momentum (`patterns/momentum.py`)

| Function | Rule | Output |
|---|---|---|
| `detect_higher_high_higher_low(candles, lookback=3)` | last 3 highs each > previous **and** last 3 lows each > previous | `+1` / `0` |
| `detect_lower_low_lower_high(candles, lookback=3)` | last 3 lows each < previous **and** last 3 highs each < previous | `-1` / `0` |
| `detect_momentum_combined(candles, lookback=3)` | union of the two (mutually exclusive) | `+1/-1/0` |
| `detect_momentum_fade_combined(candles, lookback=3)` | exact inverse of the combined signal (fade / mean-reversion reading) | `-1/+1/0` |
| `compute_forward_return(candles, signal_series, horizon=1)` | joins signals to realised returns | DataFrame `signal, forward_return, correct` |

`correct = 1` when `sign(forward_return) == sign(signal)`, else `0`. Rows where the
exit bar falls outside the sample are dropped; `signal == 0` rows are excluded by
default (`include_flat=True` keeps them).

### Fade (mean-reversion) reading

`detect_momentum_fade_combined` is implemented as the literal negation of
`detect_momentum_combined`, so it inherits the `.shift(1)` timing and the
no-look-ahead guarantees unchanged: HH/HL → `-1` (sell the bullish structure),
LL/LH → `+1` (buy the bearish structure). It runs through the **same**
`evaluate_signal` path — same `compute_forward_return`, same DM test, same
G1–G6 gates, same walk-forward — and is invoked explicitly with
`--pattern momentum_fade`. It is deliberately **not** part of `--pattern all`:
it scores the *same events* as `momentum: combined` with flipped signs, so
bundling the two would let one experiment read as two independent findings.

Two honesty caveats, stated in the docstring and enforced by the tests:

1. **A sub-50% continuation hit rate is NOT evidence of a tradable fade
   edge.** The fade hit rate is essentially the complement of the continuation
   hit rate (they sum to ~1.0 on the same events), so "momentum continues only
   49% of the time" does not imply "fading wins 51% of the time" in any
   tradable sense.
2. **The inverse must clear G1–G6 and the DM test on its own.** On synthetic
   random-walk data the fade reading lands near 50% with p > 0.05 and is
   reported `CLOSED` (see `results/SMOKE_TEST_SYNTHETIC_momentum_fade_1h.md`) —
   which is the expected outcome, not a failure of the experiment.

### Pattern 2 — Candlestick (`patterns/candlestick.py`)

| Function | Rule |
|---|---|
| `detect_bullish_engulfing` | current bullish, previous bearish, current body fully engulfs the previous body (strictly) → `+1` |
| `detect_bearish_engulfing` | mirror image → `-1` |
| `detect_doji(threshold=0.1)` | `abs(close-open) / (high-low) < threshold` → `1` |
| `detect_hammer` | lower shadow > 2×body, upper shadow < 0.5×body, body in the upper 30% of the range → `+1` |

Zero-range and zero-body bars are excluded rather than dividing by zero.
A doji is *non-directional*; it is emitted as `+1`, so what gets tested is
literally "does a doji predict an up move" — stated in every report.

### Pattern 3 — Time of day (`patterns/time_of_day.py`)

| Function | Returns |
|---|---|
| `compute_hourly_bias(candles)` | DataFrame indexed `0-23`: `hour, mean_return, win_rate, n_observations`, sorted by `win_rate` desc |
| `compute_daily_bias(candles)` | DataFrame indexed `0-6` (Mon–Sun): `day, day_name, mean_return, win_rate, n_observations` |
| `find_best_hours(hourly_df, min_win_rate=0.55)` | hours with `win_rate > threshold` **and** `n_observations > 100` |
| `build_hour_signal(candles, hours)` | `+1` on the bar preceding each selected hour |

Bar return convention: `ret[t] = close[t]/close[t-1] - 1`, attributed to the UTC
hour of bar `t`.

### Pattern 4 — Volume spike (`patterns/volume_spike.py`)

| Function | Rule |
|---|---|
| `compute_volume_ratio(candles, window=20)` | `volume[t] / rolling_mean(volume, 20)[t]` |
| `detect_volume_spike(candles, threshold=2.0)` | ratio > threshold **and** bullish body → `+1`; ratio > threshold and bearish body → `-1`; else `0` |

---

## Validator (`validator.py`)

`run_dm_test(actual_directions, predicted_directions)` → `{dm_stat, p_value,
hit_rate, n_obs, conclusion}`.
One-sided Diebold-Mariano against a 50/50 coin flip: loss = `1{actual != predicted}`,
random-benchmark loss = `0.5`, `d = 0.5 - loss`, `DM = mean(d)/(HAC_std(d)/√n)`
with a Newey-West Bartlett kernel and **3 lags**, `p = 1 - Φ(DM)`.
This is the *same* test as Phase 9A — `tests/test_validator.py::
test_dm_matches_phase9a_implementation_exactly` asserts the numbers are identical
to `phase9a.dm_test.compute_dm_statistic` to 1e-12. (It is re-implemented rather
than imported so the sandbox stays standalone.)

`run_gate_checks(results_df)` → `{G1..G6, all_pass, verdict, notes, details}`,
identical criteria to Phase 9A (all-or-nothing; any failure ⇒ `CLOSED`):

| Gate | Criterion |
|---|---|
| G1 | hit rate > 55% overall **and** on both assets |
| G2 | one-sided DM p < 0.05 |
| G3 | hit rate > 50% on **both** BTC and ETH |
| G4 | every chronological third > 50% |
| G5 | recent third not more than 10pp below the older third |
| G6 | ≥ 30 events per asset |

A single-asset run cannot satisfy the cross-asset gates by construction; the
report says so in `notes` instead of quietly relaxing the bar.

`run_walk_forward(candles, signal_func, n_splits=3)` → `{older, middle, recent,
splits, is_stable, degrading}`. The candles are cut into contiguous chronological
blocks and the signal is **re-run inside each block**, so nothing crosses a
boundary.

### Minimum sample size (rules 6 & 7)

`validator.MIN_OCCURRENCES = 50`. If a pattern produces fewer than 50 occurrences
in the sample, the runner **skips** it: no gates are run, and the report prints
`SKIPPED (<50)` together with the exact reason. Below 30 events per asset G6
fails anyway.

---

## Honest reporting

The runner writes `results/pattern_research_<pattern>_<asset>_<tf>_h<horizon>_<utc>.md`
(where `<tf>` is the timeframe, e.g. `1h`/`4h`/`1d`) containing: data coverage
(including a per-asset detected "Bar spacing" column), the timeframe/horizon
wording, the method/no-look-ahead statement, the descriptive
hourly and day-of-week bias tables, a Phase 9A-style results box per signal
(hit rates, mean forward return, DM stat, p-value, temporal thirds, G1–G6,
verdict), the per-asset walk-forward table, and a summary table listing **every**
signal including failures and skips. When nothing passes, the report says so
explicitly — negative results are results (rule 5).

### Status of the checked-in reports

`results/SMOKE_TEST_SYNTHETIC_DATA.md` is a **pipeline smoke test on seeded
synthetic random-walk data**, not a research result. It exists because the
environment this sandbox was built in has no network egress to `api.kucoin.com`
(TLS connections are refused), so no live BTC/ETH data could be downloaded here.
As expected, random data produces `CLOSED` for every pattern. Two siblings exist
for the same purpose: `SMOKE_TEST_SYNTHETIC_momentum_fade_1h.md` (the fade
reading on 1h synthetic bars — near 50%, p > 0.05, `CLOSED`) and
`SMOKE_TEST_SYNTHETIC_all_4h.md` (all families on genuine 4h synthetic bars via
`--freq 4h`, demonstrating the fewer-bars → SKIPPED behaviour).

To produce real results, run the CLI on a machine with public internet access:

```bash
python -m sandbox.pattern_research.run_pattern_research \
    --pattern all --asset both --horizon 1
```
