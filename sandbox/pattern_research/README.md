# Agent 1 — Pattern Research Sandbox

A **completely separate** research playground for testing simple, public trading
patterns on 1h crypto data.

* No database. No Supabase. No secrets. No `.env`.
* Data source: **CCXT KuCoin public API** only (same client + sanitising rules as
  `kronos_trading/alerts/har_forecaster.py`).
* Nothing in `kronos_trading/`, `execution/`, `phase9a/` or the dashboards imports
  anything from here, and nothing here imports the production system.
  (`tests/test_data_loader.py::test_sandbox_imports_no_production_or_db_modules`
  enforces that automatically.)

```
sandbox/pattern_research/
├── data_loader.py            # KuCoin public fetch (730d, 1h), CSV cache, offline CSV mode
├── patterns/
│   ├── momentum.py           # Phase 9B — HH/HL, LL/LH, compute_forward_return
│   ├── candlestick.py        # Phase 9C — engulfing, doji, hammer
│   ├── time_of_day.py        # Phase 9D — hourly / daily bias, find_best_hours
│   └── volume_spike.py       # Phase 9E — volume ratio + spike direction
├── validator.py              # DM test, G1–G6 gates, walk-forward
├── run_pattern_research.py   # CLI runner → markdown report
├── tools/make_synthetic_candles.py  # offline smoke-test data (NOT research data)
├── results/                  # generated reports
├── cache/                    # fetched OHLCV CSVs (git-ignored)
└── tests/                    # 75 unit tests, fully offline
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
```

CLI flags: `--pattern {momentum,candlestick,time,volume,all}`,
`--asset {BTC/USDT,ETH/USDT,both}`, `--horizon {1,2,3}`, `--output DIR`,
plus `--days` (default 730), `--timeframe` (default `1h`), `--cache-dir`,
`--no-cache`, `--min-win-rate`, `--splits`, `--quiet`, and
`--csv "BTC/USDT=a.csv,ETH/USDT=b.csv"` for fully offline runs.

The first live run fetches ~17,520 hourly bars per asset (paginated, rate-limited)
and caches them in `cache/`; later runs are instant and hit no API.

Tests:

```bash
pytest sandbox/pattern_research/tests -q     # 75 passed, no network needed
```

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
| `compute_forward_return(candles, signal_series, horizon=1)` | joins signals to realised returns | DataFrame `signal, forward_return, correct` |

`correct = 1` when `sign(forward_return) == sign(signal)`, else `0`. Rows where the
exit bar falls outside the sample are dropped; `signal == 0` rows are excluded by
default (`include_flat=True` keeps them).

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

The runner writes `results/pattern_research_<pattern>_<asset>_h<horizon>_<utc>.md`
containing: data coverage, the method/no-look-ahead statement, the descriptive
hourly and day-of-week bias tables, a Phase 9A-style results box per signal
(hit rates, mean forward return, DM stat, p-value, temporal thirds, G1–G6,
verdict), the per-asset walk-forward table, and a summary table listing **every**
signal including failures and skips. When nothing passes, the report says so
explicitly — negative results are results (rule 5).

### Status of the checked-in report

`results/SMOKE_TEST_SYNTHETIC_DATA.md` is a **pipeline smoke test on seeded
synthetic random-walk data**, not a research result. It exists because the
environment this sandbox was built in has no network egress to `api.kucoin.com`
(TLS connections are refused), so no live BTC/ETH data could be downloaded here.
As expected, random data produces `CLOSED` for every pattern.

To produce real results, run the CLI on a machine with public internet access:

```bash
python -m sandbox.pattern_research.run_pattern_research \
    --pattern all --asset both --horizon 1
```
