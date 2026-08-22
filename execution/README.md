# Execution Layer — Paper-Only CCXT Pipeline (logging only)

A **paper-only** execution pipeline, ready to receive a directional signal and
place sandbox orders, but **not yet connected to the live HAR bot**. For now
it only logs. `sandbox=True` is enforced everywhere — there is no code path
that can reach a live exchange endpoint.

## Hard safety invariant

* `ExchangeConfig.sandbox` defaults to `True`.
* Constructing an `ExchangeClient` with `sandbox=False` raises
  `SandboxViolation` (a `ValueError`) immediately.
* The CCXT instance is put into sandbox mode with `set_sandbox_mode(True)`.
* **Do not** wire this layer to the live HAR bot until Phase 9A gates pass and
  live trading is explicitly approved. This module deliberately does not
  import or call anything in `kronos_trading`.

## Modules

| File | Responsibility |
|------|----------------|
| `exchange_client.py` | Defensive CCXT wrapper. Every method is fail-soft (returns `None` / `False`, logs with a UTC timestamp, never raises). Sandbox-enforced. |
| `order_manager.py` | HAR-volatility-targeted position sizing (10% cap, $10 minimum), order construction, and signal execution. |
| `position_tracker.py` | Local-SQLite paper position book with realized-PnL on close. **Separate from Supabase.** |
| `execution_logger.py` | Append-only audit log of signals / order attempts / results / skips. **Separate from `har_predictions`.** |

## Position sizing (pre-registered, fixed)

```
har_vol      = har_predicted_range / current_price   # fractional move
notional     = (account_size * target_vol) / har_vol # vol-targeted $ notional
max_notional = account_size * 0.10                    # hard 10%-of-account cap
notional     = min(notional, max_notional)
size (base)  = notional / current_price
```

A trade is **skipped and logged** when: direction is `0`, HAR volatility is
non-positive (needs `har_predicted_range > 0` and `price > 0`), or the
notional is below the **$10** minimum.

## Quick example

```python
from execution.exchange_client import ExchangeClient, ExchangeConfig
from execution.order_manager import SignalInput, execute_signal
from execution.position_tracker import PositionTracker
from execution.execution_logger import ExecutionLogger

cfg = ExchangeConfig(api_key="...", api_secret="...", api_password="...",
                     sandbox=True)          # paper/testnet
client = ExchangeClient(cfg)
client.connect()

logger = ExecutionLogger()
tracker = PositionTracker()

signal = SignalInput(
    timestamp="2024-01-15T14:00:00Z", asset="BTC/USDT",
    direction=1, har_predicted_range=200.0, confidence=0.6, regime="high",
)
order = execute_signal(signal, client, account_size=10_000.0, execution_logger=logger)
if order is not None:
    tracker.open_position(...)   # caller builds OrderParams from build_order_params
```

## What the output means

* `place_market_order` returns the CCXT order dict on success, `None` on any
  error (logged, never raised).
* `PositionTracker.compute_paper_pnl()` returns
  `{realized_pnl, n_closed, n_open, avg_realized_pnl, win_rate}` — realized
  PnL summed over closed positions; open positions are not marked-to-market.
  Long PnL = `(exit - entry) * size`; short PnL = `(entry - exit) * size`.
* `ExecutionLogger.get_execution_log()` returns the full decision trail
  (`signal` / `order_attempt` / `order_result` / `skip`), JSON-decoded.

## Storage locations (local SQLite, not Supabase)

* `execution/paper_positions.db` — paper position book.
* `execution/execution_log.db` — execution audit log.

Both are configurable via the constructor `db_path=` argument (tests use
`tmp_path`). Neither is the `har_predictions` research store.

## Tests

```bash
python -m pytest execution/ -q
```

All tests use an in-memory fake exchange and `tmp_path` SQLite — no network,
no real keys, no live endpoints.
