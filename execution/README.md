# Execution Layer — Paper-Only CCXT Pipeline (logging only)

A **paper-only** execution pipeline, ready to receive a directional signal and
place sandbox orders, but **not yet connected to the live HAR bot**. For now it
only logs. `sandbox=True` is enforced everywhere — there is no code path that
can reach a live exchange endpoint.

## Hard safety invariant

* `ExchangeConfig.sandbox` defaults to `True`.
* Constructing an `ExchangeClient` with `sandbox=False` raises
  `SandboxViolation` (a `ValueError`) immediately, before anything else.
* The CCXT instance is created in `__init__` and put into sandbox mode with
  `set_sandbox_mode(True)`.
* Every method is fail-soft: logs the error with a UTC timestamp and returns
  `None` (or `False` for `cancel_order`) instead of raising.

## Modules

| File | Responsibility |
|------|----------------|
| `exchange_client.py` | `ExchangeConfig` + defensive `ExchangeClient` (sandbox-locked). |
| `order_manager.py` | `SignalInput`, `OrderParams` (USD + base size), `compute_position_size`, `build_order_params`, `execute_signal`. |
| `position_tracker.py` | Local-SQLite paper book: `initialize_db`, `open_position`, `close_position`, `get_open_positions`, `get_closed_positions`, `compute_paper_pnl`. |
| `execution_logger.py` | Separate audit log: `log_signal`, `log_skip`, `log_order_attempt`, `log_order_result`, `get_log`. |

## Position sizing (pre-registered, fixed)

```
har_vol   = har_predicted_range / current_price     # fractional move
size_usd  = (account_size * target_vol) / har_vol   # vol-targeted USD
max_size  = account_size * 0.10                      # hard 10%-of-account cap
size_usd  = min(size_usd, max_size)
size_base = size_usd / current_price
```

`compute_position_size` returns the USD notional; it returns `0.0` when
`har_vol ≤ 0` or the resulting size is below the **$10** minimum. Orders are
placed in **base currency** (`size_base`).

## Quick example

```python
from execution.exchange_client import ExchangeClient, ExchangeConfig
from execution.order_manager import SignalInput, execute_signal
from execution.position_tracker import initialize_db, open_position, compute_paper_pnl
from execution import execution_logger as el

cfg = ExchangeConfig(api_key="...", api_secret="...", api_password="...",
                     sandbox=True)          # paper/testnet
client = ExchangeClient(cfg)
client.connect()
initialize_db()                             # execution/paper_positions.db

signal = SignalInput(timestamp="2024-01-15T14:00:00Z", asset="BTC/USDT",
                     direction=1, har_predicted_range=200.0,
                     confidence=0.6, regime="high")
order = execute_signal(signal, client, account_size=10_000.0, current_price=20_000.0)
if order is not None:
    el.log_order_result(order, success=True)
```

## PnL convention

* Long (`direction = +1`): `pnl_usd = (exit − entry) × size_base`
* Short (`direction = −1`): `pnl_usd = (entry − exit) × size_base`
* `pnl_pct = direction × (exit − entry) / entry × 100`

`compute_paper_pnl()` returns `total_trades, open_trades, closed_trades,
total_pnl_usd, win_rate, avg_pnl_pct` over closed positions.

## Storage (local SQLite, not Supabase)

* `execution/paper_positions.db` — paper position book (path set via
  `initialize_db(db_path)`, e.g. `tmp_path` in tests).
* `execution_logger` — in-memory structured log by default; optional plain-text
  file via `configure(log_path)`. `reset()` clears it (test isolation).

## Tests

```bash
python -m pytest execution/ -q
```

All tests use a fake exchange and `tmp_path` SQLite — no network, no real keys.
