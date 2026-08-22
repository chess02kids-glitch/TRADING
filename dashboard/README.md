# HAR Research Platform Dashboard

## What This Is

A live Streamlit dashboard monitoring the
30-day HAR volatility calibration study.

Shows HAR prediction accuracy vs persistence
baseline in real time from Supabase.

## Research Tool Only

No trading signals. No financial advice.
No orders placed. Research monitoring only.

## Local Development

From the repository root:

```bash
pip install -r dashboard/requirements.txt
export SUPABASE_DB_URL=postgresql://...
streamlit run dashboard/app.py
```

Or from the dashboard directory:

```bash
cd dashboard
pip install -r requirements.txt
export SUPABASE_DB_URL=postgresql://...
streamlit run app.py
```

The app loads `SUPABASE_DB_URL` from the environment
or a repo-root `.env` file (never committed).

If the URL is missing the dashboard shows a
clear "DB not connected" message and stops.
That is expected behavior.

## Deploy to Streamlit Community Cloud (Free)

1. Push this repo to GitHub (already done)
2. Go to: https://share.streamlit.io
3. Sign in with GitHub (free, no credit card)
4. Click: **New app**
5. Select repo: `chess02kids-glitch/TRADING`
6. Set **Main file path**: `dashboard/app.py`
7. Click **Advanced settings**
   - Python version: 3.11 (or 3.12)
   - **Python packages file**: `dashboard/requirements.txt`
     (do not use the repo-root `requirements.txt` —
     it does not include Streamlit)
8. Still in Advanced settings → **Secrets**, add:

   ```toml
   SUPABASE_DB_URL = "postgresql://..."
   ```

   Use the Supabase **session pooler** or direct
   connection string. Never commit this value.
9. Click **Deploy**

Dashboard is live at a URL like:
`https://yourname-trading-dashboard-app-xxxx.streamlit.app`

Auto-deploys on every push to the connected branch.

## Architecture

```
dashboard/
  app.py          ← Main Streamlit app
  data_loader.py  ← Supabase read-only access
  charts.py       ← Plotly chart builders
  utils.py        ← Helper functions
  config.py       ← Configuration constants
  requirements.txt
  .streamlit/
    config.toml   ← Dark theme
  tests/
    test_data_loader.py
    test_charts.py
    test_utils.py
```

This package is standalone. It does **not**
import `kronos_trading` and does **not** write
to the database. Queries are `SELECT` only.

## Environment Variables

`SUPABASE_DB_URL`: Supabase PostgreSQL
connection string. Read-only role is enough.

The raw URL is never rendered in the UI.

## What It Shows

- Calibration progress (day N of 30)
- HAR MAE vs Persistence MAE per asset
- Improvement percentage over naive baseline
- Regime distribution (low/medium/high)
- Breakout events
- HAR model coefficients over time
- Recent prediction history

## Auto-Refresh

Dashboard automatically refreshes every
5 minutes via Streamlit's `st.cache_data` TTL.
Click **Refresh Data** in the sidebar for
an immediate reload.

## Tests

From the repository root:

```bash
pytest dashboard/tests/ -v
```
