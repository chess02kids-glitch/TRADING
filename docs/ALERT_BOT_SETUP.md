# HAR Alert Bot — Beginner Setup Guide (Windows)

This guide walks you through setting up the Telegram volatility alert bot on
your Windows laptop, step by step. It assumes you are *not* an expert — every
command is shown exactly as you type it, and common mistakes are covered at
the end.

---

## 0. What you are setting up (30 seconds)

The bot runs your **validated HAR volatility model** once per hour on BTC/USDT
and ETH/USDT 1-hour candles and sends you a Telegram message with:

- the HAR-predicted range of the next candle (e.g. "Predicted range: $515.07"),
- the volatility regime (LOW / MEDIUM / HIGH),
- an alert when a candle's actual range was more than **2×** the prediction
  (a "breakout" — unusual market conditions).

Everything is logged to a local SQLite database so that after **30 days** you
can check whether HAR is still calibrated on live data.

> ⚠️ **This is a monitoring tool only.** It predicts *how much* the price may
> move, never *which direction*. It places **no trades**. Zero financial risk.

---

## 1. What you need (checklist)

| Item | You have it? |
|---|---|
| Windows laptop (any recent version) | ✅ you are on it |
| Python 3.10.13 | ✅ you already installed it |
| Git for Windows | ✅ you already use branches |
| The TRADING repo on your machine | ✅ |
| Internet access (for Binance public API + Telegram) | ✅ |
| A Telegram account (free) | ✅ almost certainly |

You do **not** need a Binance account or API key. The bot uses Binance's
public, keyless market data endpoint.

---

## 2. Get the latest code (with the alert bot)

The alert bot lives on the branch `arena/01a01ed6-trading` (all 6 steps were
pushed there). Open **PowerShell** (press `Win`, type `PowerShell`, Enter) and
go to your repo folder:

```powershell
cd C:\path\to\TRADING        # ← change to where your repo actually is
git fetch origin
git checkout arena/01a01ed6-trading
```

You should now see the new files:

```powershell
dir kronos_trading\alerts
dir scripts\run_alert_bot.py
```

> If you prefer to keep working on your usual branch instead, run
> `git merge arena/01a01ed6-trading` from that branch. The setup below is
> identical either way.

---

## 3. Create a virtual environment

A virtual environment keeps the bot's dependencies separate from your other
Python projects (including your heavy torch install, which the bot does NOT
need).

In PowerShell, from the repo root:

```powershell
py -3.10 -m venv .venv
```

(`py -3.10` is the Windows Python launcher. If it says `py is not
recognized`, use `python -m venv .venv` instead.)

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`.

> **First-time PowerShell error?** If activation is blocked with a message
> about "execution policy", run this once and try again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
> (If you prefer cmd.exe over PowerShell, activate with
> `.venv\Scripts\activate.bat` instead — no policy issue.)

---

## 4. Install the dependencies

Still inside the activated environment (prompt shows `(.venv)`):

```powershell
python -m pip install --upgrade pip
pip install numpy pandas ccxt requests python-dotenv pytest
```

That's the **entire** install — the alert bot is deliberately lightweight
(no torch, no GPU). To confirm:

```powershell
python -c "import numpy, pandas, ccxt, requests, dotenv; print('all good')"
```

---

## 5. Create your Telegram bot (BotFather) and the `.env` file

### 5a. Get a bot token

1. Open Telegram on your phone or desktop.
2. Search for **@BotFather** (the official bot creator) and press Start.
3. Send: `/newbot`
4. Give it a **name**, e.g. `HAR Volatility Bot`.
5. Give it a **username** that ends in `bot`, e.g. `har_volatility_bot`.
   (If the username is taken, add numbers: `har_volatility_bot_2026`.)
6. BotFather replies with a **token** that looks like:
   `7123456789:AAHxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   Copy it somewhere safe. **Never share it and never commit it to git.**

### 5b. Find your chat ID

1. In Telegram, open your new bot and send it any message, e.g. `hi`.
2. Open a browser and visit:
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
   (replace `<YOUR_TOKEN>` with the real token)
3. You'll see JSON. Look for the number next to `"chat":{"id":`:
   ```
   "chat":{"id":123456789,"first_name":"You",...}
   ```
   That number (`123456789`) is your **chat ID**.
4. If the page shows `"ok":true` but an empty `result:[]`, you didn't send a
   message yet — send another message to the bot and refresh.

### 5c. Write the `.env` file

In the repo root, create a file named exactly **`.env`** (no other name) with
these two lines:

```
TELEGRAM_BOT_TOKEN=7123456789:AAHxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TELEGRAM_CHAT_ID=123456789
```

Notes:

- `.env` is already in the repo's `.gitignore` — git will never commit it.
  But be careful anyway: don't paste the token into any other file.
- Use **Notepad** (or VS Code). In Notepad, make sure "Save as type" is
  **All files** and the file name is `.env`, otherwise you'll get `.env.txt`.
- The bot reads this file from the folder you run it in, so **always run the
  bot from the repo root**.

---

## 6. Verify the code with the test suite (recommended)

From the repo root (venv activated):

```powershell
python -m pytest tests/test_har_forecaster.py tests/test_prediction_logger.py tests/test_breakout_detector.py tests/test_telegram_sender.py tests/test_scheduler.py tests/test_run_alert_bot.py -v
```

You should see **151 passed** in a few seconds. (A few tests elsewhere in the
repo may fail in some setups because they need large model files — they are
unrelated to the alert bot.)

---

## 7. Dry run — test the pipeline WITHOUT sending anything

```powershell
python scripts/run_alert_bot.py --dry-run
```

This fetches real BTC/ETH candles from Binance, computes the HAR forecast,
**prints** the would-be Telegram message to the console, and logs the
prediction to the database. Nothing is sent to Telegram.

**Expected output** (approximately):

```
[DRY RUN] Would send:
🔮 HAR Volatility Forecast
━━━━━━━━━━━━━━━━━━━━
BTC/USDT 1h
  Predicted range: $515.07
  Regime: medium
...
[DRY RUN] Cycle complete.
Assets: ['BTC/USDT', 'ETH/USDT']
Errors: []
Duration: 0.04s
```

Check the database has the prediction (still "pending", i.e. not yet closed):

```powershell
python scripts/run_alert_bot.py --status
```

You should see `Total predictions logged: 1` and `Pending (awaiting close): 1`.

> **No internet to Binance?** You may see
> `fetch failed ... 451` or a timeout — that happens when the network blocks
> Binance (some countries/ISPs do). The bot handles it gracefully and exits
> cleanly. If this appears on your home connection, that's the problem to
> solve before the live run.

---

## 8. Run the bot for real (first live test)

Still from the repo root:

```powershell
python scripts/run_alert_bot.py
```

What you should see:

1. A `🟢 HAR Alert Bot started` message in Telegram.
2. Log lines in the console showing the cycle ran.
3. Within a minute or two, the bot sleeps until the next hour mark + 30 s.
4. At the next hour + 30 s: a `🔮 HAR Volatility Forecast` message arrives in
   Telegram, and the console logs `Sleeping 3600 seconds`.

**Keep this window open** — closing it stops the bot. If you close it anyway,
that's fine: the bot is designed so missed hours are harmless. The next time
it runs, it fills in the actual ranges of the candles that closed while it
was off (that's the "pending predictions" step).

---

## 9. Run it 24/7 with Windows Task Scheduler

For the 30-day experiment the bot should run continuously. The easiest path
uses the ready-made task files (already in your repo):

### Option 0 — ready-made tasks (easiest)

Your repo contains `config/taskscheduler/` with two task definitions plus a
PowerShell installer that fills in your real paths automatically.

**Recommended — run the installer** (from the repo root, any PowerShell):

```powershell
# Always-on: starts at your Windows logon, runs forever (no window):
powershell -ExecutionPolicy Bypass -File config\taskscheduler\install_tasks.ps1

# OR hourly one-shot (runs one cycle at HH:30 every hour, then exits):
powershell -ExecutionPolicy Bypass -File config\taskscheduler\install_tasks.ps1 -Mode Hourly
```

It checks that `.venv` exists, warns if `.env` is missing, and registers the
task. Verify: Task Scheduler → Task Scheduler Library → **HAR Alert Bot**
(right-click → **Run**), then check `logs\har_bot.log` and your Telegram for
the 🟢 startup message.

**Or import the XML by hand:**

1. Open `config\taskscheduler\har_alert_bot_always_on.xml` (or
   `har_alert_bot_hourly_once.xml`) in Notepad.
2. Replace **both** occurrences of `C:\path\to\TRADING` with your real repo
   path. Save As → Encoding: **Unicode** (UTF-16).
3. Task Scheduler → right side **Import Task…** → pick the file → OK.

> ⚠️ **Never install both** the always-on and the hourly task — the bot
> would run twice per hour and send duplicate forecasts. Pick one.
> Full details: `config\taskscheduler\README.md`.

The two manual recipes below (Option A = always-on, Option B = hourly) are
the same configurations built click-by-click, in case you prefer doing it by
hand or need to tweak something.

### Option A — always-on (simplest, recommended)

1. `Win` → type "Task Scheduler" → open it.
2. Right side: **Create Task…**
3. **General** tab:
   - Name: `HAR Alert Bot`
   - Check **Run whether user is logged on or not** (or leave unchecked — see note below).
4. **Triggers** tab → **New…**:
   - Begin the task: **At startup**
   - Check **Enabled** → OK.
5. **Actions** tab → **New…**:
   - Action: **Start a program**
   - Program/script: `C:\path\to\TRADING\.venv\Scripts\pythonw.exe`
     (`pythonw.exe` runs without a console window — logs go to the log file)
   - Add arguments: `scripts/run_alert_bot.py`
   - Start in: `C:\path\to\TRADING`
   - OK.
6. **Conditions** tab: uncheck **Start the task only if the computer is on
   AC power** (so it runs on battery too).
7. OK, enter your Windows password if asked.

> If you choose "Run whether user is logged on or not", the bot starts even
> before you log in — but it will only send Telegram messages while your
> laptop is awake. For a true 30-day run, leave the laptop plugged in and
> **disable sleep**: Settings → System → Power → "Never" sleep. Laptop sleep
> pauses the bot (gaps are handled gracefully, but you want maximum coverage).

### Option B — run once per hour via `--once` (cron style)

The `--once` mode runs exactly one cycle and exits — ideal for a scheduled
task that fires every hour:

1. Task Scheduler → **Create Task…** → Name `HAR Alert Bot Hourly`.
2. **Triggers** → **New…**:
   - Begin the task: `01/01/2026` at `00:00:30`
   - Check **Repeat task every: `1 hour`**
   - **for a duration of: Indefinitely**
   - Enabled → OK.
3. **Actions** → **New…**:
   - Program/script: `C:\path\to\TRADING\.venv\Scripts\python.exe`
   - Add arguments: `scripts/run_alert_bot.py --once`
   - Start in: `C:\path\to\TRADING`
4. Conditions: uncheck AC-power requirement.
5. OK.

The `:30` start makes the bot fetch ~30 seconds after the candle closes,
exactly as designed.

---

## 10. Day-to-day checks

All from the repo root (venv activated):

| What you want | Command |
|---|---|
| See the bot's health (predictions, pending, calibration) | `python scripts/run_alert_bot.py --status` |
| Send the calibration report to Telegram now | `python scripts/run_alert_bot.py --calibrate` |
| See recent logs | `type logs\har_bot.log` |
| Follow the live log | `Get-Content logs\har_bot.log -Wait -Tail 20` |
| Check the database file exists | `dir data\db\har_predictions.db` |

Every hourly cycle appends to `logs/har_bot.log`, including the cycle
summary: `success=True assets=['BTC/USDT','ETH/USDT'] errors=0`.

---

## 11. What happens after 30 days (the decision)

When you have ~720 completed predictions, run:

```powershell
python scripts/run_alert_bot.py --status
```

Look at the **Calibration** block per asset:

- **HAR beats naive: YES** means HAR's mean absolute error is smaller than
  the "predict the same as last time" baseline on live data → the model is
  still calibrated → **proceed to designing the paper-trading experiment**.
- **HAR beats naive: NO** (or **Degrading: YES** on the report) means live
  performance no longer beats the naive baseline → the project is
  scientifically complete; archive and stop. No money was ever at risk.

Also worth a look after 30 days: how many **breakout alerts** fired
(`Breakout rate`), and whether they were followed by continued high
volatility or quiet mean reversion — that tells you whether breakouts carry
information.

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `'py' is not recognized as a command` | Use `python -m venv .venv` instead. |
| Activation blocked by execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, retry. |
| `ModuleNotFoundError: No module named 'numpy'` | The venv isn't active — your prompt must show `(.venv)`. Re-run step 3's activate command. |
| Bot sends nothing / `Telegram API error: 401` | Token is wrong — re-copy it from BotFather into `.env`. |
| `Telegram API error: 400 ... chat not found` | Chat ID is wrong or the bot hasn't received a message from you — redo step 5b. |
| `getUpdates` shows empty `result` | Send another message to your bot, refresh the URL. |
| `fetch failed ... 451` | Binance is blocked on your network (geo-block/ISP). Try a different network/VPN, or accept the bot can't run there. |
| Forecast messages are correct but no breakouts ever | Normal — breakouts (range > 2× prediction) are rare events; that's the point. |
| Bot stopped after closing the window | Expected. Use Task Scheduler (step 9) to keep it running. |
| Two forecast messages at once after a gap | Expected. The bot catches up when it starts; duplicate log entries are prevented by the database unique key. |
| Laptop sleeps and misses hours | Enable "Never" sleep or use Task Scheduler; missed hours are backfilled automatically. |

---

## 13. File map (what you just set up)

```
TRADING/
├─ .env                          ← your secret Telegram credentials (never commit)
├─ scripts/
│  └─ run_alert_bot.py           ← ENTRY POINT — the only file you run
├─ kronos_trading/alerts/
│  ├─ har_forecaster.py          ← HAR model (validated, past-only OLS)
│  ├─ prediction_logger.py       ← SQLite logging (har_predictions table)
│  ├─ breakout_detector.py       ← breakout checks + calibration statistics
│  ├─ telegram_sender.py         ← Telegram API (the only file that talks to it)
│  └─ scheduler.py               ← hourly main loop
├─ data/db/har_predictions.db    ← the experiment database (created on first run)
└─ logs/har_bot.log              ← the log file (created on first run)
```

**The three commands you'll ever need:**

```powershell
python scripts/run_alert_bot.py --dry-run    # test without sending
python scripts/run_alert_bot.py              # run forever (or via Task Scheduler)
python scripts/run_alert_bot.py --status     # check the experiment anytime
```
