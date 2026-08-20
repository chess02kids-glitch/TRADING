# Windows Task Scheduler — ready-made tasks for the HAR Alert Bot

Two ways to install, pick **one**:

## Option 1 — PowerShell installer (easiest, recommended)

Runs from any PowerShell window. It fills in your real repo paths and
registers the task — no XML editing.

```powershell
# Always-on bot (starts at your Windows logon, runs forever):
powershell -ExecutionPolicy Bypass -File config\taskscheduler\install_tasks.ps1

# OR hourly one-shot bot (runs one cycle at HH:30 every hour):
powershell -ExecutionPolicy Bypass -File config\taskscheduler\install_tasks.ps1 -Mode Hourly
```

Preview without installing:

```powershell
powershell -ExecutionPolicy Bypass -File config\taskscheduler\install_tasks.ps1 -DryRun
```

## Option 2 — manual import of the XML files

1. Open `har_alert_bot_always_on.xml` (or `har_alert_bot_hourly_once.xml`)
   in Notepad / VS Code.
2. Replace **both** occurrences of `C:\path\to\TRADING` with your real repo
   path (e.g. `C:\Users\you\TRADING`). The file must stay **UTF-16** — use
   "Save As" → Encoding: Unicode (or don't edit at all and run the installer).
3. Windows → type "Task Scheduler" → open it.
4. Right side: **Import Task…** → pick the XML file → OK.
5. Optional (hourly file only): double-click the task → Triggers → edit the
   start date if you want a different first run.

## Which one should you use?

| | `har_alert_bot_always_on.xml` | `har_alert_bot_hourly_once.xml` |
|---|---|---|
| Task name | HAR Alert Bot | HAR Alert Bot Hourly |
| Starts | at your Windows logon | first run tomorrow 00:00:30, then every hour at HH:30 |
| Runs | forever (pythonw, no window) | one cycle, then exits |
| Crash recovery | restarts after 5 min (3 tries) | next hour's trigger |
| Best for | the 30-day experiment | cron-style / low-footprint |

> ⚠️ **Never install both.** The bot would run twice per hour and send
> duplicate Telegram forecasts. Pick one.

## How it works / requirements

- Uses `.venv\Scripts\pythonw.exe` (no console window). Create the venv first:
  `py -3.10 -m venv .venv` then `pip install numpy pandas ccxt requests python-dotenv`.
- The task's **Start in** folder is the repo root — this matters: the bot
  loads `.env` (your Telegram credentials) from the working directory.
- Both tasks run only while you are logged in to Windows (interactive token,
  no password stored). To run without logging in you would need to enable
  "Run whether user is logged on or not" and enter your Windows password.
- Logs: `logs\har_bot.log` in the repo. Check any time with
  `python scripts/run_alert_bot.py --status`.

## Uninstall

```powershell
schtasks /Delete /TN "HAR Alert Bot" /F
schtasks /Delete /TN "HAR Alert Bot Hourly" /F
```
