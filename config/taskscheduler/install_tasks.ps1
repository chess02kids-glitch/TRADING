#Requires -Version 5.1
<#
.SYNOPSIS
    Installs the HAR Alert Bot as a Windows scheduled task.

.DESCRIPTION
    Builds the task XML with your real repo paths and registers it with
    schtasks - no manual XML editing needed.

    Two modes (install exactly ONE - never both, or the bot would run twice
    and send duplicate Telegram messages):

      -Mode AlwaysOn (default): task "HAR Alert Bot"
          Starts when you log in to Windows and keeps running forever
          (pythonw.exe, no console window). Best for the 30-day run.

      -Mode Hourly: task "HAR Alert Bot Hourly"
          Runs "scripts/run_alert_bot.py --once" every hour at HH:30 and
          exits. Lightweight, cron-style.

.PARAMETER Mode
    AlwaysOn (default) or Hourly.

.PARAMETER DryRun
    Print the XML and the schtasks command without registering anything.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File config\taskscheduler\install_tasks.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File config\taskscheduler\install_tasks.ps1 -Mode Hourly

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File config\taskscheduler\install_tasks.ps1 -DryRun
#>
param(
    [ValidateSet("AlwaysOn", "Hourly")]
    [string]$Mode = "AlwaysOn",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# --- Repo root = parent of the folder this script lives in (config\taskscheduler)
$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$PythonW  = Join-Path $RepoRoot ".venv\Scripts\pythonw.exe"
$TaskName = if ($Mode -eq "AlwaysOn") { "HAR Alert Bot" } else { "HAR Alert Bot Hourly" }

if (-not (Test-Path $PythonW)) {
    Write-Host "ERROR: virtual environment not found at:" -ForegroundColor Red
    Write-Host "  $PythonW" -ForegroundColor Red
    Write-Host "Create it first from the repo root:" -ForegroundColor Yellow
    Write-Host "  py -3.10 -m venv .venv" -ForegroundColor Yellow
    Write-Host "  .venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "  pip install numpy pandas ccxt requests python-dotenv" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path (Join-Path $RepoRoot ".env"))) {
    Write-Host "WARNING: .env not found in $RepoRoot" -ForegroundColor Yellow
    Write-Host "The bot needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to send messages." -ForegroundColor Yellow
    Write-Host "(It will still run and log locally, but Telegram sends will fail.)" -ForegroundColor Yellow
}

# --- Build the task XML with real paths -------------------------------------
$Date = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
if ($Mode -eq "AlwaysOn") {
    $StartBoundary = ""  # LogonTrigger: no start boundary needed
    $Arguments = "scripts/run_alert_bot.py"
    $Description = "HAR volatility alert bot - continuous monitoring mode. Sends hourly Telegram volatility forecasts. Paper research only, no trades."
    $TriggerXml = @"
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
"@
    $RestartXml = @"
    <RestartOnFailure>
      <Interval>PT5M</Interval>
      <Count>3</Count>
    </RestartOnFailure>
"@
} else {
    # Hourly: first run at tomorrow 00:00:30, then every hour.
    $StartBoundary = (Get-Date).Date.AddDays(1).AddMinutes(0.5).ToString("yyyy-MM-ddTHH:mm:ss")
    $Arguments = "scripts/run_alert_bot.py --once"
    $Description = "HAR volatility alert bot - hourly single-cycle mode. Runs one forecast cycle at HH:30 each hour. Paper research only, no trades."
    $TriggerXml = @"
    <TimeTrigger>
      <StartBoundary>$StartBoundary</StartBoundary>
      <Enabled>true</Enabled>
      <Repetition>
        <Interval>PT1H</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
    </TimeTrigger>
"@
    $RestartXml = ""
}

$xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Date>$Date</Date>
    <Author>HAR Alert Bot</Author>
    <Description>$Description</Description>
  </RegistrationInfo>
  <Triggers>
$TriggerXml  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
$RestartXml    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>$PythonW</Command>
      <Arguments>$Arguments</Arguments>
      <WorkingDirectory>$RepoRoot</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@

# --- Write as UTF-16 (what schtasks expects) and register --------------------
$tmp = Join-Path $env:TEMP "har_alert_bot_task.xml"
[System.IO.File]::WriteAllText($tmp, $xml, [System.Text.Encoding]::Unicode)

if ($DryRun) {
    Write-Host "=== DRY RUN: task '$TaskName' would be registered as ==="
    Write-Host "  schtasks /Create /TN `"$TaskName`" /XML `"$tmp`" /F"
    Write-Host ""
    Get-Content $tmp
    Write-Host ""
    Write-Host "(nothing was created)"
    exit 0
}

# Remove any existing task with the same name (ignore "not found" errors).
schtasks /Delete /TN $TaskName /F 2>$null | Out-Null

schtasks /Create /TN $TaskName /XML $tmp /F
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: schtasks failed to create task '$TaskName' (exit $LASTEXITCODE)." -ForegroundColor Red
    Write-Host "Try importing $tmp manually via Task Scheduler -> Import Task..." -ForegroundColor Yellow
    exit 1
}

Remove-Item $tmp -ErrorAction SilentlyContinue

Write-Host "SUCCESS: task '$TaskName' installed." -ForegroundColor Green
Write-Host "  Command : $PythonW $Arguments"
Write-Host "  Workdir : $RepoRoot"
if ($Mode -eq "AlwaysOn") {
    Write-Host "  Starts  : at your next Windows logon"
} else {
    Write-Host "  Starts  : $StartBoundary, then every hour at HH:30"
}
Write-Host ""
Write-Host "Verify: Task Scheduler -> Task Scheduler Library -> '$TaskName' (right-click -> Run)."
Write-Host "Logs   : $RepoRoot\logs\har_bot.log"
Write-Host "To remove:  schtasks /Delete /TN `"$TaskName`" /F"
