<#
Installs (or removes) the Windows scheduled task that keeps the alerts monitor running.

    powershell -ExecutionPolicy Bypass -File scripts\install_alerts_task.ps1              # install
    powershell -ExecutionPolicy Bypass -File scripts\install_alerts_task.ps1 -Uninstall   # remove

The task starts `pythonw -m quant.alerts run` (pythonw = no console window) when you log on, and
again every 5 minutes. While the monitor runs, Task Scheduler ignores those triggers (and a copy
started by hand finds the lock held and exits), so the 5-minute trigger is a WATCHDOG: if the
monitor dies, it is back within 5 minutes and its first tick catches up on what closed meanwhile.

It runs as you, only while you are logged on, with no stored password. Set up email first
(docs/ALERTS.md) and check it with:  python -m quant.alerts test-email
#>
param(
    [switch]$Uninstall,
    [string]$Python = ""        # python.exe to use; default = the first on PATH
)
$ErrorActionPreference = "Stop"
$name = "quant-alerts"

if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        Stop-ScheduledTask -TaskName $name
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
        Write-Host "Stopped and removed the scheduled task '$name'."
    } else {
        Write-Host "No scheduled task named '$name'."
    }
    return
}

$repo = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = (Get-Command python.exe).Source }
$pythonw = Join-Path (Split-Path $Python) "pythonw.exe"
if (-not (Test-Path $pythonw)) { throw "pythonw.exe not found next to $Python" }

# the interpreter must be the one the repo runs on -- several Pythons are installed here
Push-Location $repo
try {
    & $Python -c "import quant.alerts, dukascopy_python, requests"
    if ($LASTEXITCODE -ne 0) { throw "$Python cannot import quant.alerts; pass -Python <path>" }
    & $Python -m quant.alerts status | Out-Null
} finally { Pop-Location }

$user = "$env:USERDOMAIN\$env:USERNAME"
$action = New-ScheduledTaskAction -Execute $pythonw -Argument "-m quant.alerts run" `
    -WorkingDirectory $repo
$atLogon = New-ScheduledTaskTrigger -AtLogOn -User $user
$every5 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 5)            # no duration = indefinitely
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $name -Action $action -Trigger $atLogon, $every5 `
    -Settings $settings -Principal $principal -Force `
    -Description "quant alerts monitor (alerts/alerts_config.py). Watchdog: restarts it if it stops." |
    Out-Null
Start-ScheduledTask -TaskName $name

Write-Host "Installed '$name' with $pythonw, and started it."
Write-Host "Check it:   python -m quant.alerts status      log: alerts\state\alerts.log"
Write-Host "Remove it:  powershell -ExecutionPolicy Bypass -File scripts\install_alerts_task.ps1 -Uninstall"
