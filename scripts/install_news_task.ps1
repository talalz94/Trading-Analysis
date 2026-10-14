<#
Installs (or removes) the Windows scheduled task that keeps the economic calendar current.

    powershell -ExecutionPolicy Bypass -File scripts\install_news_task.ps1              # install
    powershell -ExecutionPolicy Bypass -File scripts\install_news_task.ps1 -Uninstall   # remove

The task runs `pythonw -m quant.news update` (pythonw = no console window) when you log on and
every 6 hours, and appends what happened to data\news\update.log. A run missed while the PC was
off or asleep happens as soon as it is back (StartWhenAvailable).

Why it matters: Forex Factory's feed only ever shows THIS week, so a week nobody fetched loses its
forecasts for good. The agencies' actuals and the official dates can always be fetched later.
Each run takes seconds; a source that is down is retried after 30 minutes, not hammered.

It runs as you, only while you are logged on, with no stored password. Keys come from .env
(FRED_API_KEY for the US actuals and dates; see docs/NEWS.md).
#>
param(
    [switch]$Uninstall,
    [string]$Python = ""        # python.exe to use; default = the first on PATH
)
$ErrorActionPreference = "Stop"
$name = "quant-news"

if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
        Write-Host "Removed the scheduled task '$name'."
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
    & $Python -c "import quant.news.update, requests, pyarrow"
    if ($LASTEXITCODE -ne 0) { throw "$Python cannot import quant.news; pass -Python <path>" }
} finally { Pop-Location }

$log = Join-Path $repo "data\news\update.log"
$user = "$env:USERDOMAIN\$env:USERNAME"
$action = New-ScheduledTaskAction -Execute $pythonw `
    -Argument "-m quant.news update --log `"$log`"" -WorkingDirectory $repo
$atLogon = New-ScheduledTaskTrigger -AtLogOn -User $user
$atLogon.Delay = "PT2M"                                    # let the network come up first
$every6h = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Hours 6)            # no duration = indefinitely
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $name -Action $action -Trigger $atLogon, $every6h `
    -Settings $settings -Principal $principal -Force `
    -Description "quant economic calendar: python -m quant.news update, every 6 h (docs/NEWS.md)" |
    Out-Null
Start-ScheduledTask -TaskName $name

Write-Host "Installed '$name' with $pythonw, and started a first run."
Write-Host "Its log:    $log"
Write-Host "Check it:   python -m quant.news status"
Write-Host "Remove it:  powershell -ExecutionPolicy Bypass -File scripts\install_news_task.ps1 -Uninstall"
