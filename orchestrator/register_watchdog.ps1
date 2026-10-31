# register_watchdog.ps1   (§4.4) the human runs this once in pwsh: registers OptD-RunPod-Watchdog (every 15 min, user session).
$ErrorActionPreference = 'Stop'
$script = Join-Path $PSScriptRoot 'watchdog.ps1'
$pwsh = (Get-Command pwsh).Source
# conhost --headless: pwsh -WindowStyle Hidden still flashes a console window for a moment.
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\conhost.exe" -WorkingDirectory (Split-Path $PSScriptRoot -Parent) `
    -Argument "--headless `"$pwsh`" -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 15)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
Register-ScheduledTask -TaskName 'OptD-RunPod-Watchdog' -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null

$t = Get-ScheduledTask -TaskName 'OptD-RunPod-Watchdog'
"registered: $($t.TaskName)  state=$($t.State)"
"  runs:  $($t.Actions[0].Execute) $($t.Actions[0].Arguments)"
"  every: $($t.Triggers[0].Repetition.Interval)  first: $($t.Triggers[0].StartBoundary)"
"  log:   $(Join-Path $PSScriptRoot 'watchdog.log')"
