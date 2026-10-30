# watch.ps1 -RunId <id> [-TimeoutMinutes n]   (§4.2) exit 0 DONE, 2 FAILED, 3 timeout. Prints only on state/step change.
param([Parameter(Mandatory)][string]$RunId, [int]$TimeoutMinutes = 60, [int]$PollSeconds = 300, [int]$StallMinutes = 30)
. "$PSScriptRoot/common.ps1"
$podId = (Get-Content (Join-Path $Root "runs/$RunId/pod.json") -Raw | ConvertFrom-Json).id
$t0 = Get-Date; $lastChange = $t0; $lastRaw = $null; $lastKey = ''

while ($true) {
    $raw = Get-S3Text "runs/$RunId/status.json"
    if ($raw -and $raw -ne $lastRaw) { $lastRaw = $raw; $lastChange = Get-Date }
    $s = ConvertFrom-JsonOrNull $raw
    $key = "$($s.state)|$($s.step)"
    if ($s -and $key -ne $lastKey) {
        $lastKey = $key
        "[$(Get-Date -Format HH:mm:ss)] state=$($s.state) step=$($s.step) $($s.msg)"
        $log = Get-S3Text "runs/$RunId/log.txt"
        if ($log) { ($log -split "`n" | Select-Object -Last 10) | ForEach-Object { "    $_" } }
    }
    if ($s.state -eq 'DONE') { exit 0 }
    if ($s.state -eq 'FAILED') { exit 2 }

    $pod = Get-Pod $podId
    if (-not $pod) {
        $s = ConvertFrom-JsonOrNull (Get-S3Text "runs/$RunId/status.json")   # DONE may have landed just before self-terminate
        if ($s.state -eq 'DONE') { "[$(Get-Date -Format HH:mm:ss)] state=DONE step=$($s.step)"; exit 0 }
        # The API hides pods while they pull their image; only a stale status.json (heartbeat is 5 min) means gone.
        if (((Get-Date) - $lastChange).TotalMinutes -ge 12) {
            "[$(Get-Date -Format HH:mm:ss)] FAILED: pod gone without DONE (last: $($s.state)|$($s.step) $($s.msg))"; exit 2
        }
    }
    if ($pod.desiredStatus -eq 'RUNNING' -and ((Get-Date) - $lastChange).TotalMinutes -ge $StallMinutes) {
        & "$PSScriptRoot/kill.ps1" $podId
        "[$(Get-Date -Format HH:mm:ss)] FAILED: stalled (status.json unchanged $StallMinutes min)"; exit 2
    }
    if (((Get-Date) - $t0).TotalMinutes -ge $TimeoutMinutes) { "[$(Get-Date -Format HH:mm:ss)] timeout after $TimeoutMinutes min (last: $key)"; exit 3 }
    Start-Sleep -Seconds $PollSeconds
}
