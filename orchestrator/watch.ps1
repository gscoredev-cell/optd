# watch.ps1 -RunId <id> [-TimeoutMinutes n]   (§4.2) exit 0 DONE, 2 FAILED, 3 timeout, 4 timeout while UNKNOWN.
# Prints only on state/step change. "Not listed" is never "gone": an invisible pod without a fresh heartbeat is UNKNOWN.
param([Parameter(Mandatory)][string]$RunId, [int]$TimeoutMinutes = 60, [int]$PollSeconds = 300, [int]$StallMinutes = 30)
. "$PSScriptRoot/common.ps1"
$runDir = Join-Path $Root "runs/$RunId"
$podId = (Get-Content (Join-Path $runDir 'pod.json') -Raw | ConvertFrom-Json).id
$logFile = Join-Path $runDir 'watch.log'
function Say($m) { $line = "[$(Get-Date -Format HH:mm:ss)] $m"; $line | Add-Content $logFile; $line }
$t0 = Get-Date; $lastChange = $t0; $lastRaw = $null; $lastKey = ''; $unknown = $false

while ($true) {
    $raw = Get-S3Text "runs/$RunId/status.json"
    if ($raw -and $raw -ne $lastRaw) { $lastRaw = $raw; $lastChange = Get-Date }
    $s = ConvertFrom-JsonOrNull $raw
    $key = "$($s.state)|$($s.step)"
    if ($s -and $key -ne $lastKey) {
        $lastKey = $key
        Say "state=$($s.state) step=$($s.step) $($s.msg)"
        $log = Get-S3Text "runs/$RunId/log.txt"
        if ($log) { ($log.TrimEnd() -split "`n" | Select-Object -Last 10) | ForEach-Object { "    $_" } }
    }
    if ($s.state -eq 'DONE') { exit 0 }
    if ($s.state -eq 'FAILED') { exit 2 }

    $pod = Get-Pod $podId
    $quiet = ((Get-Date) - $lastChange).TotalMinutes
    if ($pod) {
        if ($unknown) { Say "pod $podId visible again ($($pod.desiredStatus)/$($pod.runtimeStatus))"; $unknown = $false }
        if ($pod.desiredStatus -eq 'RUNNING' -and $quiet -ge $StallMinutes) {
            & "$PSScriptRoot/kill.ps1" $podId
            Say "FAILED: stalled (status.json unchanged $StallMinutes min)"; exit 2
        }
    } elseif ($quiet -ge 10 -and -not $unknown) {
        $unknown = $true
        Say "UNKNOWN: pod $podId not visible to the API and no heartbeat for $([int]$quiet) min — check RunPod console"
    }
    if (((Get-Date) - $t0).TotalMinutes -ge $TimeoutMinutes) {
        Say "timeout after $TimeoutMinutes min (last: $key$(if ($unknown) { ', UNKNOWN' }))"; exit $(if ($unknown) { 4 } else { 3 })
    }
    Start-Sleep -Seconds $PollSeconds
}
