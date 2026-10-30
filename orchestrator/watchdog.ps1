# watchdog.ps1   (§4.4) run every 15 min by the scheduled task OptD-RunPod-Watchdog; no Claude session involved.
# Deletes optd-* pods past the deadline in their name, older than 24 h with a malformed name, or 15 min past a final status.
. "$PSScriptRoot/common.ps1"
$logFile = Join-Path $PSScriptRoot 'watchdog.log'
function Log($m) { "$((Get-Date).ToString('s')) $m" | Add-Content $logFile }

try {
    $now = Get-Epoch
    $pods = @(Get-Pods | Where-Object name -like 'optd-*')
    $deleted = 0
    foreach ($p in $pods) {
        $why = $null
        if ($p.name -match '^optd-(.+)-d(\d+)$') {
            $runId = $Matches[1]
            if ($now -ge [long]$Matches[2]) { $why = 'deadline passed' }
            else {
                $s = ConvertFrom-JsonOrNull (Get-S3Text "runs/$runId/status.json")
                if ($s.state -in 'DONE', 'FAILED' -and $now - [long]$s.ts -ge 900) { $why = "status $($s.state) for 15+ min" }
            }
        } elseif ($created = ConvertTo-Epoch $p.createdAt) { if ($now -ge $created + 86400) { $why = 'malformed name, 24 h old' } }
        else { Log "WARN $($p.id) $($p.name): malformed name and no parseable createdAt" }
        if ($why) {
            runpodctl pod delete $p.id | Out-Null
            Log "DELETE $($p.id) $($p.name): $why (exit $LASTEXITCODE)"
            $deleted++
        }
    }
    # Unlisted is never gone: launched pods (local pod.json, last 24 h) that the API does not show, that were not killed
    # by kill.ps1 and did not log their own deletion, with no heartbeat for 10+ min are UNKNOWN.
    $killed = Get-Content (Join-Path $PSScriptRoot 'killed.log') -ErrorAction SilentlyContinue
    $unknown = 0
    foreach ($f in Get-ChildItem (Join-Path $Root 'runs/*/pod.json') -ErrorAction SilentlyContinue) {
        $pj = ConvertFrom-JsonOrNull (Get-Content $f -Raw)
        $created = ConvertTo-Epoch $pj.created
        if (-not $pj.id -or -not $created -or $now - $created -gt 86400 -or $now - $created -lt 600) { continue }
        if ($pods.id -contains $pj.id -or ($killed -match [regex]::Escape($pj.id))) { continue }
        $runId = $f.Directory.Name
        $s = ConvertFrom-JsonOrNull (Get-S3Text "runs/$runId/status.json")
        if ($s.state -in 'DONE', 'FAILED' -or ($s.ts -and $now - [long]$s.ts -lt 600)) { continue }
        if ((Get-S3Text "runs/$runId/log.txt") -match "pod delete accepted|pod `"$([regex]::Escape($pj.id))`" removed") { continue }
        Log "UNKNOWN $($pj.id) ${runId}: not listed, no heartbeat for 10+ min — check RunPod console"
        $unknown++
    }
    Log "run: $($pods.Count) optd pods, $deleted deleted, $unknown unknown"
} catch { Log "ERROR $($_.Exception.Message)"; exit 1 }
