# launch.ps1 -Phase <P> [-Seed n] [-MaxMinutes m] [-DeadlineEpoch e]   (§4.1; the overrides exist for the watchdog test)
# Success = runs/<RUN_ID>/status.json appears on the volume within 10 min of creation. SSH is debugging-only: no --wait,
# ip/port are recorded when pod get shows them, never required. Exit 0 ok, 2 FAILED: no heartbeat, 3 NO_CAPACITY.
param(
    [Parameter(Mandatory)][ValidateSet('WATCHDOGTEST', 'ENVCHECK', 'SMOKE', 'SMOKE3', 'SMOKE4', 'SMOKE5', 'SMOKE6', 'SMOKE7', 'SMOKE8', 'SMOKE9', 'FULL')][string]$Phase,
    [int]$Seed = 0, [int]$MaxMinutes = 0, [long]$DeadlineEpoch = 0
)
. "$PSScriptRoot/common.ps1"
Set-Location $Root

# 0. The local watchdog (§4.4) must be registered and enabled: it is the only guard independent of this session.
$wd = Get-ScheduledTask -TaskName 'OptD-RunPod-Watchdog' -ErrorAction SilentlyContinue
if (-not $wd) { throw 'scheduled task OptD-RunPod-Watchdog is missing: run orchestrator/register_watchdog.ps1 first' }
if ($wd.State -eq 'Disabled') { throw 'scheduled task OptD-RunPod-Watchdog is disabled: enable it first' }

# 1. Clean, pushed commit (run artifacts under runs/ do not affect the pod's code).
if (git status --porcelain -- . ':!runs') { throw 'uncommitted changes: commit and push first' }
$sha = (git rev-parse HEAD).Trim()
git fetch -q origin
git merge-base --is-ancestor $sha origin/main
if ($LASTEXITCODE -ne 0) { throw "HEAD $sha is not pushed to origin/main" }
$runId = "$Phase-$(Get-Date -Format yyyyMMdd-HHmm)-$($sha.Substring(0, 7))"
if ($PSBoundParameters.ContainsKey('Seed')) { $runId += "-s$Seed" }
$runDir = Join-Path $Root "runs/$runId"
New-Item -ItemType Directory -Force $runDir | Out-Null
$logFile = Join-Path $runDir 'launch.log'
function Log($m) { $line = "$(Get-Date -Format HH:mm:ss) $m"; $line | Add-Content $logFile; $line | Out-Host }

$budget = [double]$Cfg.phase_budget_usd[$Phase]
$ceiling = [int]$Cfg.phase_ceiling_min[$Phase]
$pubKey = (Get-Content (Join-Path $HOME '.ssh/id_ed25519.pub') -Raw).Trim()
$start = "bash -lc `"curl -fsSL $($Cfg.git_raw)/$sha/pod/bootstrap.sh | bash`""

function Save-PodJson($s) {
    $s | ConvertTo-Json | Set-Content (Join-Path $runDir 'pod.json')
    if (-not (Send-S3File (Join-Path $runDir 'pod.json') "runs/$runId/pod.json")) { Log 'WARN pod.json upload to the volume failed' }
}

function Start-Pod($g) {
    # 3. Lifetime from this GPU's price.
    $max = if ($MaxMinutes) { $MaxMinutes } else { [math]::Min($ceiling, [math]::Floor($budget / $g.pricePerHr * 60)) }
    $launch = Get-Epoch
    $deadline = if ($DeadlineEpoch) { $DeadlineEpoch } else { $launch + ($max + [int]$Cfg.deadline_grace_min) * 60 }
    $cost = $g.pricePerHr * $max / 60
    Log ([string]::Format($Inv, 'attempt {0} ({1:0.00} $/h): budget {1:0.00} $/h x {2} min = ${3:0.00} <= ${4:0.00}', $g.gpuId, $g.pricePerHr, $max, $cost, $budget))
    if ($cost -gt $budget) { throw 'budget exceeded' }
    $name = "optd-$runId-d$deadline"
    # 5. Env (integers as strings; HF_TOKEN is a RunPod Secret reference, never a value).
    $envJson = [ordered]@{
        RUN_ID = $runId; PHASE = $Phase; SEED = "$Seed"; GIT_REPO = $Cfg.git_repo; GIT_SHA = $sha
        MAX_MINUTES = "$max"; DEADLINE_EPOCH = "$deadline"; LAUNCH_EPOCH = "$launch"; GPU_TIER_OVERRIDE = ''
        PRICE_PER_HR = $g.pricePerHr.ToString($Inv)
        HF_TOKEN = '{{ RUNPOD_SECRET_HF_TOKEN }}'; PUBLIC_KEY = $pubKey; SSH_PUBLIC_KEY = $pubKey
    } | ConvertTo-Json -Compress
    # 4. Create (no --wait).
    $res = runpodctl pod create --name $name --cloud-type SECURE --gpu-id $g.gpuId --network-volume-id $Cfg.network_volume_id `
        --image $Cfg.image --min-cuda-version $Cfg.min_cuda_version --container-disk-in-gb $Cfg.container_disk_gb `
        --ports '22/tcp' --ssh --env $envJson --docker-args $start -o json 2>&1
    $out = ($res | ForEach-Object { "$_" }) -join "`n"
    $p = ConvertFrom-JsonOrNull (($res | Where-Object { $_ -isnot [System.Management.Automation.ErrorRecord] }) -join "`n")
    if ($LASTEXITCODE -eq 0 -and $p.id -and -not $p.error) {
        Log "  -> created pod $($p.id) ($name)"
        return @{ id = $p.id; name = $name; gpu = $g; max = $max; deadline = $deadline; launch = $launch }
    }
    if ($out -match '(?i)instances available|not available|no longer any|capacity|out of stock|insufficient|no gpu') {
        Log "  -> $($g.gpuId): no capacity, moving on"
        return $null
    }
    Log "  -> $($g.gpuId): create failed: $out"
    throw "pod create failed: $out"
}

$r = $null
$only = if ($Cfg.phase_gpu_order -and $Cfg.phase_gpu_order[$Phase]) { @($Cfg.phase_gpu_order[$Phase]) } else { @() }   # a phase may restrict the GPUs and their order (SMOKE5: A100s, then L40)
for ($round = 0; -not $r -and $round -le 4; $round++) {
    if ($round) { Log "no capacity in list; retry $round/4 in 15 min"; Start-Sleep -Seconds 900 }
    $cands = @(& "$PSScriptRoot/gputypes.ps1" -PassThru)
    if ($only.Count) { $cands = @($cands | Where-Object { $only -contains $_.gpuId } | Sort-Object { $only.IndexOf($_.gpuId) }) }
    foreach ($g in $cands) { if ($r = Start-Pod $g) { break } }
}
if (-not $r) { Log "NO_CAPACITY ($runId)"; exit 3 }

# 7. pod.json now (ip/port filled in if pod get ever shows them), then wait for the first heartbeat on the volume.
$pod = [ordered]@{
    id = $r.id; name = $r.name; gpu = $r.gpu.gpuId; price = $r.gpu.pricePerHr; dc = $Cfg.datacenter; ip = $null; port = $null
    created = [DateTimeOffset]::FromUnixTimeSeconds($r.launch).UtcDateTime.ToString('o'); max_minutes = $r.max; deadline_epoch = $r.deadline
}
Save-PodJson $pod
$last = ''
while ((Get-Epoch) - $r.launch -lt 600) {
    $pg = Get-Pod $r.id
    if ($pg.publicIp -and $pg.portMappings.'22' -and -not $pod.port) {
        $pod.ip = $pg.publicIp; $pod.port = $pg.portMappings.'22'; Save-PodJson $pod; Log "ssh port assigned: $($pod.ip):$($pod.port)"
    }
    $st = if ($pg) { "pod get: $($pg.desiredStatus)/$($pg.runtimeStatus)" } else { 'pod get: not visible' }
    if ($st -ne $last) { Log "+$((Get-Epoch) - $r.launch) s $st"; $last = $st }
    if ($s = ConvertFrom-JsonOrNull (Get-S3Text "runs/$runId/status.json")) {
        Log "+$((Get-Epoch) - $r.launch) s heartbeat: state=$($s.state) step=$($s.step)"
        $local = [DateTimeOffset]::FromUnixTimeSeconds($r.deadline).LocalDateTime
        "RUN_ID=$runId  pod=$($r.id)  gpu=$($r.gpu.gpuId)  `$/h=$($r.gpu.pricePerHr.ToString($Inv))  max_minutes=$($r.max)  deadline=$local (local)  ssh_port=$(if ($pod.port) { $pod.port } else { 'none yet' })"
        exit 0
    }
    Start-Sleep -Seconds 15
}

# 2. No heartbeat in 10 min: kill, report, show what the volume has for this run.
Log "FAILED: no heartbeat ($runId, pod $($r.id)) after 10 min"
& "$PSScriptRoot/kill.ps1" $r.id | ForEach-Object { Log "  $_" }
'--- volume contents for this run:' | Out-Host
aws @S3 s3 ls "$Bucket/runs/$runId/" --recursive 2>&1 | Out-Host
exit 2
