# launch.ps1 -Phase <P> [-Seed n] [-MaxMinutes m] [-DeadlineEpoch e]   (§4.1; the overrides exist for the watchdog test)
param(
    [Parameter(Mandatory)][ValidateSet('WATCHDOGTEST', 'ENVCHECK', 'SMOKE', 'FULL')][string]$Phase,
    [int]$Seed = 0, [int]$MaxMinutes = 0, [long]$DeadlineEpoch = 0
)
. "$PSScriptRoot/common.ps1"
Set-Location $Root

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

$budget = [double]$Cfg.phase_budget_usd[$Phase]
$ceiling = [int]$Cfg.phase_ceiling_min[$Phase]
$pubKey = (Get-Content (Join-Path $HOME '.ssh/id_ed25519.pub') -Raw).Trim()
$start = "bash -lc `"curl -fsSL $($Cfg.git_raw)/$sha/pod/bootstrap.sh | bash`""

function Start-Pod($g) {
    # 3. Lifetime from this GPU's price.
    $max = if ($MaxMinutes) { $MaxMinutes } else { [math]::Min($ceiling, [math]::Floor($budget / $g.pricePerHr * 60)) }
    $deadline = if ($DeadlineEpoch) { $DeadlineEpoch } else { (Get-Epoch) + ($max + [int]$Cfg.deadline_grace_min) * 60 }
    $cost = $g.pricePerHr * $max / 60
    [string]::Format($Inv, 'budget: {0:0.00} $/h x {1} min = ${2:0.00} <= ${3:0.00}', $g.pricePerHr, $max, $cost, $budget) | Out-Host
    if ($cost -gt $budget) { throw 'budget exceeded' }
    $name = "optd-$runId-d$deadline"
    # 5. Env (integers as strings; HF_TOKEN is a RunPod Secret reference, never a value).
    $envJson = [ordered]@{
        RUN_ID = $runId; PHASE = $Phase; SEED = "$Seed"; GIT_REPO = $Cfg.git_repo; GIT_SHA = $sha
        MAX_MINUTES = "$max"; DEADLINE_EPOCH = "$deadline"; GPU_TIER_OVERRIDE = ''
        PRICE_PER_HR = $g.pricePerHr.ToString($Inv)
        HF_TOKEN = '{{ RUNPOD_SECRET_HF_TOKEN }}'; PUBLIC_KEY = $pubKey; SSH_PUBLIC_KEY = $pubKey
    } | ConvertTo-Json -Compress
    "creating $name on $($g.gpuId) ..." | Out-Host
    # 4. Create and block until SSH answers.
    $res = runpodctl pod create --name $name --cloud-type SECURE --gpu-id $g.gpuId --network-volume-id $Cfg.network_volume_id `
        --image $Cfg.image --min-cuda-version $Cfg.min_cuda_version --container-disk-in-gb $Cfg.container_disk_gb `
        --ports '22/tcp' --ssh --env $envJson --docker-args $start --wait --wait-timeout 10m -o json 2>&1
    $rc = $LASTEXITCODE
    $out = ($res | ForEach-Object { "$_" }) -join "`n"
    $kept = Get-Pods | Where-Object name -eq $name
    if ($rc -eq 0 -and $kept) {
        $p = ConvertFrom-JsonOrNull (($res | Where-Object { $_ -isnot [System.Management.Automation.ErrorRecord] }) -join "`n")
        if (-not $p.id) { $p = Get-Pod $kept.id }
        return @{ pod = $p; gpu = $g; max = $max; deadline = $deadline; name = $name }
    }
    # Failed: a kept pod (e.g. --wait timeout; the error JSON carries its id) is killed; capacity errors move to the
    # next GPU; anything else stops.
    $errId = ($res | ForEach-Object { ConvertFrom-JsonOrNull "$_" } | Where-Object { $_.id } | Select-Object -Last 1).id
    if ($errId -and $out -match 'no longer listed') {
        # The API hides a pod for minutes while it pulls its image (pod list --all and pod get both miss it), which makes
        # runpodctl --wait give up. Keep polling pod get; the in-pod deadline watchdog bounds the cost meanwhile.
        "--wait lost the pod while it was initializing; polling pod get for up to 10 min" | Out-Host
        $t1 = Get-Date; $last = ''
        while (((Get-Date) - $t1).TotalMinutes -lt 10) {
            $p = Get-Pod $errId
            if ($p.publicIp -and $p.portMappings.'22') { return @{ pod = $p; gpu = $g; max = $max; deadline = $deadline; name = $name } }
            $st = if ($p) { "$($p.desiredStatus)/$($p.runtimeStatus)" } else { 'not visible to pod get' }
            if ($st -ne $last) { "  +$([int]((Get-Date) - $t1).TotalSeconds) s: $st" | Out-Host; $last = $st }
            Start-Sleep -Seconds 15
        }
    }
    $keptId = if ($kept) { $kept.id } elseif ($errId -and (Get-Pod $errId)) { $errId }
    if ($keptId) {
        & "$PSScriptRoot/kill.ps1" $keptId | Out-Host
        throw "FAILED: no ssh ($runId): $out"
    }
    if ($out -match '(?i)instances available|not available|no longer any|capacity|out of stock|insufficient|no gpu') {
        "capacity: $($g.gpuId)" | Out-Host
        return $null
    }
    throw "pod create failed: $out"
}

$r = $null
for ($round = 0; -not $r -and $round -le 4; $round++) {
    if ($round) { "no capacity in list; retry $round/4 in 15 min" | Out-Host; Start-Sleep -Seconds 900 }
    foreach ($g in (& "$PSScriptRoot/gputypes.ps1" -PassThru)) { if ($r = Start-Pod $g) { break } }
}
if (-not $r) { throw "NO_CAPACITY ($runId)" }

# 7. Record pod.json locally and on the volume.
$p = $r.pod
$podJson = [ordered]@{
    id = $p.id; name = $r.name; gpu = $r.gpu.gpuId; price = $r.gpu.pricePerHr; dc = $Cfg.datacenter
    ip = $p.publicIp; port = $p.portMappings.'22'
    created = if ($p.createdAt) { $p.createdAt } else { (Get-Date).ToUniversalTime().ToString('o') }
    max_minutes = $r.max; deadline_epoch = $r.deadline
} | ConvertTo-Json
$podJson | Set-Content (Join-Path $runDir 'pod.json')
if (-not (Send-S3File (Join-Path $runDir 'pod.json') "runs/$runId/pod.json")) { Write-Warning 'pod.json upload to the volume failed' }
$local = [DateTimeOffset]::FromUnixTimeSeconds($r.deadline).LocalDateTime
"RUN_ID=$runId  pod=$($p.id)  gpu=$($r.gpu.gpuId)  `$/h=$($r.gpu.pricePerHr.ToString($Inv))  max_minutes=$($r.max)  deadline=$local (local)"
