# pull.ps1 -RunId <id>   (§4.3) sync run files from the volume, then make sure the pod is gone.
param([Parameter(Mandatory)][string]$RunId)
. "$PSScriptRoot/common.ps1"
$runDir = Join-Path $Root "runs/$RunId"
$files = 'report.md', 'metrics.json', 'status.json', 'log.txt', 'env.json', 'pod.json'
$inc = $files | ForEach-Object { '--include', $_ }
aws @S3 s3 sync "$Bucket/runs/$RunId" $runDir --exclude '*' @inc --only-show-errors
if ($LASTEXITCODE -ne 0) { throw "s3 sync failed ($LASTEXITCODE)" }
Get-ChildItem $runDir -File | ForEach-Object { '{0,-14} {1,8} B' -f $_.Name, $_.Length }

$pod = Get-Content (Join-Path $runDir 'pod.json') -Raw | ConvertFrom-Json
$status = ConvertFrom-JsonOrNull (Get-Content (Join-Path $runDir 'status.json') -Raw -ErrorAction SilentlyContinue)
if ($status.ts -and ($c = ConvertTo-Epoch $pod.created)) {
    $min = ($status.ts - $c) / 60
    [string]::Format($Inv, 'final {0} | pod created -> final status {1:0.0} min x {2:0.00} $/h = ${3:0.00}', $status.state, $min, [double]$pod.price, $min * $pod.price / 60)
}
if (Get-Pods | Where-Object id -eq $pod.id) {
    & "$PSScriptRoot/kill.ps1" $pod.id
    'SELF_TERMINATE_FAILED' | Set-Content (Join-Path $runDir 'SELF_TERMINATE_FAILED')
    "SELF_TERMINATE_FAILED: pod $($pod.id) was still present and has been deleted"
} else { "pod $($pod.id) is gone" }
