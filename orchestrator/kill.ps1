# kill.ps1 <pod id> | kill.ps1 --all   (deletes pods; volume-attached pods can only be terminated)
param([string]$Target, [switch]$All)   # pwsh binds `--all` to -All
. "$PSScriptRoot/common.ps1"
if (-not $All -and -not $Target) { throw 'usage: kill.ps1 <pod id> | kill.ps1 --all' }

$ids = if ($All) { @(Get-Pods | Where-Object name -like 'optd-*' | ForEach-Object id) } else { @($Target) }
foreach ($id in $ids) {
    $out = (runpodctl pod delete $id 2>&1) -join ' '
    if ($LASTEXITCODE -ne 0) { Write-Warning "delete $id failed (exit $LASTEXITCODE): $out — if the pod is unlisted, check the RunPod console" }
    else { "deleted $id"; "$((Get-Date).ToString('s')) $id" | Add-Content (Join-Path $PSScriptRoot 'killed.log') }
}
$left = @(Get-Pods | Where-Object name -like 'optd-*')
"optd-* pods remaining (listed): $($left.Count)"
if ($All -and $left.Count) { exit 1 }
