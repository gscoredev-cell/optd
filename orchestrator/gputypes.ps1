# Candidate Secure Cloud GPUs for the volume's datacenter, in config gpu_order (§4.1 step 2). -PassThru returns the rows.
param([switch]$PassThru)
. "$PSScriptRoot/common.ps1"
$dc = $Cfg.datacenter
$order = @($Cfg.gpu_order)

$rows = foreach ($g in (runpodctl gpu list -o json | ConvertFrom-Json)) {
    if (-not $g.secureCloud -or $g.memoryInGb -lt 20 -or $order -notcontains $g.gpuId) { continue }
    $dcs = @($g.dataCenterAvailability | ForEach-Object { $_ })
    $here = $dcs | Where-Object dataCenterId -eq $dc
    if ($dcs.Count -and -not $here) { continue }   # listed elsewhere only: cannot attach the volume
    [pscustomobject]@{
        gpuId      = $g.gpuId
        memGb      = $g.memoryInGb
        pricePerHr = [double]$g.securePricePerHr
        dc         = if ($here) { $dc } else { 'unknown' }
        stock      = if ($here) { $here.stockStatus } else { '' }
        rank       = $(if ($here) { 0 } else { 100 }) + $order.IndexOf($g.gpuId)
    }
}
$rows = @($rows | Sort-Object rank)
for ($i = 0; $i -lt $rows.Count; $i++) { $rows[$i].rank = $i + 1 }

$rows | Format-Table @{n = '#'; e = { $_.rank } }, gpuId, memGb,
    @{n = '$/h'; e = { $_.pricePerHr.ToString('0.00', $Inv) } }, dc, stock -AutoSize | Out-Host
if (-not $rows.Count) { Write-Warning "no candidate GPU for $dc right now" }
if ($PassThru) { $rows }
