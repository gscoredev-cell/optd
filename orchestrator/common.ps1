# Dot-sourced by every orchestrator script: API key, config, S3 and pod helpers. Never prints secret values.
$ErrorActionPreference = 'Stop'
if (-not $env:RUNPOD_API_KEY) { $env:RUNPOD_API_KEY = [Environment]::GetEnvironmentVariable('RUNPOD_API_KEY', 'User') }
if (-not $env:RUNPOD_API_KEY) { throw 'RUNPOD_API_KEY is not set (user environment variable)' }
$env:AWS_MAX_ATTEMPTS = '10'   # the volume's S3 API returns occasional 502
$Root = Split-Path $PSScriptRoot -Parent
$Inv = [cultureinfo]::InvariantCulture

# config/run_defaults.yaml holds flat `key: <JSON value>` lines (JSON flow syntax is valid YAML).
function Get-Config {
    $c = @{}
    foreach ($l in Get-Content (Join-Path $Root 'config/run_defaults.yaml')) {
        if ($l -match '^([A-Za-z_]\w*):\s*(.+?)\s*$') { $c[$Matches[1]] = ConvertFrom-Json $Matches[2] -AsHashtable -NoEnumerate }
    }
    $c
}
$Cfg = Get-Config
$S3 = @('--profile', 'runpod', '--endpoint-url', $Cfg.s3_endpoint, '--region', $Cfg.s3_region)
$Bucket = "s3://$($Cfg.network_volume_id)"

function Get-S3Text([string]$Key) {
    $t = (& aws @S3 s3 cp "$Bucket/$Key" - 2>$null) -join "`n"
    if ($LASTEXITCODE -ne 0) { return $null }
    $t
}
function ConvertFrom-JsonOrNull($t) { if (-not $t) { return $null }; try { $t | ConvertFrom-Json } catch { $null } }
function Get-Pods { @(runpodctl pod list -o json | ConvertFrom-Json) }
function Get-Pod([string]$Id) {   # $null once the pod is gone
    $p = ConvertFrom-JsonOrNull ((runpodctl pod get $Id -o json 2>&1) -join "`n")
    if ($p.id -and -not $p.error) { $p } else { $null }
}
function Get-Epoch { [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() }
function ConvertTo-Epoch($v) {   # ISO or Go-style ("... +0000 UTC") timestamps; $null if unparseable
    if ($v -is [datetime]) { return ([DateTimeOffset]$v).ToUnixTimeSeconds() }
    try { ([DateTimeOffset]::Parse(($v -replace ' [A-Z]{3,4}$'), $Inv)).ToUnixTimeSeconds() } catch { $null }
}
