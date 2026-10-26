param(
    [Parameter(Mandatory = $true)][string]$SourceWav,
    [Parameter(Mandatory = $true)][string]$ChartFile,
    [Parameter(Mandatory = $true)][string]$TempoFile,
    [Parameter(Mandatory = $true)][string]$OutputBundle,
    [string]$ModelRevision,
    [string]$WeightsSha256,
    [string]$Device = 'unknown',
    [string]$Seed,
    [string]$Temperature,
    [string]$TopK,
    [string]$RuntimeSeconds,
    [string]$Python = ".\.venv\Scripts\python.exe"
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '..')
if (-not (Test-Path -LiteralPath $Python)) { throw "Python executable not found: $Python" }
$arguments = @(
    '-m', 'rb3charts.cli', 'normalize-audio2chart-chart',
    $SourceWav, $ChartFile, $OutputBundle,
    '--tempo-file', $TempoFile,
    '--device', $Device
)
if ($ModelRevision) { $arguments += @('--model-revision', $ModelRevision) }
if ($WeightsSha256) { $arguments += @('--weights-sha256', $WeightsSha256) }
if ($Seed) { $arguments += @('--seed', $Seed) }
if ($Temperature) { $arguments += @('--temperature', $Temperature) }
if ($TopK) { $arguments += @('--top-k', $TopK) }
if ($RuntimeSeconds) { $arguments += @('--runtime-seconds', $RuntimeSeconds) }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "audio2chart chart normalization failed with exit code $LASTEXITCODE" }
