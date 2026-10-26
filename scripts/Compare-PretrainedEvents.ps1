param(
    [Parameter(Mandatory = $true)][string]$ReferenceBundle,
    [Parameter(Mandatory = $true)][string[]]$PredictionBundles,
    [Parameter(Mandatory = $true)][string]$RightsManifest,
    [Parameter(Mandatory = $true)][string]$OutputReport,
    [double]$ToleranceMs = 50,
    [string]$Python = ".\.venv\Scripts\python.exe"
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '..')
if (-not (Test-Path -LiteralPath $Python)) { throw "Python executable not found: $Python" }
$arguments = @(
    '-m', 'rb3charts.cli', 'compare-pretrained-events', $ReferenceBundle
) + $PredictionBundles + @(
    '--rights-manifest', $RightsManifest,
    '--output', $OutputReport,
    '--tolerance-ms', "$ToleranceMs"
)
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Pretrained event comparison failed with exit code $LASTEXITCODE" }
