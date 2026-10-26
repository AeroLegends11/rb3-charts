param(
    [Parameter(Mandatory = $true)][string]$SourceWav,
    [Parameter(Mandatory = $true)][string]$EventBundle,
    [Parameter(Mandatory = $true)][string]$OutputFolder,
    [string]$Title,
    [string]$Artist,
    [string]$Python = ".\.venv\Scripts\python.exe"
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '..')
if (-not (Test-Path -LiteralPath $Python)) { throw "Python executable not found: $Python" }
$arguments = @('-m', 'rb3charts.cli', 'import-pretrained-events', $SourceWav, $EventBundle, $OutputFolder)
if ($Title) { $arguments += @('--title', $Title) }
if ($Artist) { $arguments += @('--artist', $Artist) }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Pretrained event import failed with exit code $LASTEXITCODE" }
