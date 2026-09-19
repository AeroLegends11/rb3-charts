param(
    [string]$Python = "python",
    [switch]$CpuOnly
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
function Run-Python {
    & $script:VenvPython @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $args" }
}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 or 3.12 is required. Supply -Python with its full path.' }
}
$script:VenvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
Run-Python -m pip install --upgrade pip
# resampy 0.4.2 (used by ADTOF) still imports pkg_resources. Setuptools 81+
# removed that compatibility module, so retain the final supported series.
Run-Python -m pip install setuptools==80.9.0
$index = if ($CpuOnly) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
Run-Python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url $index
Run-Python -m pip install -e . pytest dora-search einops julius lameenc openunmix resampy==0.4.2
# Basic Pitch's Python >=3.11 dependency metadata requests obsolete TensorFlow.
# We explicitly use its bundled ONNX model; all actual runtime deps are above.
Run-Python -m pip install basic-pitch==0.4.0 demucs==4.0.1 --no-deps
Run-Python -m pip install 'git+https://github.com/xavriley/ADTOF-pytorch.git@85c192e78f716ea0b111cc8a5ee4a8f6a3a4f8a9' --no-deps
if (-not (Test-Path -LiteralPath 'tools\onyx\onyx-command-line-20251011-windows-x64\onyx.exe')) {
    New-Item -ItemType Directory -Force tools | Out-Null
    Invoke-WebRequest -Uri 'https://github.com/mtolly/onyx/releases/download/20251011/onyx-command-line-20251011-windows-x64.zip' -OutFile 'tools\onyx.zip'
    Expand-Archive -LiteralPath 'tools\onyx.zip' -DestinationPath 'tools\onyx' -Force
}
Run-Python -m pytest -q
Write-Host 'Ready. Double-click Launch.cmd. Models download on first generation.'
