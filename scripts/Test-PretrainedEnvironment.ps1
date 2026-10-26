param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '..')

Write-Host "Diagnostics run on this PowerShell host: $([Environment]::MachineName)"
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    & $nvidiaSmi.Source --query-gpu=name,memory.total,memory.free --format=csv,noheader
    if ($LASTEXITCODE -ne 0) { Write-Warning "nvidia-smi returned exit code $LASTEXITCODE" }
} else {
    Write-Host 'nvidia-smi is unavailable; no NVIDIA GPU diagnostic is available on this host.'
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}
$probe = @'
import importlib.util
import json
import platform
import sys
report = {
    "environment": "local PowerShell host",
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch_installed": importlib.util.find_spec("torch") is not None,
    "cuda_available": False,
    "cuda_devices": [],
}
if report["torch_installed"]:
    try:
        import torch
        report["torch_version"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if report["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                report["cuda_devices"].append({
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "free_memory_bytes": free_bytes,
                    "total_memory_bytes": total_bytes,
                })
    except Exception as exc:
        report["torch_diagnostic_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(report, indent=2))
'@
& $Python -c $probe
if ($LASTEXITCODE -ne 0) { throw "Python GPU/runtime diagnostic failed with exit code $LASTEXITCODE" }
