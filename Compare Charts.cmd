@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo RB3 Charts is not set up yet. Run Setup.ps1 first.
  pause
  exit /b 1
)
start "RB3 Chart Comparison" ".venv\Scripts\pythonw.exe" -m rb3charts.cli compare-gui
