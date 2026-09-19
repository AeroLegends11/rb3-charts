@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Run Setup.ps1 first.
  pause
  exit /b 1
)
start "RB3 Charts" ".venv\Scripts\pythonw.exe" -m rb3charts.cli gui
