@echo off
setlocal
cd /d "%~dp0"

where git >nul 2>&1
if errorlevel 1 (
  echo Git is not available in PATH. Launching without update.
  goto launch
)

echo Updating repository...
git fetch --all --prune
if errorlevel 1 (
  echo git fetch failed. Launching current local version.
  goto launch
)

for /f "delims=" %%i in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%i"
if not defined CURRENT_BRANCH (
  echo Unable to determine current branch. Launching current local version.
  goto launch
)

if /i "%CURRENT_BRANCH%"=="HEAD" (
  echo Detached HEAD detected. Skipping git pull.
  goto launch
)

echo Pulling latest commits on %CURRENT_BRANCH%...
git pull --ff-only
if errorlevel 1 (
  echo git pull failed. Launching current local version.
)

:launch
call "%~dp0Launch.cmd"
endlocal
