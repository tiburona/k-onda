@echo off
if "%~1"=="" (
  echo Usage: %0 "Commit message"
  exit /b 1
)
setlocal
set "msg=%~1"

rem Space-separated list of repo paths
set repos=. "..\k-onda-analysis"

for %%r in (%repos%) do (
  if exist "%%~r\.git\" (
    echo Committing changes in %%r...
    pushd "%%~r"
    git add -A
    git commit -m "%msg%"
    popd
  ) else (
    echo No Git repo found at %%r — skipping.
  )
)
endlocal