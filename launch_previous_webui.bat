@echo off
setlocal EnableExtensions

title NULLVECTOR - Previous Web UI
set "SITE_DIR=%~dp0webui-neural"
set "SITE_URL=http://127.0.0.1:3000/"

if not exist "%SITE_DIR%\package.json" (
  echo [NULLVECTOR] Previous Web UI not found:
  echo %SITE_DIR%
  pause
  exit /b 1
)

where node.exe >nul 2>nul
if errorlevel 1 (
  echo [NULLVECTOR] Node.js is not available on PATH.
  pause
  exit /b 1
)

for /f "usebackq delims=" %%P in (`powershell.exe -NoProfile -Command "$c = Get-NetTCPConnection -State Listen -LocalPort 3000 -ErrorAction SilentlyContinue ^| Select-Object -First 1; if ($c) { $c.OwningProcess }"`) do set "LISTENER_PID=%%P"
if defined LISTENER_PID (
  echo [NULLVECTOR] Port 3000 is already in use by process %LISTENER_PID%.
  echo No second site instance was started.
  echo Close that process first, then run this launcher again.
  pause
  exit /b 2
)

cd /d "%SITE_DIR%"
if not exist "node_modules\.bin\vinext.cmd" (
  echo [NULLVECTOR] Installing the locked local dependencies once...
  call npm.cmd ci --ignore-scripts
  if errorlevel 1 (
    echo [NULLVECTOR] Dependency installation failed.
    pause
    exit /b 1
  )
)

echo.
echo [NULLVECTOR] Starting the previous neural Web UI at:
echo %SITE_URL%
echo.
echo This launcher deliberately does not open or publish anything.
echo Keep this window open while using the UI. Press Ctrl+C here to stop it.
echo.

call npm.cmd run dev -- --host 127.0.0.1 --port 3000 --strictPort
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo [NULLVECTOR] Web UI stopped with exit code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
