@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=C:\Users\forre\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PYTHON_EXE%" (
  echo Python 3.12 was not found at %PYTHON_EXE%
  pause
  exit /b 1
)
"%PYTHON_EXE%" -m forge.nature_sim_v2.demo --device cuda
if errorlevel 1 pause
endlocal
