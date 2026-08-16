@echo off
setlocal
set "PROJECT=%~dp0game"
set "GODOT=C:\Users\forre\Desktop\Godot_v4.3-stable_win64.exe"
if not exist "%GODOT%" (
  echo Godot 4.3 was not found at %GODOT%
  pause
  exit /b 1
)
start "NULLVECTOR Anatomical Demo" "%GODOT%" --path "%PROJECT%" res://AnatomicalDemo.tscn
endlocal
