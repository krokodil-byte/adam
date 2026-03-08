@echo off
setlocal
cd /d %~dp0
python install_runtime.py
if errorlevel 1 (
  echo.
  echo Install failed.
  pause
  exit /b %errorlevel%
)
echo.
echo Runtime ready. Starting chat...
python adamah_chat.py
pause
