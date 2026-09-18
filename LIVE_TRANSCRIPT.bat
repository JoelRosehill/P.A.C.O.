@echo off
rem Live transcript: talk to PACO over the USB cable and see your words here.
rem Usage: LIVE_TRANSCRIPT.bat [COM5] [--no-ask]
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run START_SERVER.bat once first to set up Python.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" tools\live.py %*
pause
