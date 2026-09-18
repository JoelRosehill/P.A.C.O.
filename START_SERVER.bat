@echo off
rem P.A.C.O. server launcher. Sets up its own Python environment on first run (no admin).
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo Python 3.10 or newer is required. Install it from python.org or the Microsoft Store.
  pause & exit /b 1
)

if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo Created server\.env - fill in DEEPSEEK_API_KEY, PACO_TOKEN and DASHBOARD_PASSWORD, then run this again.
  notepad ".env"
  pause & exit /b 1
)

rem A .venv copied from another PC does not work here - rebuild it if so.
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import aiohttp, faster_whisper, serial" >nul 2>&1
  if errorlevel 1 (
    echo The Python environment is from another PC or incomplete - rebuilding it...
    rmdir /s /q ".venv"
  )
)
if not exist ".venv\Scripts\python.exe" (
  echo Setting up the server environment - first run only, takes a few minutes...
  python -m venv .venv || (pause & exit /b 1)
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (pause & exit /b 1)
)

rem Install anything new in requirements.txt (after a git pull). Quick when nothing changed.
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r requirements.txt || (pause & exit /b 1)

".venv\Scripts\python.exe" paco_server.py
pause
