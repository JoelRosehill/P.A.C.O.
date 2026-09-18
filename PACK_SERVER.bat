@echo off
rem Makes PACO-server.zip next to this folder: everything the server PC needs,
rem without this PC's Python environment, logs or test recordings.
rem The zip contains .env (API key, token, password) - keep it private.
setlocal
cd /d "%~dp0"
set "OUT=%~dp0..\PACO-server.zip"
set "STAGE=%TEMP%\paco-server-pack"
if exist "%STAGE%" rmdir /s /q "%STAGE%"
robocopy "%~dp0." "%STAGE%\server" /E /XD .venv __pycache__ logs recordings /XF PACO-server.zip /NFL /NDL /NJH /NJS /NP >nul
if exist "%OUT%" del "%OUT%"
echo Packing - the speech model is ~480 MB, this takes a minute...
powershell -NoProfile -Command "Compress-Archive -Path \"%STAGE%\server\" -DestinationPath \"%OUT%\" -CompressionLevel Fastest"
rmdir /s /q "%STAGE%"
if exist "%OUT%" (echo Done: %OUT%) else (echo Packing failed.)
pause
