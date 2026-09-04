@echo off
REM Starts the client with the built-in demo backend - no GPU, no weights.
setlocal
cd /d "%~dp0.."
set PY=py -3-32
%PY% -c "import sys" >nul 2>&1 || set PY=python
%PY% -m zimage_studio.client --demo
if errorlevel 1 pause
endlocal
