@echo off
REM Starts the Z-Image Studio client. Works with a 32-bit Python installation.
setlocal
cd /d "%~dp0.."

REM Prefer an explicitly installed 32-bit interpreter, fall back to whatever
REM "python" resolves to.
set PY=py -3-32
%PY% -c "import sys" >nul 2>&1 || set PY=python

echo Starting Z-Image Studio client...
%PY% -m zimage_studio.client %*
if errorlevel 1 pause
endlocal
