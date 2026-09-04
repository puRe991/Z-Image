@echo off
REM Starts the Z-Image backend. Needs a 64-bit Python with PyTorch installed.
setlocal
cd /d "%~dp0.."

set PY=python
if not "%ZIMAGE_PYTHON%"=="" set PY=%ZIMAGE_PYTHON%

echo Starting Z-Image Studio server (this loads the model weights)...
%PY% -m zimage_studio.server --host 0.0.0.0 --port 8787 %*
if errorlevel 1 pause
endlocal
