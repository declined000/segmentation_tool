@echo off
setlocal ENABLEEXTENSIONS

REM One-time setup helper (Windows):
REM - Creates .venv-cyto2
REM - Installs requirements.txt

cd /d "%~dp0"

set "FROM_LAUNCHER=0"
if /I "%~1"=="--from-launcher" set "FROM_LAUNCHER=1"

set "LOG=%~dp0launcher.log"
echo.>> "%LOG%"
echo ===== %date% %time% SETUP =====>> "%LOG%"

set "REQ=requirements.txt"
if not exist "%REQ%" (
  echo Missing %REQ%>> "%LOG%"
  echo ERROR: Missing %REQ%
  pause
  exit /b 1
)

REM Prefer the Windows Python Launcher if present.
where py >nul 2>nul
if %errorlevel%==0 (
  set "SYSPY=py -3"
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    set "SYSPY=python"
  ) else (
    echo Python not found>> "%LOG%"
    echo ERROR: Python was not found. Install Python 3.10+ from python.org, then run again.
    pause
    exit /b 1
  )
)

echo Creating venv...>> "%LOG%"
%SYSPY% -m venv ".venv-cyto2" >> "%LOG%" 2>&1
if not exist ".venv-cyto2\Scripts\python.exe" (
  echo Venv creation failed>> "%LOG%"
  echo ERROR: Failed to create .venv-cyto2. See launcher.log
  pause
  exit /b 1
)

echo Installing requirements (this can take a while)...>> "%LOG%"
".venv-cyto2\Scripts\python.exe" -m pip install -r "%REQ%" >> "%LOG%" 2>&1
if %errorlevel% neq 0 (
  echo pip install failed>> "%LOG%"
  echo ERROR: Install failed. See launcher.log
  pause
  exit /b 1
)

echo Setup complete.>> "%LOG%"
echo Setup complete.>> "%LOG%"
if %FROM_LAUNCHER%==0 (
  echo Setup complete. You can now run Run_Electrotaxis_App.bat
  pause
)
exit /b 0

endlocal

