@echo off
setlocal ENABLEEXTENSIONS

REM Double-click launcher for the Streamlit GUI (Windows).
REM Uses the local venv at .venv-cyto2 and opens the browser automatically.

cd /d "%~dp0"

set "LOG=%~dp0launcher.log"
echo.>> "%LOG%"
echo ===== %date% %time% RUN =====>> "%LOG%"

REM First-run: create venv + install requirements automatically
if not exist ".venv-cyto2\Scripts\python.exe" (
  echo venv not found; running setup...>> "%LOG%"
  call "%~dp0Setup_Electrotaxis.bat" --from-launcher
)

if not exist ".venv-cyto2\Scripts\python.exe" (
  echo ERROR: venv still missing after setup>> "%LOG%"
  echo ERROR: Setup did not complete. See launcher.log
  pause
  exit /b 1
)

REM If venv exists but streamlit isn't installed (or is broken), run setup.
".venv-cyto2\Scripts\python.exe" -c "import streamlit" >nul 2>nul
if %errorlevel% neq 0 (
  echo streamlit missing; running setup...>> "%LOG%"
  call "%~dp0Setup_Electrotaxis.bat" --from-launcher
)

set "PORT=8501"

REM Open browser (Streamlit will bind shortly after)
start "" "http://localhost:%PORT%"

REM Start Streamlit in this window (most reliable). Logs also go to launcher.log.
echo Starting Streamlit on port %PORT%...>> "%LOG%"
"%~dp0.venv-cyto2\Scripts\python.exe" -m streamlit run "%~dp0streamlit_app.py" --server.address localhost --server.port %PORT% --server.headless true >> "%LOG%" 2>&1

endlocal

