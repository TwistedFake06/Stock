@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating .venv ...
  if exist "%LocalAppData%\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe" (
    "%LocalAppData%\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe" -m venv .venv
  ) else (
    python -m venv .venv
  )
  if not exist ".venv\Scripts\python.exe" (
    echo ERROR: cannot create .venv
    pause
    exit /b 1
  )
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)

.venv\Scripts\python.exe -c "import streamlit" 1>nul 2>nul
if errorlevel 1 (
  echo Installing packages ...
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)

echo.
echo ========================================
echo   Open browser: http://localhost:8501
echo   Menu: ????  -^> see timing block
echo   Stop: Ctrl+C
echo ========================================
echo.

start "" "http://localhost:8501"
.venv\Scripts\python.exe -m streamlit run app.py --server.headless true
pause