@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [1/2] Creating virtualenv...
  python -m venv .venv
  if errorlevel 1 (
    echo Failed to create .venv. Is Python installed?
    pause
    exit /b 1
  )
)

echo Using: %cd%\.venv\Scripts\python.exe
".venv\Scripts\python.exe" -c "import streamlit,yfinance,plotly,pandas" 2>nul
if errorlevel 1 (
  echo [2/2] Installing dependencies...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo pip install failed.
    pause
    exit /b 1
  )
)

echo Starting Streamlit...
".venv\Scripts\python.exe" -m streamlit run app.py
pause
