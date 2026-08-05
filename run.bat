@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "STORE_PY=%LocalAppData%\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe"
set "PY311=%LocalAppData%\Programs\Python\Python311\python.exe"
set "PY312=%LocalAppData%\Programs\Python\Python312\python.exe"

if exist "%VENV_PY%" goto HAVE_VENV

set "BASE_PY="
if exist "%STORE_PY%" set "BASE_PY=%STORE_PY%"
if not defined BASE_PY if exist "%PY311%" set "BASE_PY=%PY311%"
if not defined BASE_PY if exist "%PY312%" set "BASE_PY=%PY312%"
if not defined BASE_PY set "BASE_PY=python"

echo Creating .venv ...
"%BASE_PY%" -m venv "%~dp0.venv"
if errorlevel 1 (
  echo Failed to create .venv
  echo Install Python 3.11 from python.org and retry.
  pause
  exit /b 1
)

:HAVE_VENV
if not exist "%VENV_PY%" (
  echo Missing venv python
  pause
  exit /b 1
)

echo Checking packages ...
"%VENV_PY%" -c "import streamlit,yfinance,plotly,pandas,scipy"
if errorlevel 1 (
  echo Installing requirements ...
  "%VENV_PY%" -m pip install --upgrade pip
  "%VENV_PY%" -m pip install -r "%~dp0requirements.txt"
  if errorlevel 1 (
    echo pip install failed
    pause
    exit /b 1
  )
)

REM Load local API keys from .env (gitignored) if present
if exist "%~dp0.env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%~dp0.env") do (
    if not "%%A"=="" if not "%%B"=="" set "%%A=%%B"
  )
  echo Loaded secrets from .env
)

echo Starting Streamlit at http://localhost:8501
echo Press Ctrl+C to stop.
"%VENV_PY%" -m streamlit run "%~dp0app.py"
pause