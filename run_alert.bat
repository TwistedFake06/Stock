@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo Missing .venv — run run.bat first to create venv
  pause
  exit /b 1
)

REM Load .env into environment
if exist "%~dp0.env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%~dp0.env") do (
    if not "%%A"=="" if not "%%B"=="" set "%%A=%%B"
  )
  echo Loaded .env
)

echo.
echo Opening-hours 5-minute auto-alert every 5 minutes (US and Hong Kong sessions)
echo Telegram needs TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env
echo Ctrl+C to stop
echo.

"%VENV_PY%" "%~dp0scripts\watchlist_alert.py" --mode intraday --interval 300
pause
