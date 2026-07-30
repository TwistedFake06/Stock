@echo off
chcp 65001 >nul
cd /d "%~dp0\.."

echo ========================================
echo Running backtests: SPY / VOO / QQQ
echo ========================================

for %%S in (SPY VOO QQQ) do (
  echo.
  echo [%%S] Starting...
  py -m backtest.run_backtest --symbol %%S
  if errorlevel 1 (
    echo [%%S] Backtest failed.
    pause
    exit /b 1
  )
)

echo.
echo Done. CSV files are in backtest\results\
pause
