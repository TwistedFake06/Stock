@echo off
cd /d C:\Users\Eddie\Documents\GitHub\Stock\Stock
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
if not exist data mkdir data
".venv\Scripts\python.exe" -X utf8 confirm_digest_tonight.py > data\confirm_digest_open_20260929.log 2>&1
echo EXIT=%ERRORLEVEL%>> data\confirm_digest_open_20260929.log
