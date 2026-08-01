@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv not found
  pause
  exit /b 1
)
echo Open http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run app.py
pause