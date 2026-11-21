@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Falta el entorno .venv. Consulta la guia de instalacion.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m uvicorn api:app --host 127.0.0.1 --port 8000
if errorlevel 1 (
  echo ERROR: No se pudo iniciar la API.
  pause
)
