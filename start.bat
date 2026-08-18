@echo off
echo ========================================
echo   ManifestoAI - Windows Startup
echo ========================================

set ROOT=%~dp0

echo.

REM Activate venv
call "%ROOT%venv\Scripts\activate.bat"

echo [1/3] Installing backend dependencies...
cd /d "%ROOT%backend"
python -m pip install -r requirements.txt

echo.

echo [2/3] Installing frontend dependencies...
cd /d "%ROOT%frontend"
call npm install

echo.

echo [3/3] Starting servers...

REM ---- START BACKEND ----
cd /d "%ROOT%"
start "Backend" cmd /k python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --no-access-log --log-level warning

REM ---- START FRONTEND ----
cd /d "%ROOT%frontend"
start "Frontend" cmd /k npm run dev

echo.
echo ========================================
echo   Servers Starting...
echo ========================================
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo.

pause
