@echo off
REM Tereguwami (ተርጓሚ) Windows Production Startup Script
echo ======================================================================
echo   Launching Tereguwami (ተርጓሚ) AI Multimodal Platform & Web App
echo ======================================================================
echo Checking environment...
python -c "import fastapi, uvicorn, numpy, sqlalchemy; print('All core Python packages verified.')"
if %errorlevel% neq 0 (
    echo [ERROR] Missing dependencies. Please run: pip install -r requirements.txt
    exit /b 1
)

echo Starting Database Migrations...
python -c "from backend.db.migrations import run_migrations_and_seed; run_migrations_and_seed()"

echo Launching FastAPI Server and Web App on http://127.0.0.1:8000
echo - Swagger Docs: http://127.0.0.1:8000/docs
echo - Product App:  http://127.0.0.1:8000/app
echo - 3D Avatar:    http://127.0.0.1:8000/avatar
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
