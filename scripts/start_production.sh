#!/usr/bin/env bash
# Tereguwami (ተርጓሚ) Unix/Linux Production Startup Script
set -e

echo "======================================================================"
echo "  Launching Tereguwami (ተርጓሚ) AI Multimodal Platform & Web App"
echo "======================================================================"

python3 -c "import fastapi, uvicorn, numpy, sqlalchemy; print('Core dependencies verified.')"

echo "Applying Database Migrations & Seeding..."
python3 -c "from backend.db.migrations import run_migrations_and_seed; run_migrations_and_seed()"

echo "Starting Server on http://0.0.0.0:8000..."
exec python3 -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
