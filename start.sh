#!/bin/bash
set -e

echo "========================================"
echo "  ManifestoAI - Startup Script"
echo "========================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Install Python 3.10+ first."
    exit 1
fi

# Check Node
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js not found. Install Node.js 18+ first."
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "[1/4] Creating virtual environment..."
if [ ! -d "$ROOT_DIR/venv" ]; then
    python3 -m venv "$ROOT_DIR/venv"
fi
source "$ROOT_DIR/venv/bin/activate"

echo "[2/4] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r "$ROOT_DIR/backend/requirements.txt"

echo "[3/4] Checking demo artifacts..."
if [ ! -f "$ROOT_DIR/data/processed/predictions.json" ]; then
    echo "ERROR: Missing precomputed demo data. Restore data/processed/predictions.json before starting."
    exit 1
fi

echo "[4/4] Installing frontend dependencies..."
cd "$ROOT_DIR/frontend"
npm install --silent

echo "[5/5] Starting servers..."
echo ""
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo ""
echo "  Open http://localhost:3000 in your browser"
echo "  Click 'Run Pipeline' to start analysis"
echo ""
echo "  GROQ_API_KEY: Set in backend/.env for LLM features"
echo ""

# Start backend
cd "$ROOT_DIR"
source "$ROOT_DIR/venv/bin/activate"
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Start frontend
cd "$ROOT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo "Servers started (backend PID: $BACKEND_PID, frontend PID: $FRONTEND_PID)"
echo "Press Ctrl+C to stop."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" EXIT
wait
