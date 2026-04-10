#!/bin/bash
# ULTRAMAX v3.0 — Start Script
# Run this once to start everything

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║     ULTRAMAX v3.0 — Vibe Developing      ║"
echo "╚══════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")/backend"

# ── Check Python ────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "❌ Python3 not found. Install from python.org"
  exit 1
fi

# ── Install dependencies (first run only) ───────────────────────────────────
if [ ! -d "venv" ]; then
  echo "📦 First run — installing dependencies..."
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip -q
  # Install core deps (torch separately for size)
  pip install fastapi uvicorn httpx aiohttp aiosqlite numpy scipy scikit-learn \
    xgboost pandas python-dotenv feedparser beautifulsoup4 lxml pydantic \
    apscheduler pytz python-multipart websockets -q
  echo "✓ Dependencies installed"
  echo ""
  echo "⚠ Optional: For FinBERT NLP (better news sentiment):"
  echo "   pip install transformers torch"
  echo ""
else
  source venv/bin/activate
fi

# ── Create .env if missing ───────────────────────────────────────────────────
if [ ! -f ".env" ]; then
  if [ -f ".env.example" ]; then
    cp .env.example .env
    echo "📝 Created .env from .env.example"
    echo "   Edit backend/.env to add your API keys (or set them in the UI)"
  fi
fi

# ── Create __init__.py ───────────────────────────────────────────────────────
touch agents/__init__.py

# ── Start backend ────────────────────────────────────────────────────────────
echo "🚀 Starting ULTRAMAX Backend on http://localhost:8000"
echo ""
echo "   Open frontend/index.html in Chrome"
echo "   Or visit: http://localhost:8000"
echo ""
echo "   Press Ctrl+C to stop"
echo ""

python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
