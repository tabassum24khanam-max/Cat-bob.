#!/bin/bash
# ULTRAMAX Data Agent — 24/7 Background Pipeline
# Run this separately to continuously collect data

cd "$(dirname "$0")/backend"
source venv/bin/activate 2>/dev/null || true

echo "🤖 Starting ULTRAMAX Data Agent..."
echo "   Collects price, news, sentiment every 15-30 minutes"
echo "   Builds 5-year historical database"
echo "   Press Ctrl+C to stop"
echo ""

python3 data_agent.py
