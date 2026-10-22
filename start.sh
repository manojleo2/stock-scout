#!/usr/bin/env bash
set -e

export PYTHONUNBUFFERED=1
export TZ="Asia/Kolkata"

echo "🚀 Starting Nifty/Sensex 5-Second Sentinel in background..."
python -u -m utils.nifty_live_sentinel --ticker ^NSEI --interval 5 &

echo "🌐 Starting Streamlit Dashboard on port ${PORT:-8501}..."
streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
