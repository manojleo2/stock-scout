@echo off
title Scout CDSL 1-Min Live Watchdog
cd /d "C:\Users\manoj\scout\stock_scout"
echo ========================================================
echo   SCOUT CDSL 1-MIN LIVE EXECUTION WATCHDOG
echo   Streaming 1-min candles and Telegram notifications...
echo ========================================================
python utils\live_1min_watchdog.py
pause
