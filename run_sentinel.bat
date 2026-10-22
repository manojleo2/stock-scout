@echo off
title NIFTY & SENSEX 5-SECOND LIVE SENTINEL
cd /d C:\Users\manoj\scout\nifty_options_predictor
set PYTHONIOENCODING=utf-8
echo ========================================================
echo  STARTING NIFTY 5-SECOND REAL-TIME SENTINEL (GROWW PIPE)
echo ========================================================
python -m utils.nifty_live_sentinel --ticker ^NSEI --interval 5
pause
