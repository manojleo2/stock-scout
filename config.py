import os

# Watchlist Defaults
DEFAULT_WATCHLIST = ["CDSL.NS", "HDFCBANK.NS"]
BENCHMARK_TICKER = "^NSEI"  # Nifty 50 Index

STOCK_NAME_MAP = {
    "CDSL.NS": "Central Depository Services (CDSL)",
    "^NSEI": "Nifty 50 Index",
    "SBIN.NS": "State Bank of India (SBI)",
    "BSE.NS": "BSE Limited",
    "RELIANCE.NS": "Reliance Industries",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS": "Tata Steel",
    "ITC.NS": "ITC Limited",
    "GAUDIUMIVF.NS": "Gaudium IVF & Women Health"
}

# Common user shortcuts to accurate NSE/BSE ticker symbols
COMMON_ALIASES = {
    "SBI": "SBIN.NS",
    "SBIN": "SBIN.NS",
    "RELIANCE": "RELIANCE.NS",
    "INFY": "INFY.NS",
    "INFOSYS": "INFY.NS",
    "HDFC": "HDFCBANK.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICI": "ICICIBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "ITC": "ITC.NS",
    "CDSL": "CDSL.NS",
    "BSE": "BSE.NS",
    "GADIUM": "GAUDIUMIVF.NS",
    "GAUDIUM": "GAUDIUMIVF.NS",
    "GAUDIUMIVF": "GAUDIUMIVF.NS"
}

# Data Settings
DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"
CACHE_TTL_SECONDS = 60  # Cache live quotes for 60s for high-speed sub-5s rendering

# Indicator Defaults
RSI_PERIOD = 14
SMA_SHORT = 50
SMA_LONG = 200
EMA_SHORT = 12
EMA_LONG = 26
EMA_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# ML Model Settings
ML_TEST_SIZE = 0.20  # 20% chronological test split
ML_MAX_DEPTH = 4     # Restrain tree depth to mitigate noise overfitting
ML_MIN_SAMPLES_LEAF = 20
ML_N_ESTIMATORS = 80 # Optimized tree count for ultra-fast training (<0.4s) with identical accuracy

# 3:05 PM Gap Predictor Settings
MIN_GAP_CONVICTION_THRESHOLD = 65.0  # Minimum conviction (max(prob_up, 100-prob_up)) required for overnight carry

# CDSL Specialist Quant Parameters
CDSL_MIN_CONVICTION_WEEKDAY = 65.0   # Mon-Thu threshold (%)
CDSL_MIN_CONVICTION_FRIDAY = 70.0    # Strict Friday threshold (%) to survive 3-day weekend theta
CDSL_EXPIRY_WEEK_DAYS = 4            # Days before monthly expiry to trigger ITM strike shift
CDSL_LOT_SIZE = 475                  # Standard NSE CDSL option contract lot size

# HDFCBANK Standalone Specialist Quant Parameters (Step B — Completely Isolated from CDSL)
HDFCBANK_MIN_CONVICTION_WEEKDAY = 65.0  # Mon-Thu threshold (%) — requires 65%+ for overnight banking carry
HDFCBANK_MIN_CONVICTION_FRIDAY = 70.0   # Strict Friday threshold (%) — 3-day weekend + banking theta decay
HDFCBANK_EXPIRY_WEEK_DAYS = 4           # Days before monthly last-Thursday expiry to trigger ITM shift
HDFCBANK_LOT_SIZE = 550                 # Standard NSE HDFCBANK option contract lot size (2026)
HDFCBANK_STRIKE_STEP = 10              # Strike price increment for HDFCBANK options (₹10 intervals)
HDFCBANK_BANKNIFTY_TICKER = "^NSEBANK" # Bank Nifty Index for co-integration feature computation
HDFCBANK_US_YIELD_TICKER = "^TNX"      # US 10-Year Treasury Yield (FII banking flow leading indicator)
HDFCBANK_LEARNING_WINDOW = 10          # Rolling window (sessions) for adaptive calibration offset
HDFCBANK_MAX_OFFSET_PCT = 10.0         # Hard cap on max calibration offset applied (+/- 10%)
HDFCBANK_OFFSET_STEP_PCT = 2.5         # Gradual shift per divergence signal (prevents whipsawing)

# Full-Day AI Up/Down Forecast Predictor Settings (Swing / Intraday Trend)
AI_MIN_CONVICTION_THRESHOLD = 60.0   # Minimum conviction (max(prob_up, prob_down)) to trigger directional swing call
AI_ATR_SL_MULT = 1.0                 # Stop Loss distance (1.0 x ATR14)
AI_ATR_TP1_MULT = 1.5                # Target 1 distance (1.5 x ATR14, 1.5:1 R:R)
AI_ATR_TP2_MULT = 2.5                # Target 2 distance (2.5 x ATR14, 2.5:1 R:R)


