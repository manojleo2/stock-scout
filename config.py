import os

# Watchlist Defaults
DEFAULT_WATCHLIST = ["CDSL.NS", "NSDL.BO"]
BENCHMARK_TICKER = "^NSEI"  # Nifty 50 Index

STOCK_NAME_MAP = {
    "CDSL.NS": "Central Depository Services (CDSL)",
    "NSDL.BO": "National Securities Depository (NSDL)",
    "^NSEI": "Nifty 50 Index",
    "SBIN.NS": "State Bank of India (SBI)",
    "BSE.NS": "BSE Limited",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
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
    "TCS": "TCS.NS",
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
    "NSDL": "NSDL.BO",
    "BSE": "BSE.NS",
    "GADIUM": "GAUDIUMIVF.NS",
    "GAUDIUM": "GAUDIUMIVF.NS",
    "GAUDIUMIVF": "GAUDIUMIVF.NS"
}

# Data Settings
DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"
CACHE_TTL_SECONDS = 5  # Cache yfinance quotes for 5s for fast live refresh

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
ML_N_ESTIMATORS = 150

# 3:05 PM Gap Predictor Settings
MIN_GAP_CONVICTION_THRESHOLD = 65.0  # Minimum conviction (max(prob_up, 100-prob_up)) required for overnight carry

# CDSL Specialist Quant Parameters
CDSL_MIN_CONVICTION_WEEKDAY = 65.0   # Mon-Thu threshold (%)
CDSL_MIN_CONVICTION_FRIDAY = 70.0    # Strict Friday threshold (%) to survive 3-day weekend theta
CDSL_EXPIRY_WEEK_DAYS = 4            # Days before monthly expiry to trigger ITM strike shift
CDSL_LOT_SIZE = 475                  # Standard NSE CDSL option contract lot size


