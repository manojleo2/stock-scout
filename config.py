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
    "ITC.NS": "ITC Limited"
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
    "BSE": "BSE.NS"
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
