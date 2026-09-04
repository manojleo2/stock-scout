import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import logging
import os
import json
from config import CACHE_TTL_SECONDS, STOCK_NAME_MAP, DEFAULT_WATCHLIST

logging.basicConfig(level=logging.INFO)

WATCHLIST_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "watchlist.json")

def load_saved_watchlist() -> list:
    """Load persisted watchlist from JSON file, falling back to default."""
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r") as f:
                saved = json.load(f)
                if isinstance(saved, list) and len(saved) > 0:
                    return saved
        except Exception as e:
            logging.error(f"Error loading watchlist from file: {e}")
    return DEFAULT_WATCHLIST.copy()

def save_persistent_watchlist(watchlist: list):
    """Save current watchlist to JSON file so it survives browser refreshes."""
    try:
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(watchlist, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving watchlist to file: {e}")

def clean_num(val, round_digits=2, default="N/A"):
    """Safely format numbers, handling None, NaN, and Inf."""
    if val is None or pd.isna(val):
        return default
    try:
        f = float(val)
        if pd.isna(f) or np.isinf(f):
            return default
        return round(f, round_digits)
    except (ValueError, TypeError):
        return default

def validate_ticker(symbol: str) -> bool:
    """Check if yfinance can retrieve recent price data for a symbol."""
    try:
        t = yf.Ticker(symbol)
        h = t.history(period="5d")
        return not h.empty and len(h) > 0
    except Exception:
        return False

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_stock_data(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given ticker symbol.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            logging.warning(f"No historical data returned for symbol: {symbol}")
            return pd.DataFrame()
        
        # Clean index timezone if present
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_stock_fundamentals(symbol: str) -> dict:
    """
    Fetch fundamental metrics (P/E, Market Cap, 52W High/Low, ROE, etc.)
    Uses fast_info for real-time speed, falling back to info dict.
    """
    display_name = STOCK_NAME_MAP.get(symbol, symbol.replace(".NS", "").replace(".BO", ""))
    
    fallback = {
        "Symbol": symbol,
        "Name": display_name,
        "Current Price": "N/A",
        "Previous Close": "N/A",
        "Day Change": 0.0,
        "Day Change %": 0.0,
        "Market Cap (Cr ₹)": "N/A",
        "52W High": "N/A",
        "52W Low": "N/A",
        "Trailing P/E": "N/A",
        "Price to Book": "N/A",
        "Dividend Yield (%)": "N/A",
        "ROE (%)": "N/A",
        "Beta": "N/A"
    }
    
    try:
        ticker = yf.Ticker(symbol)
        
        cur_price = None
        prev_close = None
        mcap = None
        high_52 = None
        low_52 = None
        
        try:
            fast = ticker.fast_info
            cur_price = fast.last_price if pd.notna(fast.last_price) else None
            prev_close = fast.previous_close if pd.notna(fast.previous_close) else None
            mcap = fast.market_cap if pd.notna(fast.market_cap) else None
            high_52 = fast.year_high if pd.notna(fast.year_high) else None
            low_52 = fast.year_low if pd.notna(fast.year_low) else None
        except Exception as e:
            logging.warning(f"fast_info failed for {symbol}: {e}")
        
        # Fallback: use info dict for anything missing
        info = {}
        try:
            info = ticker.info or {}
        except Exception:
            pass
        
        if cur_price is None:
            cur_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if prev_close is None:
            prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        if mcap is None:
            mcap = info.get("marketCap")
        if high_52 is None:
            high_52 = info.get("fiftyTwoWeekHigh")
        if low_52 is None:
            low_52 = info.get("fiftyTwoWeekLow")
        
        # Format metrics safely using clean_num
        clean_price = clean_num(cur_price)
        clean_prev = clean_num(prev_close)
        
        day_change = 0.0
        day_change_pct = 0.0
        if isinstance(clean_price, (int, float)) and isinstance(clean_prev, (int, float)) and clean_prev != 0:
            day_change = round(clean_price - clean_prev, 2)
            day_change_pct = round(((clean_price - clean_prev) / clean_prev) * 100, 2)

        mcap_val = clean_num(mcap)
        mcap_cr = round(mcap_val / 1e7, 2) if isinstance(mcap_val, (int, float)) else "N/A"
        
        pe = clean_num(info.get("trailingPE"))
        pb = clean_num(info.get("priceToBook"))
        
        div_raw = info.get("dividendYield")
        div_yield = clean_num(div_raw * 100) if pd.notna(div_raw) and isinstance(div_raw, (int, float)) else "N/A"
        
        roe_raw = info.get("returnOnEquity")
        roe = clean_num(roe_raw * 100) if pd.notna(roe_raw) and isinstance(roe_raw, (int, float)) else "N/A"
        
        beta = clean_num(info.get("beta"))
        
        return {
            "Symbol": symbol,
            "Name": display_name,
            "Current Price": clean_price,
            "Previous Close": clean_prev,
            "Day Change": day_change,
            "Day Change %": day_change_pct,
            "Market Cap (Cr ₹)": mcap_cr,
            "52W High": clean_num(high_52),
            "52W Low": clean_num(low_52),
            "Trailing P/E": pe,
            "Price to Book": pb,
            "Dividend Yield (%)": div_yield,
            "ROE (%)": roe,
            "Beta": beta
        }
    except Exception as e:
        logging.error(f"Error fetching fundamentals for {symbol}: {e}")
        return fallback

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_multiple_stocks_data(symbols: list, period: str = "6m") -> dict:
    """Bulk download stock history for multiple symbols."""
    results = {}
    for s in symbols:
        df = get_stock_data(s, period=period)
        if not df.empty:
            results[s] = df
    return results
