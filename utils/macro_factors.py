import yfinance as yf
import pandas as pd
import streamlit as st
import logging
from config import CACHE_TTL_SECONDS

logging.basicConfig(level=logging.INFO)

# Global & Macro Benchmark Tickers
SP500_TICKER = "^GSPC"
NASDAQ_TICKER = "^IXIC"
INDIA_VIX_TICKER = "^INDIAVIX"
BANK_NIFTY_TICKER = "^NSEBANK"

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_macro_market_cues(period: str = "2y") -> pd.DataFrame:
    """
    Fetch and compute overnight global market cues, India VIX, and sector returns.
    """
    try:
        sp500_df = yf.Ticker(SP500_TICKER).history(period=period)
        nasdaq_df = yf.Ticker(NASDAQ_TICKER).history(period=period)
        vix_df = yf.Ticker(INDIA_VIX_TICKER).history(period=period)
        bank_df = yf.Ticker(BANK_NIFTY_TICKER).history(period=period)

        macro_df = pd.DataFrame()

        if not sp500_df.empty:
            sp500_df.index = sp500_df.index.tz_localize(None)
            macro_df['SP500_Ret1'] = sp500_df['Close'].pct_change(1)
            macro_df['SP500_Close'] = sp500_df['Close']

        if not nasdaq_df.empty:
            nasdaq_df.index = nasdaq_df.index.tz_localize(None)
            macro_df['Nasdaq_Ret1'] = nasdaq_df['Close'].pct_change(1)

        if not vix_df.empty:
            vix_df.index = vix_df.index.tz_localize(None)
            macro_df['VIX_Close'] = vix_df['Close']
            macro_df['VIX_Ret1'] = vix_df['Close'].pct_change(1)
            macro_df['VIX_Norm'] = vix_df['Close'] / 100.0

        if not bank_df.empty:
            bank_df.index = bank_df.index.tz_localize(None)
            macro_df['BankNifty_Ret1'] = bank_df['Close'].pct_change(1)

        return macro_df
    except Exception as e:
        logging.error(f"Error fetching macro cues: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_latest_macro_summary() -> dict:
    """
    Get real-time snapshot of S&P 500, Nasdaq, and India VIX.
    """
    try:
        sp = yf.Ticker(SP500_TICKER).fast_info
        vix = yf.Ticker(INDIA_VIX_TICKER).fast_info

        sp_price = getattr(sp, 'last_price', None)
        sp_prev = getattr(sp, 'previous_close', None)
        sp_chg_pct = round(((sp_price - sp_prev) / sp_prev) * 100, 2) if (sp_price and sp_prev) else 0.0

        vix_price = getattr(vix, 'last_price', None)
        vix_prev = getattr(vix, 'previous_close', None)
        vix_chg_pct = round(((vix_price - vix_prev) / vix_prev) * 100, 2) if (vix_price and vix_prev) else 0.0

        if vix_price is not None:
            if vix_price < 13:
                vix_status = "Low Volatility (Bullish Climate)"
                vix_badge = "🟢"
            elif vix_price < 18:
                vix_status = "Moderate Volatility"
                vix_badge = "🟡"
            else:
                vix_status = "High Volatility (Fear Spike)"
                vix_badge = "🔴"
        else:
            vix_status = "N/A"
            vix_badge = "⚪"

        return {
            "sp500_change_pct": sp_chg_pct,
            "vix_level": round(vix_price, 2) if vix_price else "N/A",
            "vix_change_pct": vix_chg_pct,
            "vix_status": vix_status,
            "vix_badge": vix_badge
        }
    except Exception as e:
        logging.error(f"Error in macro summary: {e}")
        return {
            "sp500_change_pct": 0.0,
            "vix_level": "N/A",
            "vix_change_pct": 0.0,
            "vix_status": "N/A",
            "vix_badge": "⚪"
        }
