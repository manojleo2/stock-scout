import yfinance as yf
import pandas as pd
import streamlit as st
import logging
from concurrent.futures import ThreadPoolExecutor
from config import CACHE_TTL_SECONDS

logging.basicConfig(level=logging.INFO)

# Global & Macro Benchmark Tickers
SP500_TICKER = "^GSPC"
NASDAQ_TICKER = "^IXIC"
INDIA_VIX_TICKER = "^INDIAVIX"
BANK_NIFTY_TICKER = "^NSEBANK"

def _fetch_single_ticker_history(ticker_sym: str, period: str) -> tuple:
    try:
        df = yf.Ticker(ticker_sym).history(period=period)
        return ticker_sym, df
    except Exception as e:
        logging.error(f"Error fetching history for {ticker_sym}: {e}")
        return ticker_sym, pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_macro_market_cues(period: str = "2y") -> pd.DataFrame:
    """
    Fetch and compute overnight global market cues, India VIX, and sector returns in parallel.
    """
    try:
        tickers = [SP500_TICKER, NASDAQ_TICKER, INDIA_VIX_TICKER, BANK_NIFTY_TICKER]
        results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_ticker = {
                executor.submit(_fetch_single_ticker_history, sym, period): sym 
                for sym in tickers
            }
            for future in future_to_ticker:
                sym, df = future.result()
                results[sym] = df

        sp500_df = results.get(SP500_TICKER, pd.DataFrame())
        nasdaq_df = results.get(NASDAQ_TICKER, pd.DataFrame())
        vix_df = results.get(INDIA_VIX_TICKER, pd.DataFrame())
        bank_df = results.get(BANK_NIFTY_TICKER, pd.DataFrame())

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

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_latest_macro_summary() -> dict:
    """
    Get real-time snapshot of S&P 500, Nasdaq, and India VIX without redundant network roundtrips.
    """
    try:
        macro_df = get_macro_market_cues(period="5d")
        
        sp_chg_pct = 0.0
        vix_price = None
        vix_chg_pct = 0.0
        
        if not macro_df.empty:
            if 'SP500_Ret1' in macro_df.columns and not macro_df['SP500_Ret1'].dropna().empty:
                sp_chg_pct = round(macro_df['SP500_Ret1'].dropna().iloc[-1] * 100, 2)
            
            if 'VIX_Close' in macro_df.columns and not macro_df['VIX_Close'].dropna().empty:
                vix_price = float(macro_df['VIX_Close'].dropna().iloc[-1])
            
            if 'VIX_Ret1' in macro_df.columns and not macro_df['VIX_Ret1'].dropna().empty:
                vix_chg_pct = round(macro_df['VIX_Ret1'].dropna().iloc[-1] * 100, 2)

        if vix_price is None:
            try:
                sp = yf.Ticker(SP500_TICKER).fast_info
                vix = yf.Ticker(INDIA_VIX_TICKER).fast_info
                sp_price = getattr(sp, 'last_price', None)
                sp_prev = getattr(sp, 'previous_close', None)
                sp_chg_pct = round(((sp_price - sp_prev) / sp_prev) * 100, 2) if (sp_price and sp_prev) else 0.0
                vix_price = getattr(vix, 'last_price', None)
                vix_prev = getattr(vix, 'previous_close', None)
                vix_chg_pct = round(((vix_price - vix_prev) / vix_prev) * 100, 2) if (vix_price and vix_prev) else 0.0
            except Exception:
                pass

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
