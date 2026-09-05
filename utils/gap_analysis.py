import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import logging
from config import CACHE_TTL_SECONDS, STOCK_NAME_MAP

logging.basicConfig(level=logging.INFO)

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def analyze_intraday_gap_and_zones(symbol: str) -> dict:
    """
    Analyzes intraday 15-minute price action, VWAP, Opening Range Breakout (ORB),
    Pivot points, and Buy/Sell Zone boundaries for a given stock.
    """
    display_name = STOCK_NAME_MAP.get(symbol, symbol.replace(".NS", "").replace(".BO", ""))
    
    try:
        ticker = yf.Ticker(symbol)
        
        # 1. Fetch 5-day 15-minute intraday data
        df_15m = ticker.history(period="5d", interval="15m")
        # Fetch daily data for pivot points calculation
        df_daily = ticker.history(period="1mo", interval="1d")

        if df_15m.empty or len(df_15m) < 10:
            return {
                "status": "error",
                "message": f"Insufficient intraday bar data for {symbol}."
            }

        # Localize timezone
        if df_15m.index.tz is not None:
            df_15m.index = df_15m.index.tz_localize(None)
        if df_daily.index.tz is not None:
            df_daily.index = df_daily.index.tz_localize(None)

        # Get latest trading day's intraday bars
        latest_date = df_15m.index[-1].date()
        today_bars = df_15m[df_15m.index.date == latest_date].copy()

        if today_bars.empty:
            today_bars = df_15m.tail(26).copy()

        # 2. Calculate VWAP (Volume Weighted Average Price) for current day
        typical_price = (today_bars['High'] + today_bars['Low'] + today_bars['Close']) / 3.0
        vp = typical_price * today_bars['Volume']
        cum_vp = vp.cumsum()
        cum_vol = today_bars['Volume'].cumsum()
        
        # Safe VWAP calculation
        vwap_series = np.where(cum_vol > 0, cum_vp / cum_vol, today_bars['Close'])
        today_bars['VWAP'] = vwap_series

        latest_bar = today_bars.iloc[-1]
        current_price = round(latest_bar['Close'], 2)
        current_vwap = round(latest_bar['VWAP'], 2)

        # 3. Calculate 15-Minute Opening Range (First 15m candle of the session)
        first_bar = today_bars.iloc[0]
        orb_high = round(first_bar['High'], 2)
        orb_low = round(first_bar['Low'], 2)
        open_price = round(first_bar['Open'], 2)

        opening_gap_rs = round(open_price - (df_daily.iloc[-2]['Close'] if len(df_daily) >= 2 else open_price), 2)
        opening_gap_pct = round((opening_gap_rs / (df_daily.iloc[-2]['Close'] if len(df_daily) >= 2 else open_price)) * 100, 2)

        # 4. Standard Pivot Points (Derived from previous daily bar)
        prev_day = df_daily.iloc[-2] if len(df_daily) >= 2 else df_daily.iloc[-1]
        p_high, p_low, p_close = prev_day['High'], prev_day['Low'], prev_day['Close']

        pivot = round((p_high + p_low + p_close) / 3.0, 2)
        r1 = round((2 * pivot) - p_low, 2)
        s1 = round((2 * pivot) - p_high, 2)
        r2 = round(pivot + (p_high - p_low), 2)
        s2 = round(pivot - (p_high - p_low), 2)

        # 5. Gap Continuation vs. Profit Booking Signal Engine
        vwap_diff = current_price - current_vwap
        vwap_diff_pct = round((vwap_diff / current_vwap) * 100, 2)

        if current_price >= orb_high and current_price > current_vwap:
            gap_signal = "🟢 GAP CONTINUATION (Going UP)"
            gap_badge = "🟢 BULLISH CONTINUATION"
            gap_explanation = (
                f"Price is trading ABOVE 15-min Opening High (₹{orb_high}) and ABOVE VWAP (₹{current_vwap}). "
                "Strong buyers are taking actual delivery. High probability of continued upside momentum."
            )
            recommendation = "BUY / HOLD (Ride the Trend)"
        elif current_price <= orb_low or current_price < current_vwap:
            gap_signal = "🔴 PROFIT BOOKING RISK (Fading DOWN)"
            gap_badge = "🔴 PROFIT BOOKING / FADE"
            gap_explanation = (
                f"Price has dropped BELOW VWAP (₹{current_vwap}) or 15-min Low (₹{orb_low}). "
                "Early buyers are liquidating/booking profit. High risk of fading down to Support 1 (₹{s1})."
            )
            recommendation = "EXIT / SHORT / WAIT (Avoid Fresh Buying)"
        else:
            gap_signal = "🟡 CONSOLIDATION (Rangebound)"
            gap_badge = "🟡 RANGEBOUND"
            gap_explanation = (
                f"Price is fluctuating inside 15-min Opening Range (₹{orb_low} - ₹{orb_high}) near VWAP (₹{current_vwap}). "
                "Market is digesting the open before taking a directional breakout."
            )
            recommendation = "WAIT FOR BREAKOUT above ₹{orb_high} or below ₹{orb_low}"

        # 6. Buy Zone / Sell Zone Level Classification
        if current_price <= s1 * 1.01:
            zone_status = "🟢 STRONG BUY ZONE"
            zone_color = "#00E676"
            zone_desc = f"Trading near Support S1 (₹{s1}) - S2 (₹{s2}). Favorable risk-reward for long entries."
        elif current_price >= r1 * 0.99:
            zone_status = "🔴 PROFIT BOOKING / SELL ZONE"
            zone_color = "#FF5252"
            zone_desc = f"Trading near Resistance R1 (₹{r1}) - R2 (₹{r2}). High risk of profit booking."
        else:
            zone_status = "🟡 NEUTRAL ZONE"
            zone_color = "#FFB300"
            zone_desc = f"Trading between Support S1 (₹{s1}) and Resistance R1 (₹{r1})."

        # Trade Levels
        stop_loss = round(s1 * 0.99, 2) if current_price >= pivot else round(orb_low * 0.99, 2)
        target_1 = round(r1, 2) if current_price < r1 else round(r2, 2)
        target_2 = round(r2, 2)
        risk_per_share = round(max(current_price - stop_loss, 1.0), 2)
        reward_per_share = round(max(target_1 - current_price, 1.0), 2)
        rr_ratio = round(reward_per_share / risk_per_share, 2)

        return {
            "status": "success",
            "symbol": symbol,
            "name": display_name,
            "current_price": current_price,
            "vwap": current_vwap,
            "vwap_diff_pct": vwap_diff_pct,
            "opening_gap_rs": opening_gap_rs,
            "opening_gap_pct": opening_gap_pct,
            "orb_high": orb_high,
            "orb_low": orb_low,
            "gap_signal": gap_signal,
            "gap_badge": gap_badge,
            "gap_explanation": gap_explanation,
            "recommendation": recommendation,
            "zone_status": zone_status,
            "zone_color": zone_color,
            "zone_desc": zone_desc,
            "pivot": pivot,
            "r1": r1,
            "r2": r2,
            "s1": s1,
            "s2": s2,
            "suggested_entry": current_price,
            "suggested_target_1": target_1,
            "suggested_target_2": target_2,
            "suggested_stop_loss": stop_loss,
            "risk_reward_ratio": f"1 : {rr_ratio}"
        }
    except Exception as e:
        logging.error(f"Error in gap analysis for {symbol}: {e}")
        return {"status": "error", "message": str(e)}
