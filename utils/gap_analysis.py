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
        prev_close = round(df_daily.iloc[-2]['Close'] if len(df_daily) >= 2 else open_price, 2)
        day_change_rs = round(current_price - prev_close, 2)
        day_change_pct = round((day_change_rs / prev_close) * 100, 2)
        day_high = round(today_bars['High'].max(), 2)
        day_low = round(today_bars['Low'].min(), 2)

        if current_price >= orb_high and current_price > current_vwap:
            gap_signal = "🟢 BULLISH MOMENTUM (Breakout Above VWAP)"
            gap_badge = "🟢 BULLISH CONTINUATION"
            active_verdict = "BUY / HOLD"
            verdict_color = "#00E676"
            gap_explanation = (
                f"Price is trading actively ABOVE 9:15 AM Opening High (₹{orb_high}) and ABOVE VWAP (₹{current_vwap}). "
                f"Aggressive institutional buying is underway. Upward momentum remains strong toward resistance."
            )
            recommendation = "🟢 BUY on pullbacks to VWAP or HOLD existing long positions."
            zone_status = "🟢 BULLISH EXPANSION ZONE"
            zone_color = "#00E676"
            zone_desc = f"Trading above VWAP (₹{current_vwap}). Next target R1 (₹{r1})."
            primary_target_1 = r1
            primary_target_2 = r2
            primary_stop_loss = current_vwap
            risk_per_share = round(max(current_price - primary_stop_loss, 1.0), 2)
            reward_per_share = round(max(primary_target_1 - current_price, 1.0), 2)
        elif current_price <= orb_low or current_price < current_vwap:
            gap_signal = "🔴 PROFIT BOOKING / BREAKDOWN (Below VWAP)"
            gap_badge = "🔴 WEAKNESS / SELL ZONE"
            active_verdict = "SELL / AVOID BUYING"
            verdict_color = "#FF5252"
            gap_explanation = (
                f"Price has fallen BELOW intraday VWAP (₹{current_vwap}) and broke 15-Min Low (₹{orb_low}). "
                f"Sellers are in full control. High probability of continued slide toward Support 1 (₹{s1})."
            )
            recommendation = "🔴 SELL / EXIT long positions or consider Put / Short entry. Avoid fresh buying."
            zone_status = "🔴 BREAKDOWN / SELL PRESSURE ZONE"
            zone_color = "#FF5252"
            zone_desc = f"Trading below VWAP (₹{current_vwap}). Testing downside Support S1 (₹{s1})."
            primary_target_1 = s1
            primary_target_2 = s2
            primary_stop_loss = current_vwap
            risk_per_share = round(max(primary_stop_loss - current_price, 1.0), 2)
            reward_per_share = round(max(current_price - primary_target_1, 1.0), 2)
        else:
            gap_signal = "🟡 CONSOLIDATION (Inside Opening Range)"
            gap_badge = "🟡 RANGEBOUND"
            active_verdict = "HOLD / WAIT"
            verdict_color = "#FFB300"
            gap_explanation = (
                f"Price is oscillating inside the 15-minute range (₹{orb_low} - ₹{orb_high}) near VWAP (₹{current_vwap}). "
                f"Neither buyers nor sellers have established dominance yet."
            )
            recommendation = f"🟡 HOLD existing positions. Wait for a breakout above ₹{orb_high} or breakdown below ₹{orb_low}."
            zone_status = "🟡 NEUTRAL RANGE ZONE"
            zone_color = "#FFB300"
            zone_desc = f"Consolidating between ₹{orb_low} and ₹{orb_high} around VWAP (₹{current_vwap})."
            primary_target_1 = r1
            primary_target_2 = s1
            primary_stop_loss = orb_low
            risk_per_share = round(max(current_price - orb_low, 1.0), 2)
            reward_per_share = round(max(r1 - current_price, 1.0), 2)

        rr_ratio = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 1.5

        # Precise Buy / Hold / Sell Playbook triggers
        playbook = {
            "buy": {
                "label": "WHEN TO BUY",
                "condition": f"Price breaks and sustains ABOVE ₹{orb_high} (15-min High) & ABOVE VWAP (₹{current_vwap})",
                "entry_zone": f"₹{orb_high} - ₹{round(orb_high * 1.005, 2)}",
                "target_1": f"₹{r1} (R1)",
                "target_2": f"₹{r2} (R2)",
                "stop_loss": f"₹{current_vwap} (VWAP)"
            },
            "hold": {
                "label": "WHEN TO HOLD",
                "condition": f"Price is trading inside ₹{orb_low} - ₹{orb_high} range without breaking either boundary",
                "action": "Hold current positions. DO NOT take aggressive new entries until a clean breakout occurs."
            },
            "sell": {
                "label": "WHEN TO SELL / EXIT / PUT",
                "condition": f"Price trades BELOW VWAP (₹{current_vwap}) or breaches BELOW ₹{orb_low} (15-min Low)",
                "action": "Exit longs immediately / Buy Put / Short",
                "downside_target_1": f"₹{s1} (S1)",
                "downside_target_2": f"₹{s2} (S2)",
                "stop_loss": f"₹{current_vwap} (Above VWAP)"
            }
        }

        return {
            "status": "success",
            "symbol": symbol,
            "name": display_name,
            "current_price": current_price,
            "open_price": open_price,
            "prev_close": prev_close,
            "day_change_rs": day_change_rs,
            "day_change_pct": day_change_pct,
            "day_high": day_high,
            "day_low": day_low,
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
            "active_verdict": active_verdict,
            "verdict_color": verdict_color,
            "zone_status": zone_status,
            "zone_color": zone_color,
            "zone_desc": zone_desc,
            "pivot": pivot,
            "r1": r1,
            "r2": r2,
            "s1": s1,
            "s2": s2,
            "suggested_entry": current_price,
            "suggested_target_1": primary_target_1,
            "suggested_target_2": primary_target_2,
            "suggested_stop_loss": primary_stop_loss,
            "risk_reward_ratio": f"1 : {rr_ratio}",
            "playbook": playbook
        }
    except Exception as e:
        logging.error(f"Error in gap analysis for {symbol}: {e}")
        return {"status": "error", "message": str(e)}
