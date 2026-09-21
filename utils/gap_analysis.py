from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import logging
from config import CACHE_TTL_SECONDS, STOCK_NAME_MAP

logging.basicConfig(level=logging.INFO)

@st.cache_data(ttl=300, show_spinner=False)
def analyze_intraday_gap_and_zones(symbol: str) -> dict:
    """
    Analyzes intraday 15-minute price action, VWAP, Opening Range Breakout (ORB),
    Pivot points, and Buy/Sell Zone boundaries for a given stock in parallel.
    """
    display_name = STOCK_NAME_MAP.get(symbol, symbol.replace(".NS", "").replace(".BO", ""))
    
    try:
        ticker = yf.Ticker(symbol)
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            f_15m = executor.submit(ticker.history, period="5d", interval="15m")
            f_5m = executor.submit(ticker.history, period="5d", interval="5m")
            f_daily = executor.submit(ticker.history, period="1mo", interval="1d")
            df_15m = f_15m.result()
            df_5m = f_5m.result()
            df_daily = f_daily.result()

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

        # 6. Compute Unified Intraday Order Flow Timeline
        bars_to_use = df_5m if (df_5m is not None and not df_5m.empty and len(df_5m) >= 3) else df_15m
        is_5m_flag = (bars_to_use is df_5m)
        timeline = compute_order_flow_timeline(bars_to_use, orb_high, orb_low, r1, r2, s1, s2, is_5m=is_5m_flag)

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
            "playbook": playbook,
            "timeline": timeline
        }
    except Exception as e:
        logging.error(f"Error in gap analysis for {symbol}: {e}")
        return {"status": "error", "message": str(e)}


def compute_order_flow_timeline(df_bars: pd.DataFrame, orb_high: float, orb_low: float, r1: float, r2: float, s1: float, s2: float, is_5m: bool = True) -> list:
    """
    Scans intraday candles chronologically to compute execution playbook timeline.
    """
    if df_bars is None or df_bars.empty:
        return []

    try:
        bars = df_bars.copy()
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)

        latest_date = bars.index[-1].date()
        today_bars = bars[bars.index.date == latest_date].copy()
        if today_bars.empty:
            today_bars = bars.tail(60 if is_5m else 20).copy()

        num_orb_bars = 3 if is_5m else 1
        if len(today_bars) < num_orb_bars:
            return []

        # Cumulative VWAP calculation
        typical_price = (today_bars['High'] + today_bars['Low'] + today_bars['Close']) / 3.0
        vp = typical_price * today_bars['Volume']
        cum_vp = vp.cumsum()
        cum_vol = today_bars['Volume'].cumsum()
        today_bars['VWAP'] = np.where(cum_vol > 0, cum_vp / cum_vol, today_bars['Close']).round(2)

        timeline = []
        first_c = round(today_bars.iloc[num_orb_bars - 1]['Close'], 2)
        timeline.append({
            "trigger_time": "09:30 AM",
            "signal": "🟡 HOLD",
            "signal_title": "Opening Range Setup",
            "type": "HOLD",
            "trigger_price": f"₹{first_c:.2f}",
            "entry_zone": f"₹{orb_low:.2f} - ₹{orb_high:.2f}",
            "target_1": f"₹{r1:.2f} (R1)",
            "target_2": f"₹{r2:.2f} (R2)",
            "stop_loss": f"₹{s1:.2f} (S1)",
            "outcome": "✅ Range Established (9:15 - 9:30 AM)",
            "outcome_time": "09:30 AM",
            "points": "0.00",
            "is_active": False,
            "badge_color": "#FFB300"
        })

        current_trade = None
        last_completed_type = None

        for i in range(num_orb_bars, len(today_bars)):
            bar = today_bars.iloc[i]
            bar_time = today_bars.index[i].strftime("%I:%M %p")
            close = round(bar['Close'], 2)
            high = round(bar['High'], 2)
            low = round(bar['Low'], 2)
            vwap = bar['VWAP']

            if current_trade:
                entry_p = current_trade['entry_num']
                t1 = current_trade['t1_num']
                t2 = current_trade['t2_num']
                sl = current_trade['sl_num']

                if current_trade['type'] == 'BUY':
                    if high >= t2:
                        pnl = round(t2 - entry_p, 2)
                        current_trade['outcome'] = f"🚀 Target 2 Hit at {bar_time} (₹{t2:.2f})"
                        current_trade['outcome_time'] = bar_time
                        current_trade['points'] = f"+₹{pnl:.2f}"
                        current_trade['is_active'] = False
                        current_trade['badge_color'] = "#00E676"
                        timeline.append(current_trade)
                        last_completed_type = 'BUY'
                        current_trade = None
                    elif high >= t1 and "Target 1 Hit" not in current_trade['outcome']:
                        pnl = round(t1 - entry_p, 2)
                        current_trade['t1_hit_time'] = bar_time
                        current_trade['outcome'] = f"🎯 Target 1 Hit at {bar_time} (₹{t1:.2f}) - Trailing"
                        current_trade['outcome_time'] = bar_time
                        current_trade['points'] = f"+₹{pnl:.2f}"
                    elif low <= sl:
                        pnl = round(sl - entry_p, 2)
                        current_trade['outcome'] = f"🛑 Stop Loss Hit at {bar_time} (₹{sl:.2f})"
                        current_trade['outcome_time'] = bar_time
                        current_trade['points'] = f"{pnl:.2f}"
                        current_trade['is_active'] = False
                        current_trade['badge_color'] = "#FF5252"
                        timeline.append(current_trade)
                        last_completed_type = 'BUY'
                        current_trade = None

                elif current_trade['type'] == 'SELL':
                    if low <= t2:
                        pnl = round(entry_p - t2, 2)
                        current_trade['outcome'] = f"🚀 Downside Target 2 Hit at {bar_time} (₹{t2:.2f})"
                        current_trade['outcome_time'] = bar_time
                        current_trade['points'] = f"+₹{pnl:.2f}"
                        current_trade['is_active'] = False
                        current_trade['badge_color'] = "#00E676"
                        timeline.append(current_trade)
                        last_completed_type = 'SELL'
                        current_trade = None
                    elif low <= t1 and "Target 1 Hit" not in current_trade['outcome']:
                        pnl = round(entry_p - t1, 2)
                        current_trade['t1_hit_time'] = bar_time
                        current_trade['outcome'] = f"🎯 Downside Target 1 Hit at {bar_time} (₹{t1:.2f}) - Trailing"
                        current_trade['outcome_time'] = bar_time
                        current_trade['points'] = f"+₹{pnl:.2f}"
                    elif high >= sl:
                        pnl = round(entry_p - sl, 2)
                        current_trade['outcome'] = f"🛑 Stop Loss Hit at {bar_time} (₹{sl:.2f})"
                        current_trade['outcome_time'] = bar_time
                        current_trade['points'] = f"{pnl:.2f}"
                        current_trade['is_active'] = False
                        current_trade['badge_color'] = "#FF5252"
                        timeline.append(current_trade)
                        last_completed_type = 'SELL'
                        current_trade = None

            if not current_trade:
                if close >= orb_high and close > vwap and last_completed_type != 'BUY':
                    current_trade = {
                        "trigger_time": bar_time,
                        "signal": "🟢 WHEN TO BUY",
                        "signal_title": "Breakout Above VWAP & 15m High",
                        "type": "BUY",
                        "trigger_price": f"₹{close:.2f}",
                        "entry_zone": f"₹{orb_high:.2f} - ₹{round(orb_high * 1.005, 2):.2f}",
                        "target_1": f"₹{r1:.2f} (R1)",
                        "target_2": f"₹{r2:.2f} (R2)",
                        "stop_loss": f"₹{vwap:.2f} (VWAP)",
                        "outcome": f"⏳ In Progress (LTP: ₹{close:.2f})",
                        "outcome_time": "Running",
                        "points": "0.00",
                        "is_active": True,
                        "badge_color": "#00E676",
                        "entry_num": close,
                        "t1_num": r1,
                        "t2_num": r2,
                        "sl_num": vwap
                    }
                    last_completed_type = None
                elif close <= orb_low and close < vwap and last_completed_type != 'SELL':
                    current_trade = {
                        "trigger_time": bar_time,
                        "signal": "🔴 WHEN TO SELL / PUT",
                        "signal_title": "Breakdown Below VWAP & 15m Low",
                        "type": "SELL",
                        "trigger_price": f"₹{close:.2f}",
                        "entry_zone": f"Below ₹{orb_low:.2f} / ₹{vwap:.2f}",
                        "target_1": f"₹{s1:.2f} (S1)",
                        "target_2": f"₹{s2:.2f} (S2)",
                        "stop_loss": f"₹{vwap:.2f} (Above VWAP)",
                        "outcome": f"⏳ In Progress (LTP: ₹{close:.2f})",
                        "outcome_time": "Running",
                        "points": "0.00",
                        "is_active": True,
                        "badge_color": "#FF5252",
                        "entry_num": close,
                        "t1_num": s1,
                        "t2_num": s2,
                        "sl_num": vwap
                    }
                    last_completed_type = None
                elif orb_low < close < orb_high:
                    last_completed_type = None

        if current_trade:
            latest_close = round(today_bars.iloc[-1]['Close'], 2)
            entry_p = current_trade['entry_num']
            if current_trade['type'] == 'BUY':
                unrealized = round(latest_close - entry_p, 2)
            else:
                unrealized = round(entry_p - latest_close, 2)
            
            p_sign = "+" if unrealized >= 0 else ""
            if "Target 1 Hit" in current_trade['outcome']:
                hit_t = current_trade.get('t1_hit_time', current_trade['trigger_time'])
                t1_val = current_trade['t1_num']
                current_trade['outcome'] = f"🎯 Target 1 Hit at {hit_t} (₹{t1_val:.2f}) — Trailing to T2 (LTP: ₹{latest_close:.2f})"
                current_trade['outcome_time'] = hit_t
            else:
                current_trade['outcome'] = f"⏳ In Progress (Running since {current_trade['trigger_time']} | LTP: ₹{latest_close:.2f})"
                current_trade['outcome_time'] = "Running"
            current_trade['points'] = f"{p_sign}₹{unrealized:.2f}"
            timeline.append(current_trade)

        return timeline
    except Exception as e:
        logging.error(f"Error computing order flow timeline: {e}")
        return []


def generate_intraday_playbook_timeline(symbol: str) -> list:
    """
    Convenience wrapper: retrieves the computed timeline from analyze_intraday_gap_and_zones.
    """
    gap_info = analyze_intraday_gap_and_zones(symbol)
    return gap_info.get("timeline", [])
