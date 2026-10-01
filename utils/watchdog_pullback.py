import sys
import os
import time
import json
import logging
from datetime import datetime

# UTF-8 stdout fix for Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.notifications import send_telegram_alert
import yfinance as yf

# Windows audio alert
try:
    import winsound
    def play_chime():
        for freq in [1000, 1400, 1800]:
            winsound.Beep(freq, 250)
except Exception:
    def play_chime():
        pass

def get_calibrated_premium(spot: float, base_spot=1273.0, base_prem=40.0):
    # Delta for ITM/ATM Put is approximately -0.58
    delta = 0.58
    prem = base_prem + (base_spot - spot) * delta
    return round(max(prem, 1.0), 2)

def run_watchdog():
    logging.info("Starting Calibrated CDSL Pullback & Breakdown Watchdog...")
    
    update_msg = (
        "🔄 *CDSL Watchdog Calibrated to Live Groww Market LTP!*\n\n"
        "🎟️ *Contract:* CDSL ₹1280 PE (October Expiry)\n"
        "💵 *Live Base LTP:* ₹40.00 (Spot @ ₹1,273.00)\n"
        "🔍 *Active Watch Conditions:*\n"
        "1. *VWAP Pullback:* Spot bounces to ₹1,275–₹1,277 (Premium cools to ₹37.50–₹38.80)\n"
        "2. *Breakdown:* Spot breaks below morning low ₹1,262.00 (Premium > ₹46.00)\n\n"
        "⚡ Instant phone buzzer and sound chime will fire upon trigger!"
    )
    send_telegram_alert(update_msg)
    print("Telegram calibration update sent.", flush=True)

    symbol = "CDSL.NS"
    target_strike = 1280
    morning_low = 1262.70
    
    while True:
        now = datetime.now()
        
        # Hard cutoff at 02:30 PM (no entries after 2:30 PM)
        if now.hour > 14 or (now.hour == 14 and now.minute >= 30):
            msg = "⏰ 02:30 PM cutoff reached. Watchdog stopped. No new entries allowed today."
            send_telegram_alert(msg)
            print(json.dumps({"status": "CUTOFF", "reason": msg}), flush=True)
            break

        try:
            t = yf.Ticker(symbol)
            df = t.history(period="1d", interval="5m")
            if df.empty:
                time.sleep(15)
                continue

            spot = round(float(df["Close"].iloc[-1]), 2)
            high = round(float(df["High"].max()), 2)
            low = round(float(df["Low"].min()), 2)
            
            typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
            cum_vol = df["Volume"].cumsum()
            vwap_series = (typical * df["Volume"]).cumsum() / cum_vol
            vwap = round(float(vwap_series.iloc[-1]), 2)
            
            prem = get_calibrated_premium(spot)

            last_open = float(df["Open"].iloc[-1])
            is_candle_red = spot <= last_open

            print(f"[{now.strftime('%H:%M:%S')}] Spot: Rs.{spot:.2f} | VWAP: Rs.{vwap:.2f} | 1280 PE Est: Rs.{prem:.2f}", flush=True)

            triggered = False
            trigger_type = ""

            # SETUP 1: VWAP Pullback Rejection (Spot in 1274 - 1277 and rejected at VWAP)
            if (1274.0 <= spot <= 1277.5) and is_candle_red:
                triggered = True
                trigger_type = "VWAP PULLBACK REJECTION"

            # SETUP 2: Breakdown Below Morning Low (< 1262.00)
            elif spot < 1261.50:
                triggered = True
                trigger_type = "BREAKDOWN BELOW MORNING LOW"

            if triggered:
                play_chime()
                
                target_prem = round(prem + 10.0, 2)
                sl_prem = round(max(prem - 5.0, 0.5), 2)
                
                alert_text = (
                    f"🚨 *CDSL ENTRY TRIGGERED!* ({trigger_type})\n\n"
                    f"🎟️ *Option:* CDSL ₹{target_strike} PE (Oct Expiry)\n"
                    f"💵 *Entry Premium:* ~₹{prem:.2f}\n"
                    f"🎯 *Target Premium:* ₹{target_prem:.2f} (+₹10.00 | +₹9,500 on 2 Lots)\n"
                    f"🛑 *Stop Loss:* ₹{sl_prem:.2f} (-₹5.00 | -₹4,750 on 2 Lots)\n"
                    f"⚡ *Action:* BUY 2 LOTS (950 Qty) AT MARKET\n"
                    f"📍 *Spot:* ₹{spot:.2f} | *VWAP:* ₹{vwap:.2f}\n"
                    f"⏰ *Cutoff:* 03:05 PM hard exit"
                )
                send_telegram_alert(alert_text)
                
                result = {
                    "status": "TRIGGERED",
                    "trigger_type": trigger_type,
                    "contract": f"CDSL ₹{target_strike} PE (October Expiry)",
                    "entry_prem": prem,
                    "target_prem": target_prem,
                    "sl_prem": sl_prem,
                    "lots": 2,
                    "spot": spot,
                    "vwap": vwap
                }
                print(json.dumps(result), flush=True)
                break

        except Exception as e:
            logging.warning(f"Watchdog tick error: {e}")

        time.sleep(20)

if __name__ == "__main__":
    run_watchdog()
