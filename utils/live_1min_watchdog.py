"""
CDSL Intraday V2 — 1-Minute Live Execution Watchdog & Telegram Notifier
Monitors 1-minute candlesticks for CDSL.NS in real time:
1. 09:15-09:20: Morning spike to VWAP +/- 1.5 sigma rejection -> Tranche 1 (1 Lot)
2. 09:20-09:25: 1-min VWAP confirmation close -> Tranche 2 (1 Lot / Full 2 Lots)
3. 09:25 onward: Precision SL (-Rs 5.00), Lot 1 Target (+Rs 10.00), 5-Min 20 EMA runner trail
4. Sends real-time Telegram push notifications to phone on every state transition.
"""

import os
import sys
import time
import datetime as dt
import logging
import json

# Ensure UTF-8 output on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import numpy as np
import pandas as pd

from utils.notifications import send_telegram_alert
from utils.options_feed import get_live_option_quote
from utils.market_calendar import is_trading_holiday
from utils.entry_conditions import (
    calculate_vwap_and_bands,
    is_rejection_candle,
    check_pre_trade_gates,
    _get_today_prediction
)

# Optional PC sound chime
try:
    import winsound
    def play_sound_alert():
        for freq in [1200, 1600, 2000]:
            winsound.Beep(freq, 200)
except Exception:
    def play_sound_alert():
        pass


class CDSL1MinWatchdog:
    def __init__(self, symbol: str = "CDSL.NS"):
        self.symbol = symbol
        self.state = 0  # 0: Scanning, 1: T1 Active, 2: Full Position Active, 3: Runner Only, 4: Exited
        self.direction = None
        self.is_call = False
        self.conviction = 0.0
        self.atm_strike = 1260
        self.option_contract = ""
        
        # Positions
        self.t1_prem = None
        self.t2_prem = None
        self.avg_entry = None
        self.active_sl = None
        self.lot1_target = None
        self.sl_trailed_state = "INITIAL"
        
        # Logging flags
        self.t1_alert_sent = False
        self.t2_alert_sent = False
        self.lot1_exit_sent = False
        self.trailed_cost_sent = False
        self.trailed_locked_sent = False
        self.closed_alert_sent = False

    def load_morning_model(self) -> bool:
        """Loads morning model forecast and validates conviction gate."""
        pred = _get_today_prediction(self.symbol)
        if not pred:
            logging.warning("No prediction found for today in prediction_audit.json.")
            return False

        pred_res = pred.get("pred_result", {}) or {}
        prob_up = float(pred_res.get("probability_up_pct", 50.0))
        prob_down = float(pred_res.get("probability_down_pct", 50.0))
        self.conviction = max(prob_up, prob_down)
        self.direction = "UP" if prob_up >= prob_down else "DOWN"
        self.is_call = (self.direction == "UP")

        # Weekday >= 60%, Friday >= 65%
        is_friday = (dt.date.today().weekday() == 4)
        min_conv = 65.0 if is_friday else 60.0

        if self.conviction < min_conv:
            logging.info(f"Model conviction {self.conviction:.1f}% below {min_conv:.0f}% threshold. Sidelined.")
            return False

        return True

    def run(self):
        logging.info("Starting CDSL 1-Minute Live Execution Watchdog...")
        today = dt.date.today()
        is_holiday, holiday_name = is_trading_holiday(today)
        if is_holiday or today.weekday() >= 5:
            logging.info(f"Market closed today ({holiday_name or 'Weekend'}). Exiting.")
            return

        has_model = self.load_morning_model()
        if not has_model:
            send_telegram_alert(
                f"⚪ *Scout Watchdog — Sidelined Today*\n\n"
                f"Model conviction ({self.conviction:.1f}%) below threshold. Sidelined to preserve capital."
            )
            return

        send_telegram_alert(
            f"🚀 *Scout 1-Min Real-Time Watchdog Active!*\n\n"
            f"📅 *{dt.datetime.now().strftime('%a, %d %b %Y')}*\n"
            f"🎯 *Forecast:* {self.direction} 📉 (Conviction: {self.conviction:.1f}%)\n"
            f"🔍 *Live Scanning:* Monitoring 1-min candles for VWAP rejection & VWAP confirmation."
        )

        while True:
            now = dt.datetime.now()
            # Stop watchdog after 03:05 PM cutoff
            if now.hour > 15 or (now.hour == 15 and now.minute >= 6):
                break

            try:
                self.tick(now)
            except Exception as e:
                logging.error(f"Watchdog tick error: {e}")

            time.sleep(15)

    def tick(self, now: dt.datetime):
        # Fetch 1-minute and 5-minute bars
        t = yf.Ticker(self.symbol)
        df_1m = t.history(period="1d", interval="1m")
        if df_1m.empty or len(df_1m) < 2:
            return

        current_spot = round(float(df_1m["Close"].iloc[-1]), 2)
        open_price = round(float(df_1m["Open"].iloc[0]), 2)
        self.atm_strike = round(current_spot / 20.0) * 20
        opt_type = "CE" if self.is_call else "PE"
        self.option_contract = f"CDSL ₹{self.atm_strike} {opt_type}"

        vwap, upper_band, lower_band, std_dev = calculate_vwap_and_bands(df_1m, sigma_mult=1.5)
        quote = get_live_option_quote(self.symbol, self.atm_strike, is_call=self.is_call, spot=current_spot)
        current_prem = float(quote.get("ltp") or 0.0)

        # ── 03:05 PM Hard Cutoff ──────────────────────────────────────────────
        if (now.hour == 15 and now.minute >= 5) or now.hour > 15:
            if self.state in (1, 2, 3) and not self.closed_alert_sent:
                self.state = 4
                self.closed_alert_sent = True
                play_sound_alert()
                msg = (
                    f"⏰ *03:05 PM HARD CUTOFF REACHED — EXIT ALL POSITIONS*\n\n"
                    f"🎟️ *Contract:* {self.option_contract}\n"
                    f"💵 *Exit Premium LTP:* ~₹{current_prem:.2f}\n"
                    f"📍 *Spot:* ₹{current_spot:,.2f}\n"
                    f"⚡ *Action:* CLOSE ALL ACTIVE LOTS AT MARKET"
                )
                send_telegram_alert(msg)
            return

        # ── STATE 0: SCANNING / NO POSITION ──────────────────────────────────
        if self.state == 0:
            # Check Branch A: Morning Spike Rejection (09:15 - 09:20 AM)
            last_bar = df_1m.iloc[-1]
            spike_condition = (current_spot >= upper_band) if not self.is_call else (current_spot <= lower_band)
            rejection = is_rejection_candle(last_bar, is_call=self.is_call)

            if spike_condition and rejection and not self.t1_alert_sent:
                gates_passed, _ = check_pre_trade_gates(quote, open_prem=current_prem)
                if gates_passed:
                    self.state = 1
                    self.t1_prem = current_prem
                    self.active_sl = round(max(current_prem - 8.0, 0.5), 2)  # Emergency SL -Rs 8.00
                    self.t1_alert_sent = True
                    play_sound_alert()

                    msg = (
                        f"🚨 *TRANCHE 1 ENTRY TRIGGERED! (1 Lot / 475 Qty)*\n\n"
                        f"🎟️ *Contract:* {self.option_contract}\n"
                        f"💵 *Fill Premium:* ~₹{self.t1_prem:.2f}\n"
                        f"🛑 *Opening Emergency SL:* ₹{self.active_sl:.2f} (-₹8.00 | -₹3,800 Max Risk)\n"
                        f"📍 *Trigger:* 1-Min Rejection Candle at VWAP ±1.5σ (Spot ₹{current_spot})\n"
                        f"⚡ *Action:* BUY 1 LOT (475 Qty) AT MARKET"
                    )
                    send_telegram_alert(msg)
                    return

            # Check Branch B: VWAP Confirmation (at/after 09:20 AM)
            if (now.hour == 9 and now.minute >= 20) or (now.hour > 9):
                vwap_confirmed = (current_spot < vwap) if not self.is_call else (current_spot > vwap)
                if vwap_confirmed and not self.t2_alert_sent:
                    # Single 2-lot entry if T1 wasn't triggered
                    self.state = 2
                    self.t1_prem = current_prem
                    self.t2_prem = current_prem
                    self.avg_entry = current_prem
                    self.active_sl = round(max(self.avg_entry - 5.0, 0.5), 2)
                    self.lot1_target = round(self.avg_entry + 10.0, 2)
                    self.t2_alert_sent = True
                    play_sound_alert()

                    msg = (
                        f"🚨 *FULL 2-LOT ENTRY TRIGGERED! (VWAP Confirmation)*\n\n"
                        f"🎟️ *Contract:* {self.option_contract}\n"
                        f"💵 *Fill Premium:* ~₹{self.avg_entry:.2f} (2 Lots / 950 Qty)\n"
                        f"🛑 *Precision SL:* ₹{self.active_sl:.2f} (-₹5.00 | -₹4,750 Max Risk)\n"
                        f"🎯 *Lot 1 Target:* ₹{self.lot1_target:.2f} (+₹10.00 | Lock +₹4,750 Cash)\n"
                        f"📍 *Spot:* ₹{current_spot:,.2f} (VWAP: ₹{vwap:,.2f})\n"
                        f"⚡ *Action:* BUY 2 LOTS (950 Qty) AT MARKET"
                    )
                    send_telegram_alert(msg)
                    return

        # ── STATE 1: TRANCHE 1 ACTIVE ────────────────────────────────────────
        elif self.state == 1:
            # Check Tranche 2 Trigger (1-min candle closes on correct side of VWAP)
            vwap_confirmed = (current_spot < vwap) if not self.is_call else (current_spot > vwap)
            if vwap_confirmed and not self.t2_alert_sent:
                self.state = 2
                self.t2_prem = current_prem
                self.avg_entry = round((self.t1_prem + self.t2_prem) / 2.0, 2)
                self.active_sl = round(max(self.avg_entry - 5.0, 0.5), 2)
                self.lot1_target = round(self.t1_prem + 10.0, 2)
                self.t2_alert_sent = True
                play_sound_alert()

                msg = (
                    f"🚨 *TRANCHE 2 ENTRY TRIGGERED! (1 Lot / 475 Qty)*\n\n"
                    f"🎟️ *Contract:* {self.option_contract}\n"
                    f"💵 *T2 Fill Prem:* ~₹{self.t2_prem:.2f} | *Blended Avg:* ~₹{self.avg_entry:.2f}\n"
                    f"🛑 *Precision SL Armed:* ₹{self.active_sl:.2f} (-₹5.00 from avg | -₹4,750 Max Risk)\n"
                    f"🎯 *Lot 1 Target:* ₹{self.lot1_target:.2f} (+₹10.00 | Lock +₹4,750)\n"
                    f"⚡ *Action:* BUY TRANCHE 2 (1 LOT / 475 Qty) — FULL 2 LOTS ACTIVE"
                )
                send_telegram_alert(msg)
                return

            # Check Emergency SL in State 1
            if current_prem <= self.active_sl:
                self.state = 4
                play_sound_alert()
                msg = f"🛑 *EMERGENCY SL HIT (-₹8.00) in Tranche 1.* Exit 1 Lot @ ₹{current_prem:.2f}."
                send_telegram_alert(msg)
                return

        # ── STATE 2: FULL POSITION ACTIVE (T1 + T2 = 2 Lots) ─────────────────
        elif self.state == 2:
            gain = current_prem - self.avg_entry

            # Trailing Rule 1: At +Rs 7.00 Gain -> Trail SL to Cost
            if gain >= 7.0 and not self.trailed_cost_sent:
                self.active_sl = self.avg_entry
                self.trailed_cost_sent = True
                play_sound_alert()
                send_telegram_alert(
                    f"🛡️ *SL TRAILED TO COST (ZERO RISK)*\n\n"
                    f"Option gain reached +₹{gain:.2f} (LTP ₹{current_prem:.2f}).\n"
                    f"🛑 *New SL:* ₹{self.active_sl:.2f} (Breakeven)."
                )

            # Trailing Rule 2: At +Rs 8.50 Gain -> Trail SL to +Rs 5.00
            elif gain >= 8.5 and not self.trailed_locked_sent:
                self.active_sl = round(self.avg_entry + 5.0, 2)
                self.trailed_locked_sent = True
                play_sound_alert()
                send_telegram_alert(
                    f"💰 *PROFIT LOCKED (+₹5.00 / +₹4,750)*\n\n"
                    f"Option gain reached +₹{gain:.2f} (LTP ₹{current_prem:.2f}).\n"
                    f"🛑 *New Locked SL:* ₹{self.active_sl:.2f} (+₹5.00 locked profit)."
                )

            # Lot 1 Target Hit (+Rs 10.00)
            if current_prem >= self.lot1_target and not self.lot1_exit_sent:
                self.state = 3  # Runner only
                self.lot1_exit_sent = True
                play_sound_alert()
                msg = (
                    f"🎯 *LOT 1 TARGET HIT (+₹10.00)!* 🚀\n\n"
                    f"🎟️ *Contract:* {self.option_contract}\n"
                    f"💵 *Exit Premium:* ₹{current_prem:.2f} (+₹10.00 / +₹4,750 Cash Locked)\n"
                    f"🏃 *Lot 2 Status:* RUNNER ACTIVE — Trailing using 5-Min 20 EMA\n"
                    f"⚡ *Action:* SELL LOT 1 (475 Qty) AT LIMIT / HOLD LOT 2 RUNNER"
                )
                send_telegram_alert(msg)
                return

            # Hard Stop Hit
            if current_prem <= self.active_sl:
                self.state = 4
                play_sound_alert()
                pnl = (current_prem - self.avg_entry) * 950
                msg = f"🛑 *STOP LOSS TRIGGERED @ ₹{current_prem:.2f}*. Exit both lots (P&L: ₹{pnl:+,.0f})."
                send_telegram_alert(msg)
                return

        # ── STATE 3: RUNNER ACTIVE (Lot 2 Only) ──────────────────────────────
        elif self.state == 3:
            # Trail using 5-min 20 EMA on CDSL spot
            df_5m = t.history(period="1d", interval="5m")
            if not df_5m.empty and len(df_5m) >= 20:
                ema_20 = float(df_5m['Close'].ewm(span=20, adjust=False).mean().iloc[-1])
                ema_exit = (current_spot > ema_20) if not self.is_call else (current_spot < ema_20)
                if ema_exit:
                    self.state = 4
                    play_sound_alert()
                    msg = (
                        f"🏁 *LOT 2 RUNNER EXITED (5-Min 20 EMA Cross)*\n\n"
                        f"📍 *Spot crossed 20 EMA:* Spot ₹{current_spot:.2f} vs 20 EMA ₹{ema_20:.2f}\n"
                        f"💵 *Exit Premium LTP:* ~₹{current_prem:.2f}\n"
                        f"⚡ *Action:* SELL REMAINING RUNNER AT MARKET"
                    )
                    send_telegram_alert(msg)
                    return


if __name__ == "__main__":
    watchdog = CDSL1MinWatchdog("CDSL.NS")
    watchdog.run()
