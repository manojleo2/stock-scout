"""
CDSL Intraday V3.3 — Institutional Real-Time Execution Watchdog (09:15 – 03:30)
─────────────────────────────────────────────────────────────────────────────
Full Dual-Engine & Risk Management Architecture:
1. ENGINE 1 (09:15 - 09:25 AM): Morning FSM Rejection (Requires Model Conviction >= 60%).
2. ENGINE 2 (09:30 - 02:30 PM): Live Intraday Breakout & 4-Factor Spike Classifier
   - Streams CDSL alongside Nifty 50 (^NSEI) for real-time Alpha vs Beta.
   - 4 Archetypes:
     • 🏛️ Institutional Cash Accumulation (High Alpha, Vol > 2.5x, BPI > 1.5) -> Target +Rs 8.50 | SL -Rs 3.00
     • ⚡ Strike Gamma Squeeze (Rs 20 strike breach on high vol) -> Target +Rs 6.50 | SL -Rs 2.50
     • 🌊 Passive Nifty Beta Sympathy (Index-driven) -> Target +Rs 3.50 | SL -Rs 2.00
     • 🪤 Liquidity Trap / Manipulation (Wick >= 40% or Vol < 1.4x) -> ❌ BLOCKED (0 trades)
3. Active Trade & Stagnation Engine:
   - Dynamic Trailing: SL moves to Cost (Zero Risk) at +Rs 2.00 to +Rs 4.00 gain.
   - Theta Stagnation Sentinel: If held >= 60 mins past 01:00 PM with < Rs 1.00 progress -> Exit to Cash.
   - Hard Cutoff: 03:05 PM exit all active positions at market.
4. Capital & Overtrading Protection:
   - Daily Trade Cap: Max 2 trades per day.
   - Post-Trade 15-Minute Cooldown before scanning for Trade 2.
   - 2R Daily Stop Limit: 2 SL hits triggers immediate, permanent daily lockout.
5. Filtered Reversal Sentinel (Emergency cash exit + 1-candle confirmation).
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
from utils.options_feed import get_live_option_quote, get_atm_order_flow_push
from utils.market_calendar import is_trading_holiday
from utils.entry_conditions import (
    calculate_vwap_and_bands,
    is_rejection_candle,
    check_pre_trade_gates,
    _get_today_prediction
)

# Optional PC sound alert
try:
    import winsound
    def play_sound_alert(count: int = 2):
        for _ in range(count):
            for freq in [1400, 1800]:
                winsound.Beep(freq, 120)
except Exception:
    def play_sound_alert(count: int = 2):
        pass



def acquire_single_instance_lock():
    """Ensures only ONE instance of the Watchdog can run at any time."""
    lock_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "watchdog.lock")
    current_pid = os.getpid()
    if os.path.exists(lock_file):
        try:
            with open(lock_file, "r") as f:
                old_pid = int(f.read().strip())
            if old_pid != current_pid:
                import subprocess
                check = subprocess.run(f'tasklist /FI "PID eq {old_pid}"', capture_output=True, text=True, shell=True)
                if str(old_pid) in check.stdout:
                    print(f"⚠️ Terminating ghost watchdog instance (PID: {old_pid})...")
                    subprocess.run(f"taskkill /F /PID {old_pid}", shell=True, capture_output=True)
                    import time
                    time.sleep(1)
        except Exception:
            pass
    try:
        with open(lock_file, "w") as f:
            f.write(str(current_pid))
    except Exception:
        pass

class CDSLV3LiveWatchdog:
    def __init__(self, symbol: str = "CDSL.NS"):
        self.symbol = symbol
        self.nifty_symbol = "^NSEI"
        
        # State Machine:
        # 0: Scanning / Cash
        # 1: Tranche 1 Active
        # 2: Full Position Active
        # 3: Runner Active
        # 4: Exited / Closed
        # 5: Invalidation — Awaiting Reversal Confirmation Filter
        # 6: Reversal Position Active (1 Lot strictly)
        self.state = 0
        
        # Direction & Contract
        self.direction = None
        self.is_call = False
        self.conviction = 0.0
        self.morning_sidelined = False
        self.atm_strike = 1260
        self.option_contract = ""
        
        # Positions & Sizing
        self.lot_size = 475  # 1 Lot default; up to 950 on 2 Lots
        self.active_lots = 0
        self.t1_prem = None
        self.t2_prem = None
        self.avg_entry = None
        self.active_sl = None
        self.active_target = None
        self.active_regime = "NORMAL"
        self.sl_trailed_state = "INITIAL"
        
        # Structural S/R Map
        self.pdh = None
        self.pdl = None
        self.pdc = None
        self.orb_high = None
        self.orb_low = None
        self.orb_calculated = False
        
        # Daily Limits & Risk Control
        self.daily_trade_count = 0
        self.max_daily_trades = 2
        self.daily_sl_count = 0
        self.max_daily_sl = 1   # changed from 2: stop after 1 SL hit per day, preserve capital
        self.weekly_sl_count = 0
        self.max_weekly_sl = 2  # max 2 SL hits per week before sidelining
        self.is_daily_locked_out = False
        self.cooldown_until = None
        self.entry_time = None
        self.stagnation_alert_sent = False

        # Alerts & State flags
        self.t1_alert_sent = False
        self.t2_alert_sent = False
        self.invalidation_alert_sent = False
        self.reversal_alert_sent = False
        self.lot1_exit_sent = False
        self.trailed_cost_sent = False
        self.trailed_locked_sent = False
        self.closed_alert_sent = False
        self.breakout_alert_sent = False
        self.last_fakeout_alert_time = 0.0

        # Reversal Sentinel internal memory
        self.reversal_armed = False
        self.reversal_is_call = False
        self.reversal_direction = None
        self.reversal_trigger_level = None
        self.reversal_filter_candle_count = 0
        self.reversal_fill_prem = 0.0
        self.reversal_sl = 0.0
        self.reversal_target = 0.0

    def reset_for_next_trade(self, now: dt.datetime):
        """Reset execution state for Trade 2 with a 15-minute cooldown."""
        self.state = 0
        self.active_lots = 0
        self.t1_prem = None
        self.t2_prem = None
        self.avg_entry = None
        self.active_sl = None
        self.active_target = None
        self.t1_alert_sent = False
        self.t2_alert_sent = False
        self.invalidation_alert_sent = False
        self.reversal_alert_sent = False
        self.lot1_exit_sent = False
        self.trailed_cost_sent = False
        self.trailed_locked_sent = False
        self.breakout_alert_sent = False
        self.entry_time = None
        self.stagnation_alert_sent = False
        self.cooldown_until = now + dt.timedelta(minutes=15)

    def load_structural_levels(self):
        """Fetch PDH, PDL, PDC from daily history."""
        try:
            ticker = yf.Ticker(self.symbol)
            hist = ticker.history(period="10d", interval="1d")
            if len(hist) >= 2:
                prev_day = hist.iloc[-2]
                self.pdh = round(float(prev_day["High"]), 2)
                self.pdl = round(float(prev_day["Low"]), 2)
                self.pdc = round(float(prev_day["Close"]), 2)
                logging.info(f"Loaded S/R: PDH=Rs {self.pdh} | PDL=Rs {self.pdl} | PDC=Rs {self.pdc}")
        except Exception as e:
            logging.error(f"Error loading structural levels: {e}")

    def load_morning_model(self) -> bool:
        """Fetch pre-market calibrated prediction."""
        try:
            pred = _get_today_prediction(self.symbol)
            if not pred:
                return False
            raw_dir = pred.get("predicted_direction") or pred.get("direction") or pred.get("pred_result", {}).get("direction", "DOWN")
            self.direction = "UP" if "UP" in str(raw_dir).upper() else "DOWN"
            self.conviction = float(pred.get("calibrated_conviction_pct") or pred.get("conviction") or pred.get("pred_result", {}).get("conviction_pct") or 50.0)
            self.is_call = (self.direction == "UP")

            weekday = dt.date.today().weekday()
            min_thresh = 65.0 if weekday == 4 else 60.0
            if self.conviction < min_thresh:
                logging.info(f"Model conviction {self.conviction:.1f}% below {min_thresh}% threshold.")
                return False
            return True
        except Exception as e:
            logging.error(f"Error loading morning model: {e}")
            return False

    def get_nifty_5m_metrics(self) -> float:
        """Fetch 5-minute percentage return for Nifty 50 (^NSEI)."""
        try:
            nifty_df = yf.Ticker(self.nifty_symbol).history(period="1d", interval="5m")
            if len(nifty_df) >= 2:
                c_now = float(nifty_df["Close"].iloc[-1])
                c_prev = float(nifty_df["Close"].iloc[-2])
                return round(((c_now - c_prev) / c_prev) * 100.0, 2)
        except Exception:
            pass
        return 0.0

    def classify_spike_reason(
        self,
        df_1m: pd.DataFrame,
        current_spot: float,
        is_breakout_up: bool,
        vol_surge: float,
        nifty_ret_5m: float,
        bpi: float,
        order_flow: dict
    ) -> tuple:
        """
        CDSL-Specific 4-Factor Spike Reason Classifier.
        Returns: (category_name, details_dict, is_fakeout_trap)
        """
        last_bar = df_1m.iloc[-1]
        candle_open = float(last_bar["Open"])
        candle_close = float(last_bar["Close"])
        candle_high = float(last_bar["High"])
        candle_low = float(last_bar["Low"])
        candle_range = max(candle_high - candle_low, 0.01)

        # 1. Fakeout & Manipulation Check
        is_fakeout = False
        fakeout_reason = ""
        if is_breakout_up:
            upper_wick = candle_high - max(candle_open, candle_close)
            wick_ratio = upper_wick / candle_range
            if wick_ratio >= 0.40:
                is_fakeout = True
                fakeout_reason = f"Exhaustion upper wick ({wick_ratio*100:.0f}% of range)"
            elif vol_surge < 1.4:
                is_fakeout = True
                fakeout_reason = f"Thin breakout volume ({vol_surge:.1f}x vs 1.4x required)"
        else:
            lower_wick = min(candle_open, candle_close) - candle_low
            wick_ratio = lower_wick / candle_range
            if wick_ratio >= 0.40:
                is_fakeout = True
                fakeout_reason = f"Exhaustion lower absorption wick ({wick_ratio*100:.0f}% of range)"
            elif vol_surge < 1.4:
                is_fakeout = True
                fakeout_reason = f"Thin breakdown volume ({vol_surge:.1f}x vs 1.4x required)"

        if is_fakeout:
            details = {
                "profile": "🪤 Thin-Volume Liquidity Trap / Manipulation",
                "reason": fakeout_reason,
                "vol_surge": vol_surge,
                "bpi": bpi
            }
            return "LIQUIDITY_TRAP", details, True

        # 2. Relative Strength (CDSL Alpha vs Nifty Beta)
        cdsl_5m_ret = 0.0
        if len(df_1m) >= 5:
            c5 = float(df_1m["Close"].iloc[-5])
            cdsl_5m_ret = round(((current_spot - c5) / c5) * 100.0, 2)
        alpha_5m = round(cdsl_5m_ret - nifty_ret_5m, 2)

        # 3. Strike Gamma Squeeze Check (₹20 strike magnet)
        dist_to_strike = abs(current_spot - (round(current_spot / 20.0) * 20.0))
        is_strike_crossing = (dist_to_strike <= 2.5) and (vol_surge >= 2.0)

        # 4. Market Depth & Order Flow
        of_signal = order_flow.get("signal", "BALANCED")

        # Classification Hierarchy:
        if alpha_5m >= 0.7 and vol_surge >= 2.5 and bpi >= 1.5:
            cat = "INSTITUTIONAL_ACCUMULATION"
            desc = "🏛️ Institutional Cash Accumulation (High Idiosyncratic Alpha)"
        elif is_strike_crossing:
            cat = "GAMMA_SQUEEZE"
            desc = "⚡ Strike Gamma / Short Covering Squeeze (₹20 Magnet Breach)"
        elif abs(alpha_5m) <= 0.35 and abs(nifty_ret_5m) >= 0.4:
            cat = "NIFTY_SYMPATHY"
            desc = "🌊 Nifty Beta Sympathy (Driven by Index Momentum)"
        else:
            cat = "TECHNICAL_BREAKOUT"
            desc = "📈 Technical Price & Volume Momentum"

        details = {
            "profile": desc,
            "alpha_5m": alpha_5m,
            "cdsl_5m_ret": cdsl_5m_ret,
            "nifty_ret_5m": nifty_ret_5m,
            "vol_surge": vol_surge,
            "bpi": bpi,
            "of_signal": of_signal
        }
        return cat, details, False

    def compute_dynamic_target_and_sl(
        self,
        df_1m: pd.DataFrame,
        entry_prem: float,
        current_spot: float,
        is_reversal: bool = False,
        is_breakout: bool = False,
        breakout_category: str = ""
    ) -> tuple:
        """
        Dynamically calculate Target and Stop Loss in Options Premium (Rs) tailored to the spike archetype.
        """
        recent_bars = df_1m.tail(15)
        highs = recent_bars["High"].values
        lows = recent_bars["Low"].values
        closes = recent_bars["Close"].values
        tr_list = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            tr_list.append(tr)
        atr_5m = float(np.mean(tr_list)) if tr_list else 2.5

        if is_breakout:
            if breakout_category == "INSTITUTIONAL_ACCUMULATION":
                regime = "INSTITUTIONAL_EXPANSION"
                target_prem = round(entry_prem + 8.50, 2)
                sl_prem = round(max(entry_prem - 3.00, 0.5), 2)
            elif breakout_category == "GAMMA_SQUEEZE":
                regime = "GAMMA_MOMENTUM"
                target_prem = round(entry_prem + 6.50, 2)
                sl_prem = round(max(entry_prem - 2.50, 0.5), 2)
            elif breakout_category == "NIFTY_SYMPATHY":
                regime = "NIFTY_SCALP"
                target_prem = round(entry_prem + 3.50, 2)
                sl_prem = round(max(entry_prem - 2.00, 0.5), 2)
            else:
                regime = "BREAKOUT"
                target_prem = round(entry_prem + 5.50, 2)
                sl_prem = round(max(entry_prem - 3.00, 0.5), 2)
        elif is_reversal:
            regime = "REVERSAL"
            target_prem = round(entry_prem + 4.5, 2)
            sl_prem = round(max(entry_prem - 2.5, 0.5), 2)
        elif atr_5m > 4.5:
            regime = "EXPANSION"
            target_prem = round(entry_prem + 10.0, 2)
            sl_prem = round(max(entry_prem - 5.0, 0.5), 2)
        elif atr_5m < 2.5:
            regime = "SCALP"
            target_prem = round(entry_prem + 2.5, 2)
            sl_prem = round(max(entry_prem - 2.0, 0.5), 2)
        else:
            regime = "NORMAL"
            target_prem = round(entry_prem + 5.5, 2)
            sl_prem = round(max(entry_prem - 3.5, 0.5), 2)

        return regime, target_prem, sl_prem

    def eval_intraday_breakout(
        self,
        df_1m: pd.DataFrame,
        current_spot: float,
        vwap: float,
        vol_surge: float,
        bpi: float,
        now: dt.datetime
    ) -> bool:
        """
        ENGINE 2: Live Intraday Breakout Sentinel (09:30 AM – 02:30 PM).
        Fires irrespective of morning model conviction if 4 structural conditions are met.
        """
        if self.breakout_alert_sent or self.state != 0 or self.is_daily_locked_out:
            return False

        if self.daily_trade_count >= self.max_daily_trades:
            return False

        if self.pdh is None or self.pdl is None:
            return False

        is_breakout_up = False
        is_breakdown_down = False
        level_breached = ""

        # Bullish Breakout (Call Trigger - Must be above VWAP and actually breaking PDH/ORB)
        if current_spot >= self.pdh and current_spot > (vwap * 1.002):
            # Check if this was a fresh cross above PDH or breakout above ORB high
            is_breakout_up = True
            level_breached = f"Prior Day High (Rs {self.pdh:.2f})"
        elif self.orb_high is not None and current_spot >= self.orb_high and current_spot > (vwap * 1.002):
            is_breakout_up = True
            level_breached = f"15-Min ORB High (Rs {self.orb_high:.2f})"

        # Bearish Breakdown (Put Trigger - Must be below VWAP)
        if current_spot <= self.pdl and current_spot < (vwap * 0.998):
            is_breakdown_down = True
            level_breached = f"Prior Day Low (Rs {self.pdl:.2f})"
        elif self.orb_low is not None and current_spot <= self.orb_low and current_spot < (vwap * 0.998):
            is_breakdown_down = True
            level_breached = f"15-Min ORB Low (Rs {self.orb_low:.2f})"

        if not is_breakout_up and not is_breakdown_down:
            return False

        # Gather real-time metrics
        nifty_ret_5m = self.get_nifty_5m_metrics()
        atm_strike = round(current_spot / 20.0) * 20
        order_flow = get_atm_order_flow_push(self.symbol, atm_strike, current_spot)

        # 4-Factor Spike Reason Classifier
        category, details, is_trap = self.classify_spike_reason(
            df_1m, current_spot, is_breakout_up, vol_surge, nifty_ret_5m, bpi, order_flow
        )

        if is_trap:
            curr_ts = time.time()
            if (curr_ts - self.last_fakeout_alert_time) > 300:  # Log once per 5 minutes to keep terminal clean
                self.last_fakeout_alert_time = curr_ts
                print(
                    f"[{now.strftime('%H:%M:%S')}] 🛡️ FAKEOUT FILTERED: {details['profile']} near {level_breached} (Thin vol: {vol_surge:.1f}x). Sidelined in cash.",
                    flush=True
                )
            return False

        # Genuine Breakout Confirmed
        target_is_call = is_breakout_up
        target_opt_type = "CE" if target_is_call else "PE"
        target_contract = f"CDSL Rs {atm_strike} {target_opt_type}"

        opt_quote = get_live_option_quote(self.symbol, atm_strike, is_call=target_is_call, spot=current_spot)
        entry_prem = float(opt_quote.get("ltp") or 0.0)
        source = str(opt_quote.get("source", "LIVE")).replace('_', ' ')

        if entry_prem <= 0.5:
            return False

        gates_passed, gate_reason = check_pre_trade_gates(opt_quote, open_prem=entry_prem)
        if not gates_passed:
            logging.info(f"Breakout gates failed: {gate_reason}")
            return False

        regime, dyn_target, dyn_sl = self.compute_dynamic_target_and_sl(
            df_1m, entry_prem, current_spot, is_breakout=True, breakout_category=category
        )

        self.state = 2
        self.is_call = target_is_call
        self.direction = "UP" if target_is_call else "DOWN"
        self.atm_strike = atm_strike
        self.option_contract = target_contract
        self.t1_prem = entry_prem
        self.t2_prem = entry_prem
        self.avg_entry = entry_prem
        self.active_lots = 1
        self.active_regime = regime
        self.active_target = dyn_target
        self.active_sl = dyn_sl
        self.breakout_alert_sent = True
        self.t2_alert_sent = True
        self.entry_time = now

        trade_num = self.daily_trade_count + 1
        play_sound_alert(3)
        send_telegram_alert(
            f"🚀 *LIVE INTRADAY BREAKOUT TRIGGERED! (Trade {trade_num}/{self.max_daily_trades})*\n\n"
            f"🎟️ *Contract:* {target_contract}\n"
            f"💵 *Fill Premium LTP:* ~Rs {entry_prem:.2f} ({source})\n"
            f"🎯 *Breakout Target:* Rs {dyn_target:.2f} (+Rs {(dyn_target - entry_prem):.2f})\n"
            f"🛑 *Dynamic SL:* Rs {dyn_sl:.2f} (-Rs {(entry_prem - dyn_sl):.2f})\n\n"
            f"🧬 *SPIKE DIAGNOSIS:* {details['profile']}\n"
            f"   • *Structure:* Crossed {level_breached}\n"
            f"   • *Relative Strength:* CDSL {details.get('cdsl_5m_ret', 0.0):+0.2f}% vs Nifty {details.get('nifty_ret_5m', 0.0):+0.2f}% (Alpha: {details.get('alpha_5m', 0.0):+0.2f}%)\n"
            f"   • *1-Min Volume:* {vol_surge:.1f}x 20-SMA | BPI: {bpi:.2f}\n"
            f"   • *Order Flow:* {details.get('of_signal', 'BALANCED')}\n\n"
            f"⚡ *Action:* BUY 1 LOT (475 Qty) AT MARKET"
        )
        return True

    def run(self):
        acquire_single_instance_lock()
        logging.info("Starting CDSL V3.3 Institutional Real-Time Execution Watchdog...")
        today = dt.date.today()
        is_holiday, holiday_name = is_trading_holiday(today)
        if is_holiday or today.weekday() >= 5:
            logging.info(f"Market closed today ({holiday_name or 'Weekend'}). Exiting.")
            return

        self.load_structural_levels()
        has_model = self.load_morning_model()

        # ── AUTO-ADOPT ACTIVE BROKER POSITION (If trade exists in Groww) ──
        try:
            from utils.options_feed import get_groww_positions
            positions = get_groww_positions()
            for p in positions:
                sym = p.get("trading_symbol", "")
                qty = int(p.get("quantity", 0))
                if "CDSL" in sym and qty > 0:
                    is_pe = "PE" in sym
                    # Extract strike
                    import re
                    m = re.search(r'(\d{4})(CE|PE)', sym)
                    strike_val = int(m.group(1)) if m else 1340
                    entry_p = float(p.get("credit_price") or p.get("net_price") or 38.25)
                    self.state = 2
                    self.is_call = not is_pe
                    self.direction = "DOWN" if is_pe else "UP"
                    self.atm_strike = strike_val
                    self.active_strike = strike_val
                    self.option_contract = f"CDSL Rs {strike_val} {'PE' if is_pe else 'CE'}"
                    self.avg_entry = entry_p
                    self.t1_prem = entry_p
                    self.t2_prem = entry_p
                    self.active_lots = 1
                    self.active_sl = round(max(entry_p - 3.20, 0.5), 2)
                    self.active_target = round(entry_p + 5.30, 2)
                    self.entry_time = dt.datetime.now()
                    self.last_known_live_prem = entry_p
                    self.entry_spot = 1330.80
                    print(
                        f"🔄 ADOPTED ACTIVE GROWW POSITION: {self.option_contract} (Qty {qty} @ Rs {entry_p:.2f}) | SL: Rs {self.active_sl:.2f} | Target: Rs {self.active_target:.2f}",
                        flush=True
                    )
                    send_telegram_alert(
                        f"🔄 *ADOPTED LIVE BROKER POSITION: {self.option_contract}*\n\n"
                        f"💵 *Fill Entry:* Rs {entry_p:.2f}\n"
                        f"🛑 *Armed SL:* Rs {self.active_sl:.2f} (-Rs {(entry_p - self.active_sl):.2f})\n"
                        f"🎯 *Target:* Rs {self.active_target:.2f} (+Rs {(self.active_target - entry_p):.2f})\n"
                        f"⚡ *Action:* ACTIVE REAL-TIME MONITORING RESUMED"
                    )
                    break
        except Exception as e:
            logging.warning(f"Could not auto-adopt broker position: {e}")

        pdh_str = f"Rs {self.pdh:.2f}" if self.pdh else "N/A"
        pdl_str = f"Rs {self.pdl:.2f}" if self.pdl else "N/A"

        if not has_model:
            self.morning_sidelined = True
            send_telegram_alert(
                f"⚪ *CDSL Watchdog V3.3 — Pre-Market Sidelined*\n\n"
                f"Morning conviction ({self.conviction:.1f}%) is below 60%. Morning FSM (09:15–09:25) disabled.\n"
                f"🛡️ *S/R Structure:* PDH: {pdh_str} | PDL: {pdl_str}\n"
                f"⚡ *Intraday Breakout Sentinel Active:* Scanning live candles (09:30–02:30) for institutional spikes, alpha vs Nifty & gamma squeeze."
            )
        else:
            self.morning_sidelined = False
            send_telegram_alert(
                f"🚀 *CDSL V3.3 Institutional Sentinel Active!*\n\n"
                f"📅 *Session:* {dt.datetime.now().strftime('%a, %d %b %Y')}\n"
                f"🎯 *Model Forecast:* {self.direction} (Conviction: {self.conviction:.1f}%)\n"
                f"🛡️ *S/R Structure:* PDH: {pdh_str} | PDL: {pdl_str}\n"
                f"⚡ *Live Engine:* Morning FSM + Intraday Spike Classifier + Dynamic Scalp/Trend Targets Active."
            )

        while True:
            now = dt.datetime.now()
            if now.hour > 15 or (now.hour == 15 and now.minute >= 6):
                break

            try:
                self.tick(now)
            except Exception as e:
                logging.error(f"Watchdog tick error: {e}")

            time.sleep(12)

    def tick(self, now: dt.datetime):
        if self.is_daily_locked_out:
            return

        # Cooldown check between trades
        if self.cooldown_until and now < self.cooldown_until and self.state == 0:
            return

        t = yf.Ticker(self.symbol)
        df_1m = t.history(period="1d", interval="1m")
        if df_1m.empty or len(df_1m) < 2:
            return

        if "Datetime" not in df_1m.columns:
            df_1m = df_1m.reset_index()

        current_spot = round(float(df_1m["Close"].iloc[-1]), 2)
        open_price = round(float(df_1m["Open"].iloc[0]), 2)
        
        # LOCK STRIKE: Strike and contract MUST NOT change during active trades!
        if self.state == 0:
            self.atm_strike = round(current_spot / 20.0) * 20
            self.active_strike = self.atm_strike
            opt_type = "CE" if self.is_call else "PE"
            self.option_contract = f"CDSL Rs {self.atm_strike} {opt_type}"

        # Gap-Day Detection (> Rs 12 Gap)
        if not hasattr(self, "is_gap_day") and len(df_1m) > 0 and getattr(self, "pdc", None):
            open_p = float(df_1m.iloc[0]["Open"])
            gap_pts = abs(open_p - self.pdc)
            self.is_gap_day = (gap_pts > 12.0)
            if self.is_gap_day:
                logging.info(f"GAP DAY DETECTED: Gap of Rs {gap_pts:.2f} > Rs 12.0. Morning window shifted to 09:35 - 09:45 AM.")

        # 15-Minute Opening Range Breakout (ORB) calculation
        if not self.orb_calculated and len(df_1m) >= 15 and (now.hour > 9 or (now.hour == 9 and now.minute >= 30)):
            first_15 = df_1m.iloc[:15]
            self.orb_high = round(float(first_15["High"].max()), 2)
            self.orb_low = round(float(first_15["Low"].min()), 2)
            self.orb_calculated = True
            logging.info(f"15-Min ORB Set: High=Rs {self.orb_high:.2f} | Low=Rs {self.orb_low:.2f}")

        # Order Flow & Buyer-Seller Push (BPI)
        last_bar = df_1m.iloc[-1]
        candle_range = (last_bar["High"] - last_bar["Low"])
        vol_avg = df_1m["Volume"].rolling(15).mean().iloc[-1]
        vol_surge = (last_bar["Volume"] / vol_avg) if vol_avg > 0 else 1.0
        bpi = (((last_bar["Close"] - last_bar["Low"]) / candle_range) * vol_surge) if candle_range > 0 else 1.0

        # VWAP & Bands
        vwap, upper_band, lower_band, std_dev = calculate_vwap_and_bands(df_1m, sigma_mult=1.5)
        
        # Live Option Quote (Use locked active_strike if position is open)
        tracked_strike = getattr(self, "active_strike", self.atm_strike) if self.state in (1, 2, 3, 6) else self.atm_strike
        quote = get_live_option_quote(self.symbol, tracked_strike, is_call=self.is_call, spot=current_spot)
        quote_status = quote.get("status", "LIVE")
        
        # Robust Option Premium Resolution
        if quote_status == "LIVE" and float(quote.get("ltp") or 0.0) > 0:
            current_prem = float(quote.get("ltp"))
            self.last_known_live_prem = current_prem
            source = str(quote.get("source", "LIVE")).replace('_', ' ')
        elif self.state in (1, 2, 3, 6):
            # Active Position: Anchor to real traded price + real-time spot delta tracking
            base_prem = getattr(self, "last_known_live_prem", None) or self.avg_entry
            ref_spot = getattr(self, "entry_spot", current_spot)
            delta = -0.55 if not self.is_call else 0.55
            spot_diff = current_spot - ref_spot
            current_prem = round(max(base_prem + (delta * spot_diff), 0.50), 2)
            source = "REAL DELTA TRACKING"
        else:
            current_prem = float(quote.get("ltp") or 0.0)
            source = str(quote.get("source", "ESTIMATED")).replace('_', ' ')

        # ── LIVE POWERSHELL TELEMETRY HEARTBEAT (Every 12s) ──────────────────
        state_names = {
            0: "SCANNING",
            1: "T1 ACTIVE",
            2: "FULL POSITION",
            3: "RUNNER ONLY",
            4: "EXITED",
            5: "REVERSAL PENDING",
            6: "REVERSAL ACTIVE"
        }
        state_str = state_names.get(self.state, f"STATE {self.state}")
        time_str = now.strftime("%H:%M:%S")
        orb_str = f"[{self.orb_low:.1f}-{self.orb_high:.1f}]" if self.orb_calculated else "Forming"
        
        if self.state in (1, 2, 3, 6):
            ref_entry = self.avg_entry or self.t1_prem or self.reversal_fill_prem or current_prem
            diff = current_prem - ref_entry
            pnl = diff * (self.active_lots * self.lot_size)
            tgt_str = f"Rs {self.active_target:.2f}" if self.active_target else "N/A"
            sl_str = f"Rs {self.active_sl:.2f}" if self.active_sl else "N/A"
            print(
                f"[{time_str}] 🎟️ {self.option_contract} LTP: Rs {current_prem:.2f} | "
                f"Entry: Rs {ref_entry:.2f} | P&L: Rs {pnl:+,.0f} ({diff:+0.2f}) | "
                f"Tgt: {tgt_str} | SL: {sl_str} | "
                f"Spot: Rs {current_spot:.2f} (VWAP Rs {vwap:.2f}) | {state_str}",
                flush=True
            )
        else:
            other_is_call = not self.is_call
            other_quote = get_live_option_quote(self.symbol, self.atm_strike, is_call=other_is_call, spot=current_spot)
            other_prem = float(other_quote.get("ltp") or 0.0)
            pe_prem = current_prem if not self.is_call else other_prem
            ce_prem = other_prem if not self.is_call else current_prem
            vwap_dist = current_spot - vwap
            print(
                f"[{time_str}] 🏷️ {state_str} | "
                f"CDSL {self.atm_strike} PE: Rs {pe_prem:.2f} | CE: Rs {ce_prem:.2f} | "
                f"Spot: Rs {current_spot:.2f} (VWAP: Rs {vwap:.2f}, Diff: {vwap_dist:+0.2f}) | "
                f"ORB: {orb_str} | BPI: {bpi:.2f}",
                flush=True
            )

        # ── HARD MARKET CUTOFF (02:45 PM Friday / 03:05 PM Mon-Thu) ──────────
        is_friday = (now.weekday() == 4)
        cutoff_reached = (now.hour == 14 and now.minute >= 45) if is_friday else ((now.hour == 15 and now.minute >= 5) or now.hour > 15)
        cutoff_label = "02:45 PM (Friday Theta Protection)" if is_friday else "03:05 PM"
        if cutoff_reached:
            if self.state in (1, 2, 3, 6) and not self.closed_alert_sent:
                self.state = 4
                self.closed_alert_sent = True
                play_sound_alert(3)
                send_telegram_alert(
                    f"⏰ *{cutoff_label} HARD CUTOFF REACHED — EXIT ALL POSITIONS*\n\n"
                    f"🎟️ *Contract:* {self.option_contract}\n"
                    f"💵 *Exit Premium LTP:* ~Rs {current_prem:.2f} ({source})\n"
                    f"📍 *Spot:* Rs {current_spot:,.2f}\n"
                    f"⚡ *Action:* CLOSE ALL ACTIVE LOTS AT MARKET"
                )
            return

        # ── THETA STAGNATION SENTINEL (States 2, 6) ──────────────────────────
        if self.state in (2, 6) and self.entry_time is not None:
            elapsed_mins = (now - self.entry_time).total_seconds() / 60.0
            if elapsed_mins >= 60.0 and now.hour >= 13 and not self.stagnation_alert_sent:
                ref_entry = self.avg_entry or self.reversal_fill_prem or current_prem
                gain = current_prem - ref_entry
                if gain < 1.00:
                    self.stagnation_alert_sent = True
                    self.daily_trade_count += 1
                    play_sound_alert(2)
                    pnl = gain * (self.active_lots * 475)
                    send_telegram_alert(
                        f"⏳ *THETA STAGNATION EXIT — CASH CAPITAL PRESERVED*\n\n"
                        f"🎟️ *Contract:* {self.option_contract}\n"
                        f"💵 *Exit Premium LTP:* ~Rs {current_prem:.2f} ({source})\n"
                        f"⏱️ *Duration:* Stagnant for {int(elapsed_mins)} mins with no directional expansion (P&L: Rs {pnl:+,.0f})\n"
                        f"🛡️ *Action:* POSITION CLOSED TO CASH before rapid afternoon theta acceleration."
                    )
                    if self.daily_trade_count >= self.max_daily_trades or now.hour >= 14:
                        self.state = 4
                        send_telegram_alert(f"🏁 *DAILY TRADE LIMIT REACHED ({self.daily_trade_count}/{self.max_daily_trades})*. Day concluded.")
                    else:
                        self.reset_for_next_trade(now)
                        send_telegram_alert(f"⏳ *Entering 15-Min Cooldown*. Scanning for Trade 2 after {self.cooldown_until.strftime('%H:%M:%S')}.")
                    return

        # ── REVERSAL SENTINEL: CHECK TECHNICAL INVALIDATION (States 1, 2) ──
        if self.state in (1, 2) and (now.hour < 14 or (now.hour == 14 and now.minute <= 45)):
            invalidation = False
            reversal_reason = ""

            if not self.is_call:  # Holding PUT (PE)
                if current_spot >= upper_band and last_bar["Close"] > last_bar["Open"]:
                    invalidation = True
                    reversal_reason = f"CDSL pierced Upper VWAP Band (Rs {upper_band:.2f}) with buying push"
                elif self.pdh is not None and current_spot >= self.pdh and vol_surge >= 1.4:
                    invalidation = True
                    reversal_reason = f"CDSL broke Prior Day High (Rs {self.pdh:.2f}) with {vol_surge:.1f}x volume"
            else:  # Holding CALL (CE)
                if current_spot <= lower_band and last_bar["Close"] < last_bar["Open"]:
                    invalidation = True
                    reversal_reason = f"CDSL pierced Lower VWAP Band (Rs {lower_band:.2f}) with selling dump"
                elif self.pdl is not None and current_spot <= self.pdl and vol_surge >= 1.4:
                    invalidation = True
                    reversal_reason = f"CDSL broke Prior Day Low (Rs {self.pdl:.2f}) with {vol_surge:.1f}x volume"

            if invalidation and not self.invalidation_alert_sent:
                self.state = 5
                self.invalidation_alert_sent = True
                play_sound_alert(2)
                
                self.reversal_armed = True
                self.reversal_is_call = not self.is_call
                self.reversal_direction = "UP" if self.reversal_is_call else "DOWN"
                self.reversal_trigger_level = current_spot
                self.reversal_filter_candle_count = 0

                rev_opt = "CE" if self.reversal_is_call else "PE"
                send_telegram_alert(
                    f"⚠️ *TRADE INVALIDATED — EMERGENCY EXIT TO CASH!*\n\n"
                    f"🚨 *Reason:* {reversal_reason}\n"
                    f"🎟️ *Exited Contract:* {self.option_contract} @ ~Rs {current_prem:.2f}\n"
                    f"🛡️ *Action:* POSITION CLOSED TO 100% CASH\n"
                    f"⏳ *Reversal Sentinel Armed:* Watching for 1-candle confirmation to BUY {rev_opt}."
                )
                return

        # ── REVERSAL STEP 2: 1-CANDLE INSTITUTIONAL FILTER (State 5) ────────
        if self.state == 5:
            self.reversal_filter_candle_count += 1
            if self.reversal_filter_candle_count >= 2:
                candle_confirmed = False
                if self.reversal_is_call:
                    candle_confirmed = (last_bar["Close"] > last_bar["Open"]) and (last_bar["Close"] >= self.reversal_trigger_level)
                else:
                    candle_confirmed = (last_bar["Close"] < last_bar["Open"]) and (last_bar["Close"] <= self.reversal_trigger_level)

                if candle_confirmed and not self.reversal_alert_sent:
                    rev_strike = round(current_spot / 20.0) * 20
                    rev_type = "CE" if self.reversal_is_call else "PE"
                    rev_contract = f"CDSL Rs {rev_strike} {rev_type}"
                    rev_quote = get_live_option_quote(self.symbol, rev_strike, is_call=self.reversal_is_call, spot=current_spot)
                    rev_prem = float(rev_quote.get("ltp") or 0.0)

                    regime, rev_tgt, rev_sl = self.compute_dynamic_target_and_sl(df_1m, rev_prem, current_spot, is_reversal=True)
                    
                    self.state = 6
                    self.option_contract = rev_contract
                    self.is_call = self.reversal_is_call
                    self.reversal_fill_prem = rev_prem
                    self.reversal_target = rev_tgt
                    self.reversal_sl = rev_sl
                    self.reversal_alert_sent = True
                    self.entry_time = now
                    play_sound_alert(3)

                    send_telegram_alert(
                        f"🔄 *CONFIRMED REVERSAL ENTRY TRIGGERED! (1 Lot / 475 Qty)*\n\n"
                        f"🎟️ *Contract:* {rev_contract}\n"
                        f"💵 *Fill Premium:* ~Rs {rev_prem:.2f} ({rev_quote.get('source', 'LIVE')})\n"
                        f"🛑 *Dynamic Reversal SL:* Rs {self.reversal_sl:.2f} (-Rs {(rev_prem - self.reversal_sl):.2f})\n"
                        f"🎯 *Scalp Target:* Rs {self.reversal_target:.2f} (+Rs {(self.reversal_target - rev_prem):.2f})\n"
                        f"📍 *Spot Confirmation:* Rs {current_spot:,.2f} (BPI: {bpi:.2f})\n"
                        f"⚡ *Action:* BUY 1 LOT (475 Qty) AT MARKET"
                    )
                    return
                else:
                    self.state = 0
                    logging.info("Reversal confirmation filter failed. Sidelined to cash.")
                    send_telegram_alert(
                        f"🛡️ *REVERSAL TRAP AVOIDED — STAYING IN CASH*\n\n"
                        f"Reverse move lacked follow-through volume. Capital protected with Rs 0 exposure."
                    )
                    return

        # ── STATE 6: REVERSAL POSITION MONITORING (1 Lot) ───────────────────
        if self.state == 6:
            gain = current_prem - self.reversal_fill_prem
            if current_prem >= self.reversal_target:
                play_sound_alert(2)
                pnl = gain * 475
                self.daily_trade_count += 1
                send_telegram_alert(
                    f"🎯 *REVERSAL SCALP TARGET HIT! (+Rs {gain:.2f})*\n\n"
                    f"🎟️ *Contract:* {self.option_contract}\n"
                    f"💵 *Exit Premium:* Rs {current_prem:.2f} (Net: +Rs {pnl:,.0f})\n"
                    f"⚡ *Action:* BOOK PROFIT ON 1 LOT AT MARKET"
                )
                if self.daily_trade_count >= self.max_daily_trades or now.hour >= 14:
                    self.state = 4
                    send_telegram_alert(f"🏁 *DAILY TRADING COMPLETE!* Profit secured.")
                else:
                    self.reset_for_next_trade(now)
                    send_telegram_alert(f"⏳ *15-Min Cooldown Active*. Scanning for Trade 2 after {self.cooldown_until.strftime('%H:%M:%S')}.")
                return

            if current_prem <= self.reversal_sl:
                play_sound_alert(1)
                pnl = gain * 475
                self.daily_sl_count += 1
                self.daily_trade_count += 1
                send_telegram_alert(
                    f"🛑 *REVERSAL SL TRIGGERED @ Rs {current_prem:.2f}*\n\n"
                    f"🎟️ *Contract:* {self.option_contract} (Loss: -Rs {abs(pnl):,.0f})\n"
                    f"⚡ *Action:* CLOSE REVERSAL POSITION AT MARKET"
                )
                if self.daily_sl_count >= self.max_daily_sl or self.daily_trade_count >= self.max_daily_trades or now.hour >= 14:
                    self.state = 4
                    self.is_daily_locked_out = True
                    send_telegram_alert(f"🛑 *PERMANENT DAILY LOCKOUT ACTIVATED*. Max daily loss limit reached.")
                else:
                    self.reset_for_next_trade(now)
                    send_telegram_alert(f"⏳ *15-Min Cooldown Armed*. Scanning after {self.cooldown_until.strftime('%H:%M:%S')}.")
                return

        # ── STATE 0: SCANNING / CASH ─────────────────────────────────────────
        if self.state == 0:
            if not self.morning_sidelined:
                # Branch A: Morning Spike Rejection (09:15 - 09:20 AM)
                spike_condition = (current_spot >= upper_band) if not self.is_call else (current_spot <= lower_band)
                rejection = is_rejection_candle(last_bar, is_call=self.is_call)

                if spike_condition and rejection and not self.t1_alert_sent:
                    gates_passed, _ = check_pre_trade_gates(quote, open_prem=current_prem)
                    if gates_passed:
                        self.state = 1
                        self.t1_prem = current_prem
                        self.active_lots = 1
                        self.active_sl = round(max(current_prem - 8.0, 0.5), 2)
                        self.t1_alert_sent = True
                        self.entry_time = now
                        play_sound_alert(1)

                        send_telegram_alert(
                            f"🚨 *TRANCHE 1 ENTRY TRIGGERED! (1 Lot / 475 Qty)*\n\n"
                            f"🎟️ *Contract:* {self.option_contract}\n"
                            f"💵 *Fill Premium:* ~Rs {self.t1_prem:.2f} ({source})\n"
                            f"🛑 *Opening Emergency SL:* Rs {self.active_sl:.2f} (-Rs 8.00)\n"
                            f"📍 *Trigger:* 1-Min Rejection Candle at VWAP ±1.5σ (Spot Rs {current_spot})\n"
                            f"📊 *Order Flow BPI:* {bpi:.2f}\n"
                            f"⚡ *Action:* BUY 1 LOT (475 Qty) AT MARKET"
                        )
                        return

                # Branch B: Dynamic VWAP Window (09:18 - 09:30 AM / 09:35 - 09:45 AM on Gap Days)
                is_gap_day = getattr(self, "is_gap_day", False)
                in_vwap_window = (now.hour == 9 and 35 <= now.minute <= 45) if is_gap_day else (now.hour == 9 and 18 <= now.minute <= 30)
                if in_vwap_window:
                    vwap_confirmed = (current_spot < vwap) if not self.is_call else (current_spot > vwap)
                    if vwap_confirmed and not self.t2_alert_sent:
                        regime, dyn_target, dyn_sl = self.compute_dynamic_target_and_sl(df_1m, current_prem, current_spot)
                        self.state = 2
                        self.t1_prem = current_prem
                        self.t2_prem = current_prem
                        self.avg_entry = current_prem
                        self.active_lots = 1
                        self.active_regime = regime
                        self.active_target = dyn_target
                        self.active_sl = dyn_sl
                        self.t2_alert_sent = True
                        self.entry_time = now
                        play_sound_alert(2)

                        send_telegram_alert(
                            f"🚨 *ENTRY TRIGGERED! (Tier 1 Dynamic VWAP Window)*\n\n"
                            f"🎟️ *Contract:* {self.option_contract}\n"
                            f"💵 *Fill Premium:* ~Rs {self.avg_entry:.2f} ({source})\n"
                            f"⚡ *Regime:* {regime.replace('_', ' ')} (ATR Dynamic Sizing)\n"
                            f"🛑 *Armed SL:* Rs {self.active_sl:.2f} (-Rs {(self.avg_entry - self.active_sl):.2f})\n"
                            f"🎯 *Target:* Rs {self.active_target:.2f} (+Rs {(self.active_target - self.avg_entry):.2f})\n"
                            f"📍 *Spot:* Rs {current_spot:,.2f} (VWAP: Rs {vwap:,.2f} | BPI: {bpi:.2f})\n"
                            f"⚡ *Action:* ENTER 1 LOT (475 Qty) AT MARKET"
                        )
                        return

            # Evaluate Tier 2: Selective Continuation (09:30 AM – 11:00 AM)
            if (now.hour == 9 and now.minute >= 30) or (now.hour == 10):
                breakout_fired = self.eval_intraday_breakout(df_1m, current_spot, vwap, vol_surge, bpi, now)
                if breakout_fired:
                    return
            elif now.hour >= 11 and self.state == 0 and not getattr(self, "morning_cutoff_sent", False):
                self.state = 4
                self.morning_cutoff_sent = True
                print(f"[{now.strftime('%H:%M:%S')}] 🏁 11:00 AM MORNING SESSION CUTOFF: Locked in 100% Cash for the day.", flush=True)
                send_telegram_alert(
                    "🏁 *11:00 AM MORNING SESSION CUTOFF ACTIVATED*\n\n"
                    "🛡️ *Status:* Morning trading window complete.\n"
                    "🚫 *Action:* Terminal locked in 100% Cash for the day to avoid midday chop."
                )

        # ── STATE 1: TRANCHE 1 ACTIVE ────────────────────────────────────────
        elif self.state == 1:
            vwap_confirmed = (current_spot < vwap) if not self.is_call else (current_spot > vwap)
            if vwap_confirmed and not self.t2_alert_sent:
                self.state = 2
                self.t2_prem = current_prem
                self.avg_entry = round((self.t1_prem + self.t2_prem) / 2.0, 2)
                self.active_lots = 2
                regime, dyn_target, dyn_sl = self.compute_dynamic_target_and_sl(df_1m, self.avg_entry, current_spot)
                self.active_regime = regime
                self.active_target = dyn_target
                self.active_sl = dyn_sl
                self.t2_alert_sent = True
                play_sound_alert(2)

                send_telegram_alert(
                    f"🚨 *TRANCHE 2 ENTRY TRIGGERED! (1 Lot / 475 Qty)*\n\n"
                    f"🎟️ *Contract:* {self.option_contract}\n"
                    f"💵 *T2 Prem:* ~Rs {self.t2_prem:.2f} | *Avg Entry:* ~Rs {self.avg_entry:.2f}\n"
                    f"🛑 *Dynamic SL Armed:* Rs {self.active_sl:.2f} (-Rs {(self.avg_entry - self.active_sl):.2f})\n"
                    f"🎯 *Target:* Rs {self.active_target:.2f} (+Rs {(self.active_target - self.avg_entry):.2f})\n"
                    f"⚡ *Action:* BUY TRANCHE 2 (FULL 2 LOTS ACTIVE)"
                )
                return

            if current_prem <= self.active_sl:
                play_sound_alert(1)
                self.daily_sl_count += 1
                self.daily_trade_count += 1
                send_telegram_alert(f"🛑 *EMERGENCY SL HIT (-Rs 8.00) in Tranche 1.* Exit 1 Lot @ Rs {current_prem:.2f}.")

                if self.daily_sl_count >= self.max_daily_sl or self.daily_trade_count >= self.max_daily_trades or now.hour >= 14:
                    self.state = 4
                    self.is_daily_locked_out = True
                    send_telegram_alert(f"🛑 *PERMANENT DAILY LOCKOUT ACTIVATED*. Max daily loss reached.")
                else:
                    self.reset_for_next_trade(now)
                    send_telegram_alert(f"⏳ *15-Min Cooldown Armed*. Scanning after {self.cooldown_until.strftime('%H:%M:%S')}.")
                return

        # ── STATE 2: FULL POSITION ACTIVE ────────────────────────────────────
        elif self.state == 2:
            gain = current_prem - self.avg_entry

            if self.active_regime in ("SCALP", "BREAKOUT", "NIFTY_SCALP", "GAMMA_MOMENTUM"):
                if gain >= 2.0 and not self.trailed_cost_sent:
                    self.active_sl = self.avg_entry
                    self.trailed_cost_sent = True
                    play_sound_alert(1)
                    send_telegram_alert(f"🛡️ *BREAKOUT SL TRAILED TO COST (LTP Rs {current_prem:.2f})*")
            else:
                if gain >= 3.0 and not self.trailed_cost_sent:
                    self.active_sl = self.avg_entry
                    self.trailed_cost_sent = True
                    play_sound_alert(1)
                    send_telegram_alert(f"🛡️ *SL TRAILED TO COST (ZERO RISK) @ LTP Rs {current_prem:.2f}*")
                elif gain >= 5.0 and not self.trailed_locked_sent:
                    self.active_sl = round(self.avg_entry + 3.0, 2)
                    self.trailed_locked_sent = True
                    play_sound_alert(1)
                    send_telegram_alert(f"💰 *PROFIT LOCKED (+Rs 3.00) @ LTP Rs {current_prem:.2f}*")

            # Target Hit
            if current_prem >= self.active_target and not self.lot1_exit_sent:
                if self.active_regime in ("SCALP", "BREAKOUT", "NIFTY_SCALP", "GAMMA_MOMENTUM"):
                    play_sound_alert(2)
                    pnl = gain * (self.active_lots * 475)
                    self.daily_trade_count += 1
                    send_telegram_alert(
                        f"🎯 *TARGET HIT (+Rs {gain:.2f})!* 🚀\n\n"
                        f"🎟️ *Contract:* {self.option_contract}\n"
                        f"💵 *Exit Premium:* Rs {current_prem:.2f} (P&L: +Rs {pnl:,.0f})\n"
                        f"⚡ *Action:* CLOSE ENTIRE POSITION AT MARKET"
                    )
                    if self.daily_trade_count >= self.max_daily_trades or now.hour >= 14:
                        self.state = 4
                        send_telegram_alert(
                            f"🏁 *DAILY TRADING GOAL COMPLETE!* 💰\n\n"
                            f"📊 *Trades Taken:* {self.daily_trade_count}/{self.max_daily_trades}\n"
                            f"🛡️ *Status:* Profits secured in cash. System shutting down to prevent overtrading."
                        )
                    else:
                        self.reset_for_next_trade(now)
                        send_telegram_alert(
                            f"⏳ *15-Minute Profit Lock Cooldown Active*\n\n"
                            f"Trade 1 completed. Watchdog will scan for high-alpha Trade 2 after {self.cooldown_until.strftime('%H:%M:%S')}."
                        )
                    return
                else:
                    self.state = 3
                    self.lot1_exit_sent = True
                    play_sound_alert(2)
                    send_telegram_alert(
                        f"🎯 *LOT 1 TARGET HIT (+Rs {gain:.2f})!* 🚀\n\n"
                        f"🎟️ *Contract:* {self.option_contract}\n"
                        f"💵 *Exit Premium:* Rs {current_prem:.2f} (+Rs {(gain * 475):,.0f} Locked)\n"
                        f"🏃 *Lot 2 Status:* RUNNER ACTIVE — Trailing with 5-Min 20 EMA"
                    )
                    return

            # Stop Loss Hit
            if current_prem <= self.active_sl:
                if quote_status == "ESTIMATED":
                    print(f"[{time_str}] ⚠️ Live tick dropped. Synthetic BSM quote ({current_prem:.2f}) ignored to prevent false SL.", flush=True)
                    return
                play_sound_alert(1)
                pnl = (current_prem - self.avg_entry) * (self.active_lots * 475)
                self.daily_sl_count += 1
                self.daily_trade_count += 1
                send_telegram_alert(f"🛑 *STOP LOSS TRIGGERED @ Rs {current_prem:.2f}* (P&L: Rs {pnl:+,.0f}). Exit all.")

                if self.daily_sl_count >= self.max_daily_sl or self.daily_trade_count >= self.max_daily_trades or now.hour >= 14:
                    self.state = 4
                    self.is_daily_locked_out = True
                    self.weekly_sl_count += 1
                    weekly_lockout_msg = ""
                    if self.weekly_sl_count >= self.max_weekly_sl:
                        weekly_lockout_msg = "\n⛔ *WEEKLY SL LIMIT REACHED.* No more trades this week."
                    send_telegram_alert(
                        f"🛑 *PERMANENT DAILY LOCKOUT ACTIVATED*\n\n"
                        f"⚠️ *Reason:* Daily Stop Limit Reached ({self.daily_sl_count} SL hits | {self.daily_trade_count} trades).\n"
                        f"📅 *Weekly SL Count:* {self.weekly_sl_count}/{self.max_weekly_sl}\n"
                        f"🛡️ *Action:* Zero revenge trading allowed. Terminal locked until tomorrow 09:15 AM.{weekly_lockout_msg}"
                    )
                else:
                    self.reset_for_next_trade(now)
                    send_telegram_alert(
                        f"⏳ *Cooldown Armed after SL Hit*\n\n"
                        f"1 trade remaining today. Scanning only for strict, high-volume recovery setup after {self.cooldown_until.strftime('%H:%M:%S')}."
                    )
                return

        # ── STATE 3: RUNNER ACTIVE (Lot 2 Only) ──────────────────────────────
        elif self.state == 3:
            df_5m = t.history(period="1d", interval="5m")
            if not df_5m.empty and len(df_5m) >= 20:
                ema_20 = float(df_5m['Close'].ewm(span=20, adjust=False).mean().iloc[-1])
                ema_exit = (current_spot > ema_20) if not self.is_call else (current_spot < ema_20)
                if ema_exit:
                    self.daily_trade_count += 1
                    play_sound_alert(2)
                    send_telegram_alert(
                        f"🏁 *LOT 2 RUNNER EXITED (5-Min 20 EMA Cross)*\n\n"
                        f"📍 *Spot crossed 20 EMA:* Spot Rs {current_spot:.2f} vs 20 EMA Rs {ema_20:.2f}\n"
                        f"💵 *Exit Premium LTP:* ~Rs {current_prem:.2f}\n"
                        f"⚡ *Action:* SELL RUNNER AT MARKET"
                    )
                    if self.daily_trade_count >= self.max_daily_trades or now.hour >= 14:
                        self.state = 4
                        send_telegram_alert(
                            f"🏁 *DAILY TRADING GOAL COMPLETE!* 💰\n\n"
                            f"📊 *Trades Taken:* {self.daily_trade_count}/{self.max_daily_trades}\n"
                            f"🛡️ *Status:* Profits secured in cash. System shutting down."
                        )
                    else:
                        self.reset_for_next_trade(now)
                        send_telegram_alert(
                            f"⏳ *15-Minute Profit Lock Cooldown Active*\n\n"
                            f"Trade 1 runner completed. Watchdog will scan for Trade 2 after {self.cooldown_until.strftime('%H:%M:%S')}."
                        )
                    return


if __name__ == "__main__":
    watchdog = CDSLV3LiveWatchdog("CDSL.NS")
    watchdog.run()
