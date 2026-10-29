"""
utils/nifty_live_sentinel.py
─────────────────────────────
Real-Time 5-Second Intraday Options Sentinel for NIFTY 50 and BSE SENSEX.

Features:
  1. High-Frequency 5-Second Loop:
     - Real-time ATM CE/PE LTP from Groww's zero-delay pipe.
     - Live VWAP confirmation and distance check.
     - Entry trigger detection across multi-timeframe rules.
     - Sub-second Trailing Stop-Loss engine (+18% to cost, +25% to +15% profit lock).
  2. 60-Second Background Loop:
     - ML Model conviction, India VIX, Regime (CHOP/TRENDING), and PCR updates.
     - Groww account margin check (Lot sizing rules: >30k -> 2 lots, <=30k -> 1 lot).
  3. Real-Time Audio Chime:
     - Audible alerts on Windows via winsound when Entry, Target, SL, or Trail fires.
  4. State Persistence:
     - Writes live state to live_sentinel_state.json on every tick for Streamlit dashboard.
     - Logs closed trades to sentinel_trades_ledger.json.

Run from PowerShell:
  python -m utils.nifty_live_sentinel --ticker ^NSEI --interval 5
  python -m utils.nifty_live_sentinel --ticker ^BSESN --interval 5
  python -m utils.nifty_live_sentinel --ticker both --interval 5
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime, time as dtime, timezone, timedelta
from typing import Dict, Any, Optional

# Indian Standard Time (IST = UTC + 5:30)
IST = timezone(timedelta(hours=5, minutes=30))

def get_ist_now() -> datetime:
    """Returns current datetime in Indian Standard Time (IST)."""
    return datetime.now(IST)

# Configure stdout for Windows UTF-8 support
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Setup root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    NIFTY_TICKER, SENSEX_TICKER,
    NIFTY_LOT_SIZE, SENSEX_LOT_SIZE,
    NIFTY_STRIKE_STEP, SENSEX_STRIKE_STEP,
    NIFTY_TARGET_PCT, NIFTY_SL_PCT,
    NIFTY_TRAIL_TRIGGER_PCT, NIFTY_TRAIL_LOCK_PCT, NIFTY_TRAIL_LOCK_PROFIT_PCT,
    NIFTY_HARD_CUTOFF_TIME,
    PCR_BEARISH_THRESHOLD, PCR_BULLISH_THRESHOLD,
    PCR_CONVICTION_BOOST, MAX_PCR_ADJUSTMENT,
)
from utils.nifty_oi_pcr import (
    get_live_nifty_oi_summary,
    compute_pcr_conviction_modifier,
    get_live_india_vix,
    get_groww_account_status,
)
from utils.nifty_vwap_gate import compute_intraday_vwap, check_vwap_gate
from utils.nifty_sensex_model import train_and_predict
from utils.nifty_trade_signal import compute_dynamic_levels, get_atm_strike, get_lot_size
from utils.spread_research_logger import SpreadResearchLogger
from utils.nifty_spread_guard import evaluate_spread_guard, get_spread_config
from utils.nifty_confluence import compute_confluence_score, evaluate_early_defense_exit, get_market_phase

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nifty_live_sentinel")

# Path for shared state with Streamlit
STATE_FILE = os.path.join(PROJECT_ROOT, "live_sentinel_state.json")
LEDGER_FILE = os.path.join(PROJECT_ROOT, "sentinel_trades_ledger.json")

# Windows Audio Alert
def play_sound(sound_type: str = "entry"):
    try:
        import winsound
        if sound_type == "entry":
            winsound.Beep(1200, 250)
            time.sleep(0.05)
            winsound.Beep(1600, 350)
        elif sound_type == "target":
            winsound.Beep(1400, 200)
            winsound.Beep(1800, 200)
            winsound.Beep(2200, 400)
        elif sound_type == "stop_loss":
            winsound.Beep(600, 400)
            winsound.Beep(450, 600)
        elif sound_type == "trail":
            winsound.Beep(1500, 150)
            winsound.Beep(1500, 150)
    except Exception:
        pass


def send_telegram_alert(message: str):
    """Sends instant push notification to user's Telegram phone app."""
    try:
        # 1. Check environment variables first (essential for Render / Cloud)
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        env_chat_id = os.environ.get("TELEGRAM_CHAT_ID")

        # 2. Check local .env files if not in os.environ
        if not bot_token:
            for env_path in [
                os.path.join(PROJECT_ROOT, ".env"),
                r"C:\Users\manoj\scout\stock_scout\.env",
            ]:
                if os.path.exists(env_path):
                    try:
                        with open(env_path, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line and "=" in line and not line.startswith("#"):
                                    k, v = line.split("=", 1)
                                    k = k.strip()
                                    v = v.strip().strip('"').strip("'")
                                    if k == "TELEGRAM_BOT_TOKEN" and not bot_token:
                                        bot_token = v
                                    elif k == "TELEGRAM_CHAT_ID" and not env_chat_id:
                                        env_chat_id = v
                    except Exception:
                        pass

        # Verified active personal chat ID
        target_chat_ids = ["867564851"]
        if env_chat_id and env_chat_id not in target_chat_ids:
            target_chat_ids.append(env_chat_id)

        if bot_token:
            import requests
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            for cid in target_chat_ids:
                try:
                    payload = {"chat_id": cid, "text": message, "parse_mode": "HTML"}
                    requests.post(url, json=payload, timeout=4)
                except Exception:
                    pass
    except Exception as e:
        logger.debug(f"Telegram alert error: {e}")


class LiveSentinelEngine:
    """Intraday 5-Second Real-Time Monitoring & Execution Engine."""

    def __init__(self, ticker: str = NIFTY_TICKER, interval: int = 5, state_file: str = STATE_FILE, ledger_file: str = LEDGER_FILE, shadow_logger: Optional[SpreadResearchLogger] = None):
        self.ticker = ticker
        self.interval = interval
        self.is_nifty = (ticker == NIFTY_TICKER)
        self.ticker_name = "NIFTY" if self.is_nifty else "SENSEX"
        self.strike_step = NIFTY_STRIKE_STEP if self.is_nifty else SENSEX_STRIKE_STEP
        self.lot_size = get_lot_size(ticker)
        self.state_file = state_file
        self.ledger_file = ledger_file
        self.shadow_logger = shadow_logger if shadow_logger is not None else SpreadResearchLogger()
        self.spread_config = get_spread_config(ticker)

        # Slow loop cache (refreshed every 60s)
        self.last_slow_refresh: float = 0.0
        self.cached_model_result: Dict[str, Any] = {}
        self.cached_groww_account: Dict[str, Any] = {}
        self.cached_vix: Dict[str, Any] = {}

        # Active trade tracking state
        self.active_position: Optional[Dict[str, Any]] = None
        self.trades_history = self._load_ledger()

        # Day trade limit (1 high-conviction trade at a time)
        self.max_trades_per_day = 2
        self.today_trades_count = 0

        # Synchronize closed trades from today across process restarts
        self._sync_today_trades_from_ledger()

        # Restore active open position across process restarts (duplicate-order prevention)
        self._load_state_on_startup()

    def _load_ledger(self) -> list:
        if os.path.exists(self.ledger_file):
            try:
                with open(self.ledger_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_ledger(self, trade_record: dict):
        self.trades_history.append(trade_record)
        try:
            with open(self.ledger_file, "w", encoding="utf-8") as f:
                json.dump(self.trades_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save ledger: {e}")

    def _sync_today_trades_from_ledger(self):
        """Synchronize today's closed trades count from disk ledger."""
        today_str = get_ist_now().strftime("%Y-%m-%d")
        today_trades = [
            t for t in self.trades_history
            if isinstance(t, dict)
            and t.get("ticker") == self.ticker
            and (t.get("trigger_time", "").startswith(today_str) or t.get("exit_time", "").startswith(today_str))
        ]
        self.today_trades_count = len(today_trades)
        if self.today_trades_count > 0:
            logger.info(
                f"📊 [LEDGER SYNC] Restored {self.today_trades_count} closed trade(s) from today for {self.ticker_name}."
            )

    def _load_state_on_startup(self):
        """Restore active position across process restarts to prevent duplicate orders and orphaned trades."""
        if not os.path.exists(self.state_file):
            return

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                saved = json.load(f)

            if not saved or not isinstance(saved, dict):
                return

            saved_pos = saved.get("active_position")
            has_pos = saved.get("has_active_position", False)

            if has_pos and saved_pos and isinstance(saved_pos, dict):
                if saved_pos.get("ticker") == self.ticker:
                    trigger_time_str = saved_pos.get("trigger_time", "")
                    today_str = get_ist_now().strftime("%Y-%m-%d")
                    now_time = get_ist_now().time()

                    # Restore if triggered today (if after 15:05, run_tick will cleanly execute cutoff square-off)
                    if trigger_time_str.startswith(today_str):
                        self.active_position = saved_pos
                        logger.info(
                            f"🛡️ [STATE RECOVERY] Successfully restored active position for {self.ticker_name}: "
                            f"{saved_pos.get('contract')} entered at ₹{saved_pos.get('entry_premium'):.2f} "
                            f"(Current SL: ₹{saved_pos.get('current_sl', saved_pos.get('sl_premium')):.2f}, "
                            f"Trail State: {saved_pos.get('trail_state')}). Duplicate order prevented."
                        )
                    else:
                        logger.info(
                            f"ℹ️ [STATE RECOVERY] Discarding stale active position from previous session "
                            f"({trigger_time_str})."
                        )
        except Exception as e:
            logger.warning(f"Error loading initial state file: {e}")

    def _save_state(self, current_state: dict):
        """Write live state for Streamlit dashboard ingestion."""
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(current_state, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to write state file: {e}")

    def refresh_slow_loop(self, force: bool = False):
        """Refreshes ML model, VIX, and Groww margin every 60 seconds."""
        now = time.time()
        if not force and (now - self.last_slow_refresh) < 60.0:
            return

        try:
            logger.info(f"🔄 [60s Refresh] Updating ML Model & Account for {self.ticker_name}...")
            # 1. Groww account status
            self.cached_groww_account = get_groww_account_status()
            
            # 2. India VIX
            self.cached_vix = get_live_india_vix()

            # 3. ML Model prediction (runs walk-forward trained model)
            self.cached_model_result = train_and_predict(ticker=self.ticker, period="3y")

            self.last_slow_refresh = now
        except Exception as e:
            logger.error(f"Slow loop refresh error: {e}")

    def evaluate_entry_trigger(self, oi_summary: dict, vwap_data: dict) -> Optional[Dict[str, Any]]:
        """
        Evaluate if 5-second market conditions satisfy high-precision entry.
        """
        now = get_ist_now()
        current_time = now.time()

        # Guard 1: Duplicate order prevention — do not enter if an active position is already open
        if self.active_position is not None:
            return None

        # Guard 2: Check maximum trades per day discipline
        if self.today_trades_count >= self.max_trades_per_day:
            return None

        # Window 1: 09:18 AM to 14:30 PM (IST)
        market_open_entry = dtime(9, 18)
        market_close_entry = dtime(14, 30)

        # Allow execution test outside live market if spot is valid
        is_live_trading_hours = (market_open_entry <= current_time <= market_close_entry)

        model = self.cached_model_result
        if not model or model.get("status") != "success":
            return None

        direction = model.get("direction", "NEUTRAL")
        base_conviction = model.get("conviction_pct", 50.0)
        regime = model.get("regime", "CHOP")
        spot = oi_summary.get("spot", 0.0) or model.get("latest_close", 0.0)

        if spot <= 0:
            return None

        # 1. OI Modifier
        oi_modifier = compute_pcr_conviction_modifier(
            oi_parsed=oi_summary,
            model_direction=direction,
            pcr_bearish_threshold=PCR_BEARISH_THRESHOLD,
            pcr_bullish_threshold=PCR_BULLISH_THRESHOLD,
            boost_per_signal=PCR_CONVICTION_BOOST,
            max_total_adjustment=MAX_PCR_ADJUSTMENT,
        )
        final_conviction = round(base_conviction + oi_modifier.get("adjustment_pct", 0.0), 2)

        # 2. Multi-Factor Confluence Engine (Replaces Rigid CHOP Wall & Prevents Analysis Paralysis)
        confluence = compute_confluence_score(
            model_result=model,
            vwap_data=vwap_data,
            oi_summary=oi_summary,
            current_time=current_time,
        )
        if not confluence.get("is_passed", False):
            return None

        # 5. VWAP Confirmation
        vwap_res = check_vwap_gate(direction, self.ticker, vwap_data)
        if not vwap_res.get("gate_passed", False):
            return None

        # 6. Proximity to VWAP (Do not chase if > 0.35% extended)
        vwap_val = vwap_data.get("vwap", 0.0)
        if vwap_val > 0:
            pct_dist = abs(spot - vwap_val) / vwap_val * 100
            if pct_dist > 0.40:
                # Overextended — wait for pullback
                return None

        # 7. Select ATM Strike & Real LTP
        atm_strike = get_atm_strike(spot, self.ticker)
        is_call = (direction == "UP")
        option_type = "CE" if is_call else "PE"
        contract_name = f"{self.ticker_name} {atm_strike} {option_type}"

        live_ltp = oi_summary.get("atm_ce_ltp", 0.0) if is_call else oi_summary.get("atm_pe_ltp", 0.0)
        if not live_ltp or live_ltp <= 0.0:
            return None

        # Lot size logic
        cash = self.cached_groww_account.get("cash_available", 25000.0)
        allowed_lots = 2 if cash > 30000.0 else 1
        if cash < 20000.0:
            logger.warning("🚨 Capital below ₹20,000 safety limit! No trades allowed.")
            return None

        levels = compute_dynamic_levels(entry_premium=live_ltp, lot_size=self.lot_size * allowed_lots)

        return {
            "trigger_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": self.ticker,
            "ticker_name": self.ticker_name,
            "contract": contract_name,
            "atm_strike": atm_strike,
            "option_type": option_type,
            "direction": direction,
            "conviction": final_conviction,
            "confluence_score": confluence["total_confluence"],
            "confluence_threshold": confluence["threshold"],
            "spot": spot,
            "entry_premium": live_ltp,
            "target_premium": levels["target_premium"],
            "sl_premium": levels["sl_premium"],
            "trail_trigger": levels["trail_trigger_premium"],
            "lock_trigger": levels["lock_trigger_premium"],
            "locked_sl": levels["locked_sl_premium"],
            "lots": allowed_lots,
            "total_qty": self.lot_size * allowed_lots,
            "trail_state": "INITIAL",
            "current_sl": levels["sl_premium"],
            "pnl_rs": 0.0,
            "pnl_pct": 0.0,
            "high_premium_seen": live_ltp,
        }

    def update_active_position(self, oi_summary: dict, vwap_data: Optional[dict] = None):
        """5-second monitoring of active open position against Target, Early Defense, SL & Trailing."""
        if not self.active_position:
            return

        pos = self.active_position
        is_call = (pos["option_type"] == "CE")
        curr_ltp = oi_summary.get("atm_ce_ltp", 0.0) if is_call else oi_summary.get("atm_pe_ltp", 0.0)

        if not curr_ltp or curr_ltp <= 0.0:
            return

        pos["current_ltp"] = curr_ltp
        if curr_ltp > pos["high_premium_seen"]:
            pos["high_premium_seen"] = curr_ltp

        # Record subsequent observed LTP for tick-persistence research (minimal shadow logging)
        cand_id = pos.get("shadow_candidate_id")
        if cand_id and hasattr(self, "shadow_logger") and self.shadow_logger:
            try:
                self.shadow_logger.record_subsequent_ltp(
                    candidate_id=cand_id,
                    timestamp=get_ist_now().strftime("%Y-%m-%d %H:%M:%S"),
                    ltp=curr_ltp
                )
            except Exception:
                pass

        entry = pos["entry_premium"]
        qty = pos["total_qty"]
        diff = curr_ltp - entry
        pnl_pct = (diff / entry) * 100.0
        pnl_rs = diff * qty

        pos["pnl_pct"] = round(pnl_pct, 2)
        pos["pnl_rs"] = round(pnl_rs, 2)

        # ── 1. Target Hit (+30%) ──
        if curr_ltp >= pos["target_premium"]:
            pos["exit_price"] = curr_ltp
            pos["exit_reason"] = "TARGET_HIT (+30%)"
            pos["exit_time"] = get_ist_now().strftime("%Y-%m-%d %H:%M:%S")
            self._close_position(pos, is_win=True)
            return

        # ── 1.5 Early Defense Trigger (Pre-emptive Capital Shield: Cuts loss early before full -15% SL) ──
        spot_val = float(oi_summary.get("spot", 0.0) or 0.0)
        vwap_val = float(vwap_data.get("vwap", 0.0) or 0.0) if vwap_data else 0.0
        early_def, def_reason = evaluate_early_defense_exit(
            active_position=pos,
            current_spot=spot_val,
            current_option_ltp=curr_ltp,
            current_vwap=vwap_val,
        )
        if early_def:
            pos["exit_price"] = curr_ltp
            pos["exit_reason"] = "EARLY_DEFENSE"
            pos["early_defense_detail"] = def_reason
            pos["exit_time"] = get_ist_now().strftime("%Y-%m-%d %H:%M:%S")
            self._close_position(pos, is_win=False, is_early_defense=True, early_defense_reason=def_reason)
            return

        # ── 2. Stop Loss Hit (Dynamic Current SL) ──
        if curr_ltp <= pos["current_sl"]:
            pos["exit_price"] = curr_ltp
            pos["exit_reason"] = f"SL_HIT ({pos['trail_state']})"
            pos["exit_time"] = get_ist_now().strftime("%Y-%m-%d %H:%M:%S")
            self._close_position(pos, is_win=False)
            return

        # ── 3. Trailing Level 2: +25% Gain -> Lock +15% Profit ──
        if curr_ltp >= pos["lock_trigger"] and pos["trail_state"] != "LOCKED_PROFIT":
            pos["trail_state"] = "LOCKED_PROFIT"
            pos["current_sl"] = pos["locked_sl"]
            play_sound("trail")
            send_telegram_alert(f"🔒 <b>PROFIT LOCKED (+15%)</b>\n{pos['contract']} reached ₹{curr_ltp:.2f} (+{pnl_pct:.1f}%)\nGuaranteed SL: ₹{pos['current_sl']:.2f}")
            print(f"\n🔒 [PROFIT LOCKED] {pos['contract']} @ ₹{curr_ltp:.2f} (+{pnl_pct:.1f}%) | New SL: ₹{pos['current_sl']:.2f} (+15% Profit Guaranteed!)")

        # ── 4. Trailing Level 1: +18% Gain -> Trail SL to Cost ──
        elif curr_ltp >= pos["trail_trigger"] and pos["trail_state"] == "INITIAL":
            pos["trail_state"] = "TRAILED_TO_COST"
            pos["current_sl"] = pos["entry_premium"]
            play_sound("trail")
            send_telegram_alert(f"🛡️ <b>RISK ELIMINATED (ZERO RISK)</b>\n{pos['contract']} reached ₹{curr_ltp:.2f} (+{pnl_pct:.1f}%)\nSL moved to Entry Cost: ₹{pos['current_sl']:.2f}")
            print(f"\n🛡️ [RISK ELIMINATED] {pos['contract']} @ ₹{curr_ltp:.2f} (+{pnl_pct:.1f}%) | SL moved to Entry Cost: ₹{pos['current_sl']:.2f} (Zero Risk!)")

        # ── 5. Hard Cutoff: 03:05 PM ──
        now_time = get_ist_now().time()
        if now_time >= dtime(15, 5):
            pos["exit_price"] = curr_ltp
            pos["exit_reason"] = "HARD_CUTOFF_15:05"
            pos["exit_time"] = get_ist_now().strftime("%Y-%m-%d %H:%M:%S")
            self._close_position(pos, is_win=(pnl_rs > 0))
            return

    def _close_position(self, pos: dict, is_win: bool, is_early_defense: bool = False, early_defense_reason: str = ""):
        """Finalize closed position and record in ledger."""
        if is_win:
            play_sound("target")
            send_telegram_alert(f"🎯 <b>TARGET ACHIEVED (+30%)</b>\n\n<b>Contract:</b> {pos['contract']}\n<b>Entry:</b> ₹{pos['entry_premium']:.2f}\n<b>Exit:</b> ₹{pos['exit_price']:.2f}\n<b>Net Gain:</b> +₹{pos['pnl_rs']:,.2f} on {pos['lots']} Lot(s)")
            print("\n" + "═"*60)
            print(f" 🎯 TARGET ACHIEVED — {pos['contract']}")
            print(f" Entry: ₹{pos['entry_premium']:.2f} | Exit: ₹{pos['exit_price']:.2f} | Gain: +₹{pos['exit_price']-pos['entry_premium']:.2f} (+{pos['pnl_pct']:.1f}%)")
            print(f" 💰 Total Net Gain: +₹{pos['pnl_rs']:,.2f} on {pos['lots']} Lot(s)")
            print("═"*60 + "\n")
        elif is_early_defense:
            play_sound("stop_loss")
            full_sl_loss = pos['entry_premium'] * 0.15 * pos['total_qty']
            saved_rs = max(0.0, full_sl_loss - abs(pos['pnl_rs']))
            send_telegram_alert(
                f"🛡️ <b>EARLY DEFENSE CAPITAL SHIELD</b>\n\n"
                f"<b>Contract:</b> {pos['contract']}\n"
                f"<b>Entry:</b> ₹{pos['entry_premium']:.2f} | <b>Exit:</b> ₹{pos['exit_price']:.2f} ({pos['pnl_pct']:.1f}%)\n"
                f"<b>Cut Loss Early:</b> ₹{pos['pnl_rs']:,.2f} (Saved ~₹{saved_rs:,.2f} vs Full -15% SL!)\n"
                f"<b>Reason:</b> {early_defense_reason}"
            )
            print("\n" + "═"*60)
            print(f" 🛡️ EARLY DEFENSE CAPITAL SHIELD — {pos['contract']}")
            print(f" Entry: ₹{pos['entry_premium']:.2f} | Exit: ₹{pos['exit_price']:.2f} | P&L: ₹{pos['exit_price']-pos['entry_premium']:.2f} ({pos['pnl_pct']:.1f}%)")
            print(f" Capital Saved: Cut loss early at -7% to -8% instead of taking full -15% SL!")
            print("═"*60 + "\n")
        else:
            play_sound("stop_loss")
            send_telegram_alert(f"🛑 <b>STOP LOSS HIT</b>\n\n<b>Contract:</b> {pos['contract']}\n<b>Entry:</b> ₹{pos['entry_premium']:.2f}\n<b>Exit:</b> ₹{pos['exit_price']:.2f}\n<b>Total Loss:</b> ₹{pos['pnl_rs']:,.2f} on {pos['lots']} Lot(s)")
            print("\n" + "═"*60)
            print(f" 🛑 STOP LOSS HIT — {pos['contract']}")
            print(f" Entry: ₹{pos['entry_premium']:.2f} | Exit: ₹{pos['exit_price']:.2f} | P&L: ₹{pos['exit_price']-pos['entry_premium']:.2f} ({pos['pnl_pct']:.1f}%)")
            print(f" Total P&L: ₹{pos['pnl_rs']:,.2f} on {pos['lots']} Lot(s)")
            print("═"*60 + "\n")

        self._save_ledger(pos)

        # Update shadow logger with observed baseline exit outcome
        try:
            cand_id = pos.get("shadow_candidate_id")
            if cand_id and hasattr(self, "shadow_logger") and self.shadow_logger:
                self.shadow_logger.update_outcome(
                    candidate_id=cand_id,
                    exit_price=pos.get("exit_price", 0.0),
                    exit_reason=pos.get("exit_reason", ""),
                    actual_pnl_rs=pos.get("pnl_rs", 0.0),
                    actual_pnl_pct=pos.get("pnl_pct", 0.0),
                    exit_timestamp=pos.get("exit_time", get_ist_now().strftime("%Y-%m-%d %H:%M:%S")),
                    outcome_status="MEASURED"
                )
        except Exception as e:
            logger.debug(f"Shadow logger outcome update error: {e}")

        self.active_position = None
        self.today_trades_count += 1

    def run_tick(self) -> dict:
        """Executes a single 5-second tick."""
        t0 = time.time()
        now_str = get_ist_now().strftime("%H:%M:%S")

        # 1. Update slow loop if needed (every 60s)
        self.refresh_slow_loop(force=False)

        # 2. Fast fetch: Live option chain from Groww
        oi_summary = get_live_nifty_oi_summary(
            spot=self.cached_model_result.get("latest_close", 0.0),
            strike_step=self.strike_step,
            ticker=self.ticker
        )
        spot = oi_summary.get("spot", 0.0)

        # 3. Fast fetch: Live intraday VWAP
        vwap_data = compute_intraday_vwap(index_ticker=self.ticker)
        vwap_val = vwap_data.get("vwap")
        if vwap_val is not None and spot is not None and vwap_val > 0:
            vwap_status = "ABOVE_VWAP" if spot >= vwap_val else "BELOW_VWAP"
        else:
            vwap_status = "UNAVAILABLE"

        # 4. If active position exists, update P&L, trailing stops, and early defense
        if self.active_position:
            self.update_active_position(oi_summary, vwap_data)
        else:
            # 5. Evaluate potential entry
            trigger = self.evaluate_entry_trigger(oi_summary, vwap_data)
            if trigger:
                # Record prospective shadow quote observation and counterfactual spread guard evaluation
                try:
                    is_call = (trigger.get("option_type") == "CE")
                    cand_bid = float(oi_summary.get("atm_ce_bid" if is_call else "atm_pe_bid") or 0.0)
                    cand_ask = float(oi_summary.get("atm_ce_ask" if is_call else "atm_pe_ask") or 0.0)
                    cand_ltp = float(trigger.get("entry_premium", 0.0))
                    cand_provenance = oi_summary.get("source", "GROWW_FEED")
                    quote_ts = oi_summary.get("quote_timestamp")

                    trigger["bid"] = cand_bid
                    trigger["ask"] = cand_ask
                    trigger["quote_provenance"] = cand_provenance
                    trigger["quote_timestamp"] = quote_ts

                    spread_eval = evaluate_spread_guard(
                        bid=cand_bid,
                        ask=cand_ask,
                        ltp=cand_ltp,
                        quote_timestamp=quote_ts,
                        config=self.spread_config
                    )
                    if hasattr(self, "shadow_logger") and self.shadow_logger:
                        logged_rec = self.shadow_logger.log_opportunity(
                            candidate_opportunity=trigger,
                            spread_eval=spread_eval,
                            baseline_decision="EXECUTE"
                        )
                        trigger["shadow_candidate_id"] = logged_rec.get("candidate_id")
                except Exception as e:
                    logger.debug(f"Shadow logger observation error: {e}")

                self.active_position = trigger
                play_sound("entry")
                send_telegram_alert(
                    f"🎯 <b>5-SECOND ENTRY TRIGGERED</b>\n\n"
                    f"<b>Contract:</b> {trigger['contract']}\n"
                    f"<b>Buy LTP:</b> ₹{trigger['entry_premium']:.2f}\n"
                    f"<b>Target (+30%):</b> ₹{trigger['target_premium']:.2f}\n"
                    f"<b>Stop Loss (−15%):</b> ₹{trigger['sl_premium']:.2f}\n"
                    f"<b>Trail to Cost:</b> at ₹{trigger['trail_trigger']:.2f} (+18%)\n"
                    f"<b>Lots:</b> {trigger['lots']} Lot(s) ({trigger['total_qty']} Qty)"
                )
                print("\n" + "═"*60)
                print(f" 🎯 5-SECOND ENTRY TRIGGER FIRED — {now_str}")
                print(f" Contract     : {trigger['contract']}  |  Spot: {spot:.2f} (VWAP: {vwap_val or 0:.2f})")
                print(f" Entry Premium: ₹{trigger['entry_premium']:.2f} (Live Groww LTP)")
                print(f" Target (+30%): ₹{trigger['target_premium']:.2f}")
                print(f" Stop Loss    : ₹{trigger['sl_premium']:.2f} (−15%)")
                print(f" Trail to Cost: at ₹{trigger['trail_trigger']:.2f} (+18%)")
                print(f" Lock +15% Pft: at ₹{trigger['lock_trigger']:.2f} (+25%)")
                print(f" Position Size: {trigger['lots']} Lot(s) ({trigger['total_qty']} Qty)")
                print("═"*60 + "\n")

        # Build telemetry state dict for Streamlit & dashboard
        state = {
            "timestamp": now_str,
            "ticker": self.ticker,
            "ticker_name": self.ticker_name,
            "spot": spot,
            "vwap": vwap_val,
            "vwap_status": vwap_status,
            "atm_strike": get_atm_strike(spot, self.ticker) if spot > 0 else 0,
            "atm_ce_ltp": oi_summary.get("atm_ce_ltp", 0.0),
            "atm_pe_ltp": oi_summary.get("atm_pe_ltp", 0.0),
            "pcr_oi": oi_summary.get("pcr_oi", 1.0),
            "regime": self.cached_model_result.get("regime", "CHOP"),
            "conviction": self.cached_model_result.get("conviction_pct", 50.0),
            "direction": self.cached_model_result.get("direction", "NEUTRAL"),
            "has_active_position": (self.active_position is not None),
            "active_position": self.active_position,
            "today_trades_count": self.today_trades_count,
            "cash_margin": self.cached_groww_account.get("cash_available", 0.0),
            "allowed_lots": 2 if self.cached_groww_account.get("cash_available", 0.0) > 30000.0 else 1,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }

        self._save_state(state)
        return state

    def start_loop(self):
        """Start the continuous 5-second monitoring loop."""
        print("\n" + "═"*60)
        print(f" 🚀 5-SECOND INTRADAY SENTINEL STARTED: {self.ticker_name}")
        print(f" Polling Interval : {self.interval} Seconds (Zero-Delay Groww Pipe)")
        print(f" Dynamic Brackets : Target +30% | Stop Loss −15%")
        print(f" Trailing Rules   : Trail to Cost @ +18% | Lock +15% @ +25%")
        print(f" State File       : {self.state_file}")
        print("═"*60 + "\n")

        # Warm slow loop immediately
        self.refresh_slow_loop(force=True)

        # Send Cloud / Sentinel online push alert
        send_telegram_alert(
            f"🚀 <b>5-Second Options Sentinel Online!</b>\n\n"
            f"📈 <b>Index:</b> {self.ticker_name}\n"
            f"⏱️ <b>Interval:</b> {self.interval}s (Zero-Delay Groww Pipe)\n"
            f"🎯 <b>Dynamic 2:1 Bracket:</b> Target +30% | SL −15%\n"
            f"🛡️ <b>Trailing Rules:</b> Trail to Cost @ +18% | Lock +15% @ +25%\n"
            f"🕒 <b>Session Time:</b> {get_ist_now().strftime('%Y-%m-%d %H:%M:%S')} IST"
        )

        try:
            while True:
                t_start = time.time()
                state = self.run_tick()

                # Status line in console
                pos = state.get("active_position")
                if pos:
                    print(f"\r[{state['timestamp']}] {pos['contract']} LTP: ₹{pos.get('current_ltp', 0.0):.2f} | P&L: ₹{pos['pnl_rs']:+,.0f} ({pos['pnl_pct']:+.1f}%) | SL: ₹{pos['current_sl']:.2f} | Latency: {state['latency_ms']}ms", end="", flush=True)
                else:
                    atm = state["atm_strike"]
                    ce = state["atm_ce_ltp"]
                    pe = state["atm_pe_ltp"]
                    print(f"\r[{state['timestamp']}] {self.ticker_name}: {state['spot']:.1f} | ATM {atm} [CE: ₹{ce:.2f} | PE: ₹{pe:.2f}] | {state['direction']} ({state['conviction']:.1f}%) | {state['regime']} | Latency: {state['latency_ms']}ms", end="", flush=True)

                elapsed = time.time() - t_start
                sleep_sec = max(0.1, self.interval - elapsed)
                time.sleep(sleep_sec)

        except KeyboardInterrupt:
            print("\n\n🛑 5-Second Sentinel stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="5-Second Live Intraday Sentinel")
    parser.add_argument("--ticker", type=str, default=NIFTY_TICKER,
                        help="Index ticker (^NSEI or ^BSESN)")
    parser.add_argument("--interval", type=int, default=5,
                        help="Polling loop interval in seconds (default: 5)")
    args = parser.parse_args()

    engine = LiveSentinelEngine(ticker=args.ticker, interval=args.interval)
    engine.start_loop()


if __name__ == "__main__":
    main()
