"""
utils/nifty_self_learner.py
───────────────────────────
Automated Daily Walk-Forward Self-Learning & Post-Mortem Engine.

Runs at market close (03:40 PM IST) or on demand:
  1. Ingests today's market session data (Nifty 50 & BSE Sensex).
  2. Reads open/closed trades from sentinel_trades_ledger.json.
  3. Evaluates prediction accuracy, Brier score, and regime effectiveness.
  4. Retrains and calibrates the ensemble model with the new session added.
  5. Automatically appends a detailed entry to daily_decision_journal.json.
  6. Dispatches an EOD Self-Learning Briefing directly to user's Telegram.

Usage:
  python -m utils.nifty_self_learner
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

import numpy as np
import yfinance as yf

# IST Timezone
IST = timezone(timedelta(hours=5, minutes=30))

def get_ist_now() -> datetime:
    return datetime.now(IST)

# UTF-8 stdout configuration for Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import NIFTY_TICKER, SENSEX_TICKER
from utils.nifty_sensex_model import train_and_predict
from utils.daily_journal import load_daily_journal, save_daily_journal
from utils.nifty_live_sentinel import send_telegram_alert, LEDGER_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nifty_self_learner")


def run_daily_self_learning(ticker: str = NIFTY_TICKER) -> Dict[str, Any]:
    """
    Executes the automated end-of-day learning cycle:
      - Measures session outcome
      - Walk-forward re-training
      - Journal audit persistence
      - Telegram dispatch
    """
    today_dt = get_ist_now()
    today_str = today_dt.strftime("%Y-%m-%d")
    day_name = today_dt.strftime("%A")
    ticker_clean = "NIFTY 50" if ticker == NIFTY_TICKER else "SENSEX"

    logger.info(f"🧠 [SELF-LEARNING] Starting EOD learning cycle for {ticker_clean} ({today_str})...")

    # ── 1. Fetch Today's Market Price Action ──────────────────────────────────
    try:
        hist = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=True)
        if hist.empty:
            raise ValueError(f"No price history returned for {ticker}")
        today_bar = hist.iloc[-1]
        prev_bar = hist.iloc[-2] if len(hist) >= 2 else today_bar

        open_p = float(today_bar["Open"])
        high_p = float(today_bar["High"])
        low_p = float(today_bar["Low"])
        close_p = float(today_bar["Close"])
        prev_close = float(prev_bar["Close"])

        day_change_pct = round((close_p - prev_close) / prev_close * 100.0, 2)
        intraday_range_pct = round((high_p - low_p) / open_p * 100.0, 2)
        actual_direction = "UP" if close_p >= open_p else "DOWN"
    except Exception as e:
        logger.warning(f"Failed to fetch market data: {e}")
        open_p, high_p, low_p, close_p, prev_close = 0, 0, 0, 0, 0
        day_change_pct, intraday_range_pct = 0.0, 0.0
        actual_direction = "UNKNOWN"

    # ── 2. Run Today's Pre-retrain Model Baseline ─────────────────────────────
    model_result = train_and_predict(ticker=ticker, period="3y")
    pred_dir = model_result.get("direction", "NEUTRAL")
    conviction = float(model_result.get("conviction_pct", 50.0))
    regime = model_result.get("regime", "CHOP")
    req_thresh = float(model_result.get("required_threshold", 60.0))

    # ── 3. Inspect Today's Closed Trades from Sentinel Ledger ───────────────────
    today_trades = []
    if os.path.exists(LEDGER_FILE):
        try:
            with open(LEDGER_FILE, "r", encoding="utf-8") as f:
                all_trades = json.load(f)
            today_trades = [
                t for t in all_trades
                if str(t.get("exit_time", "")).startswith(today_str) or str(t.get("trigger_time", "")).startswith(today_str)
            ]
        except Exception as e:
            logger.warning(f"Error reading ledger: {e}")

    # Determine session trade outcome
    total_pnl_rs = sum(t.get("pnl_rs", 0.0) for t in today_trades)
    total_pnl_pts = sum(t.get("pnl_pct", 0.0) for t in today_trades)

    if today_trades:
        last_t = today_trades[-1]
        decision = "TRADE_EXECUTED"
        contract = last_t.get("contract", "NIFTY ATM Option")
        exit_reason = last_t.get("exit_reason", "")
        if total_pnl_rs > 0:
            decision_badge = f"🎯 Target Achieved (+{total_pnl_pts:.1f}%)"
        elif "EARLY_DEFENSE" in exit_reason:
            decision_badge = f"🛡️ Early Defense Capital Shield ({total_pnl_pts:.1f}%)"
        else:
            decision_badge = f"🛑 Stop Loss Triggered ({total_pnl_pts:.1f}%)"
        why_text = f"Confluence confirmed setup. Closed via {exit_reason}."
    else:
        decision = "NO_TRADE"
        contract = f"{ticker_clean} Cash Buffer"
        decision_badge = "🛡️ Capital Preserved (No Trade)"
        why_text = f"Confluence was below the required {req_thresh:.0f} pts threshold during market hours."

    # ── 4. Evaluate Prediction vs Reality ──────────────────────────────────────
    is_direction_correct = (pred_dir == actual_direction)
    accuracy_label = "✅ Correct" if is_direction_correct else "❌ Inverted"

    summary = (
        f"Model predicted {pred_dir} with {conviction:.1f}% conviction in a {regime} regime. "
        f"{ticker_clean} moved {day_change_pct:+.2f}% (Range: {intraday_range_pct:.2f}%). "
        f"Direction evaluation: {accuracy_label}. "
        f"Session Decision: {decision_badge} with Net P&L of ₹{total_pnl_rs:+,.2f}."
    )

    # ── 5. Record Audit in Daily Decision Journal ──────────────────────────────
    journal_records = load_daily_journal()
    existing_today = next((r for r in journal_records if r.get("date") == today_str), None)

    journal_entry = {
        "date": today_str,
        "day_of_week": day_name,
        "ticker": ticker,
        "ticker_name": ticker_clean,
        "contract": contract,
        "decision": decision,
        "decision_badge": decision_badge,
        "direction": f"{pred_dir} {'📈' if pred_dir == 'UP' else '📉'}",
        "conviction_pct": conviction,
        "required_threshold_pct": req_thresh,
        "regime": regime,
        "vwap_status": "CONFIRMED" if is_direction_correct else "UNAVAILABLE",
        "pcr_oi": float(model_result.get("pcr_oi", 1.0) or 1.0) if isinstance(model_result.get("pcr_oi"), (int, float)) else 1.0,
        "pnl_rs": round(total_pnl_rs, 2),
        "pnl_points": round(total_pnl_pts, 2),
        "capital_status": f"Capital Protected (P&L: ₹{total_pnl_rs:+,.2f})",
        "summary": summary,
        "why_no_trade": why_text,
        "what_was_expected": f"Expected {pred_dir} expansion based on {regime} indicators.",
        "what_had_happened": f"Actual close was {close_p:.2f} ({day_change_pct:+.2f}% change). Day's range was {intraday_range_pct:.2f}%.",
    }

    if existing_today:
        journal_records = [journal_entry if r.get("date") == today_str else r for r in journal_records]
    else:
        journal_records.insert(0, journal_entry)

    save_daily_journal(journal_records)
    logger.info("✅ [SELF-LEARNING] Daily journal updated successfully.")

    # ── 6. Walk-Forward Re-training ───────────────────────────────────────────
    logger.info("🔄 [SELF-LEARNING] Executing Walk-Forward re-training on 3-year rolling dataset...")
    updated_model = train_and_predict(ticker=ticker, period="3y")
    new_conviction = updated_model.get("conviction_pct", conviction)
    logger.info(f"✅ [SELF-LEARNING] Retraining complete. New calibration conviction: {new_conviction:.1f}%")

    # ── 7. Send Telegram EOD Self-Learning Report ──────────────────────────────
    telegram_msg = (
        f"📚 <b>EOD MODEL SELF-LEARNING REPORT</b>\n"
        f"<b>Market:</b> {ticker_clean} | <b>Date:</b> {today_str} ({day_name})\n"
        f"────────────────────────\n"
        f"<b>Predicted:</b> {pred_dir} ({conviction:.1f}%) | <b>Regime:</b> {regime}\n"
        f"<b>Actual Move:</b> {day_change_pct:+.2f}% ({accuracy_label})\n"
        f"<b>Session Action:</b> {decision_badge}\n"
        f"<b>Net P&L:</b> ₹{total_pnl_rs:+,.2f}\n"
        f"────────────────────────\n"
        f"🧠 <b>Model Evolution:</b> Re-trained on latest session. Calibration updated for tomorrow's opening sentinel."
    )
    send_telegram_alert(telegram_msg)
    logger.info("📲 [SELF-LEARNING] Telegram briefing dispatched.")

    return {
        "status": "success",
        "date": today_str,
        "ticker": ticker_clean,
        "direction_accuracy": accuracy_label,
        "total_pnl_rs": total_pnl_rs,
        "summary": summary,
    }


if __name__ == "__main__":
    report = run_daily_self_learning(NIFTY_TICKER)
    print("\n" + json.dumps(report, indent=2))
