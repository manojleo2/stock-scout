# -*- coding: utf-8 -*-
"""
=============================================================================
 HDFCBANK AUTONOMOUS INTRADAY AUDIT ENGINE  —  utils/hdfc_intraday_audit.py
=============================================================================
 PURPOSE:
   Closed-loop feedback and self-learning system for HDFCBANK Intraday Specialist.
   Completely independent from utils/prediction_audit.py (CDSL audit engine).

 WHAT IT DOES:
   1. WHAT HAPPENED?     — Records 9:25 AM prediction & evaluates 3:30 PM actual close
   2. WHY DID IT HAPPEN? — Diagnoses divergence (Bank Nifty midday shift, VIX surge)
   3. ADAPTIVE LEARNING  — Computes rolling 10-session smoothed calibration offset

 LEDGER: hdfc_intraday_audit.json  (100% separate from prediction_audit.json)

 RULE: THIS FILE DOES NOT IMPORT FROM utils/prediction_audit.py OR utils/ml_model.py.
=============================================================================
"""

import os
import json
import logging
import datetime as dt
import pandas as pd

from utils.data_loader import get_stock_data
from utils.macro_factors import get_macro_market_cues
from config import (
    HDFCBANK_LEARNING_WINDOW,
    HDFCBANK_MAX_OFFSET_PCT,
    HDFCBANK_OFFSET_STEP_PCT,
    HDFCBANK_BANKNIFTY_TICKER,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hdfc_intraday_audit")

SYMBOL = "HDFCBANK.NS"

HDFC_INTRADAY_AUDIT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "hdfc_intraday_audit.json"
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PERSISTENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> dt.date:
    if not date_str:
        return dt.date.today()
    for fmt in ["%a, %d %b %Y", "%Y-%m-%d"]:
        try:
            return dt.datetime.strptime(date_str, fmt).date()
        except Exception:
            pass
    return dt.date.today()


def load_hdfc_intraday_audit_history() -> list:
    """Load HDFC intraday audit history from its independent JSON ledger."""
    if os.path.exists(HDFC_INTRADAY_AUDIT_FILE):
        try:
            with open(HDFC_INTRADAY_AUDIT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            logger.error(f"Error loading HDFC intraday audit: {e}")
    return []


def _save_hdfc_intraday_audit_history(audit_list: list):
    """Persist HDFC intraday audit to disk."""
    try:
        with open(HDFC_INTRADAY_AUDIT_FILE, "w", encoding="utf-8") as f:
            json.dump(audit_list, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving HDFC intraday audit: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. RECORD INTRADAY PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def record_hdfc_intraday_prediction(target_date_str: str, pred_result: dict):
    """
    Save HDFC intraday forecast to hdfc_intraday_audit.json.
    Updates in-place if already existing and pending evaluation.
    """
    history = load_hdfc_intraday_audit_history()
    blueprint = pred_result.get("blueprint", {})

    existing = next((r for r in history if r.get("target_date") == target_date_str), None)

    record_data = {
        "symbol": SYMBOL,
        "target_date": target_date_str,
        "prediction_time": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "predicted_direction": pred_result.get("direction"),
        "probability_up_pct": pred_result.get("probability_up_pct"),
        "probability_down_pct": pred_result.get("probability_down_pct"),
        "confidence": pred_result.get("confidence"),
        "action": blueprint.get("action", "HOLD"),
        "entry_price": blueprint.get("entry_price"),
        "stop_loss": blueprint.get("stop_loss"),
        "target_1": blueprint.get("target_1"),
        "target_2": blueprint.get("target_2"),
        "suggested_strike": blueprint.get("suggested_strike", "N/A"),
        "intraday_offset_applied": pred_result.get("intraday_offset_pct", 0.0),
        "top_features": list(pred_result.get("feature_importances", {}).items())[:6],
        "banking_radar": pred_result.get("banking_radar", {}),
        "actual_close": None,
        "actual_session_return_pct": None,
        "actual_direction": "⏳ Session in Progress / Pending 3:30 PM Close",
        "is_correct": None,
        "divergence_reasons": ["⏳ Intraday outcome will be evaluated after market closes at 3:30 PM."],
        "learning_note": "",
    }

    if existing:
        if existing.get("is_correct") is None:
            existing.update(record_data)
    else:
        history.append(record_data)

    _save_hdfc_intraday_audit_history(history)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROOT-CAUSE DIVERGENCE DIAGNOSIS
# ─────────────────────────────────────────────────────────────────────────────

def _diagnose_intraday_divergence(pred_direction: str, actual_ret_pct: float,
                                  target_date_str: str) -> list:
    """Diagnose why the intraday trend diverged from the 9:25 AM prediction."""
    reasons = []
    try:
        macro_df = get_macro_market_cues(period="1mo")
        if not macro_df.empty and len(macro_df) >= 2:
            latest = macro_df.iloc[-1]

            # 1. Bank Nifty midday reversal
            bn_ret = 0.0
            try:
                bn_df = get_stock_data(HDFCBANK_BANKNIFTY_TICKER, period="5d")
                if not bn_df.empty and len(bn_df) >= 2:
                    bn_ret = float(bn_df['Close'].pct_change(1).iloc[-1]) * 100.0
            except Exception:
                pass

            if abs(bn_ret) > 0.6:
                if bn_ret < -0.6 and "UP" in pred_direction:
                    reasons.append(
                        f"🏦 **Bank Nifty Midday Drop ({bn_ret:+.2f}%)**: Broad banking sector selling "
                        "pulled HDFC Bank down, overriding early morning accumulation."
                    )
                elif bn_ret > 0.6 and "DOWN" in pred_direction:
                    reasons.append(
                        f"🏦 **Bank Nifty Afternoon Surge (+{bn_ret:.2f}%)**: Institutional short-covering across "
                        "private banks lifted HDFC Bank into a green close."
                    )

            # 2. VIX surge
            vix_change = float(latest.get('VIX_Ret1', 0.0)) * 100.0
            if vix_change > 5.0 and "UP" in pred_direction:
                reasons.append(
                    f"🌋 **Volatility Expansion (+{vix_change:.1f}%)**: Intraday fear spike forced risk-off unwinding."
                )

    except Exception as e:
        logger.error(f"Error diagnosing intraday divergence: {e}")

    if not reasons:
        if "UP" in pred_direction and actual_ret_pct < 0:
            reasons.append("📌 **Afternoon Profit Booking**: Late institutional distribution reversed morning trend.")
        elif "DOWN" in pred_direction and actual_ret_pct > 0:
            reasons.append("📌 **DII Value Buying**: Institutional support emerged at intraday key support levels.")
        elif abs(actual_ret_pct) < 0.1:
            reasons.append("⚪ **Sideways Session / Chop Preserved**: Stock closed virtually flat. 100% Capital preserved in cash.")

    return reasons


# ─────────────────────────────────────────────────────────────────────────────
# 4. OUTCOME EVALUATION (Run on page load — skips evaluated records)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_hdfc_intraday_outcomes():
    """
    Check all pending HDFC intraday predictions and evaluate outcomes.
    Compares day's Close against Open price.
    """
    history = load_hdfc_intraday_audit_history()
    if not history:
        return history

    updated = False
    today = dt.date.today()

    for record in history:
        if record.get("is_correct") is not None:
            continue

        target_date_obj = _parse_date(record.get("target_date", ""))
        if target_date_obj > today:
            continue

        try:
            df_stock = get_stock_data(SYMBOL, period="1mo")
            if df_stock.empty or len(df_stock) < 2:
                continue

            df_dates = pd.to_datetime(df_stock.index).date
            matching = [i for i, d in enumerate(df_dates) if d == target_date_obj]
            if not matching:
                continue

            target_idx = matching[-1]
            target_bar = df_stock.iloc[target_idx]

            actual_close = round(float(target_bar['Close']), 2)
            open_price   = round(float(target_bar['Open']), 2)

            ret_pct = round(((actual_close - open_price) / open_price) * 100.0, 2) if open_price > 0 else 0.0

            actual_dir = "UP 📈" if ret_pct > 0 else ("DOWN 📉" if ret_pct < 0 else "FLAT ⚪")
            pred_dir   = record.get("predicted_direction", "")
            is_correct = bool(
                ("UP"   in pred_dir and ret_pct > 0) or
                ("DOWN" in pred_dir and ret_pct < 0)
            )

            record["actual_close"]              = actual_close
            record["actual_session_return_pct"] = ret_pct
            record["actual_direction"]          = actual_dir
            record["is_correct"]                = is_correct

            if is_correct:
                record["divergence_reasons"] = [
                    f"✅ **Intraday Trend Confirmed**: Closed at ₹{actual_close:,.2f} ({ret_pct:+.2f}%) matching AI forecast."
                ]
                record["learning_note"] = "✅ Daytime direction confirmed. No calibration offset adjustment needed."
            else:
                record["divergence_reasons"] = _diagnose_intraday_divergence(
                    pred_dir, ret_pct, record.get("target_date")
                )
                record["learning_note"] = "⚠️ Intraday divergence logged. Rolling calibration offset will update on next run."

            updated = True

        except Exception as e:
            logger.error(f"Error evaluating HDFC intraday outcome for {record.get('target_date')}: {e}")

    if updated:
        _save_hdfc_intraday_audit_history(history)

    return history


# ─────────────────────────────────────────────────────────────────────────────
# 5. ADAPTIVE CALIBRATION OFFSET ENGINE (Rolling 10-Session Smoothed Adjustment)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_hdfc_intraday_offset() -> tuple:
    """
    Compute adaptive probability offset using a rolling 10-session smoothed window.
    Avoids recency bias / knee-jerk 1-day swings.
    """
    evaluate_hdfc_intraday_outcomes()
    history = load_hdfc_intraday_audit_history()

    evaluated = [e for e in history if e.get("is_correct") is not None]
    recent_lessons = []

    if not evaluated:
        return 0.0, "🔄 HDFC Intraday Baseline (First session pending evaluation)", []

    for record in reversed(evaluated[-5:]):
        correct  = record.get("is_correct", False)
        date_str = record.get("target_date", "")
        pred_dir = record.get("predicted_direction", "")
        actual   = record.get("actual_session_return_pct", 0.0)
        action   = record.get("action", "")
        diag     = record.get("divergence_reasons", [])
        recent_lessons.append({
            "date": date_str,
            "result": "✅ Correct" if correct else "❌ Diverged",
            "predicted": pred_dir,
            "actual_pct": actual,
            "action": action,
            "diagnosis": diag[0] if diag else "",
        })

    window = evaluated[-HDFCBANK_LEARNING_WINDOW:]
    diverged = [e for e in window if not e.get("is_correct")]

    if not diverged:
        return 0.0, "✅ HDFC Intraday: 100% accuracy in recent sessions — No offset needed.", recent_lessons

    if len(diverged) == 1:
        return 0.0, "⚪ HDFC Intraday: 1 divergence in recent sessions — Treated as noise. No calibration adjustment.", recent_lessons

    up_misses   = sum(1 for e in diverged if "DOWN" in e.get("predicted_direction", "") and (e.get("actual_session_return_pct") or 0) > 0)
    down_misses = sum(1 for e in diverged if "UP" in e.get("predicted_direction", "") and (e.get("actual_session_return_pct") or 0) < 0)

    net_bias = up_misses - down_misses

    if net_bias == 0:
        return 0.0, "⚪ HDFC Intraday: Balanced divergence pattern. No calibration adjustment.", recent_lessons

    raw_offset = HDFCBANK_OFFSET_STEP_PCT * net_bias
    offset = float(max(-HDFCBANK_MAX_OFFSET_PCT, min(HDFCBANK_MAX_OFFSET_PCT, raw_offset)))

    if offset > 0:
        reason = f"🟢 **HDFC Intraday Offset +{offset:.1f}%**: Understated bullish trend in {up_misses} session(s). Upward calibration applied."
    else:
        reason = f"🔴 **HDFC Intraday Offset {offset:.1f}%**: Understated bearish selling in {down_misses} session(s). Downward calibration applied."

    return offset, reason, recent_lessons
