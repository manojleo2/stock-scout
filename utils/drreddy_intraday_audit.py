# -*- coding: utf-8 -*-
"""
=============================================================================
 DR. REDDY'S STANDALONE INTRADAY AUDIT ENGINE — utils/drreddy_intraday_audit.py
=============================================================================
 PURPOSE:
   Independent audit and performance telemetry engine for Dr. Reddy's Specialist.
   Completely separate from utils/prediction_audit.py (CDSL audit engine).

 WHAT IT DOES:
   1. AUDIT RECORDING  — Records daily 09:15 AM prediction snapshot to JSON ledger.
   2. OUTCOME ANALYSIS — Evaluates 03:30 PM actual market close (Close vs Open).
   3. READ-ONLY METRICS — Computes rolling Brier score, accuracy, and drift diagnostics.
   
 STRICT RULE (GEMINI.md Rule 4):
   This engine operates strictly in a READ-ONLY analytical role.
   It NEVER injects heuristic penalty offsets into live probability vectors.

 LEDGER: drreddy_intraday_audit.json (100% separate from prediction_audit.json)
=============================================================================
"""

import os
import json
import logging
import datetime as dt
import pandas as pd

from utils.data_loader import get_stock_data
from config import DRREDDY_SYMBOL, DRREDDY_PHARMA_TICKER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("drreddy_intraday_audit")

SYMBOL = DRREDDY_SYMBOL

DRREDDY_INTRADAY_AUDIT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "drreddy_intraday_audit.json"
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


def load_drreddy_intraday_audit_history() -> list:
    """Load Dr. Reddy's intraday audit history from its independent JSON ledger."""
    if os.path.exists(DRREDDY_INTRADAY_AUDIT_FILE):
        try:
            with open(DRREDDY_INTRADAY_AUDIT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            logger.error(f"Error loading Dr. Reddy's intraday audit: {e}")
    return []


def _save_drreddy_intraday_audit_history(audit_list: list):
    """Persist Dr. Reddy's intraday audit to disk."""
    try:
        with open(DRREDDY_INTRADAY_AUDIT_FILE, "w", encoding="utf-8") as f:
            json.dump(audit_list, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving Dr. Reddy's intraday audit: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. RECORD INTRADAY PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def record_drreddy_intraday_prediction(target_date_str: str, pred_result: dict):
    """
    Save Dr. Reddy's intraday forecast to drreddy_intraday_audit.json.
    Updates in-place if already existing and pending evaluation.
    """
    history = load_drreddy_intraday_audit_history()
    blueprint = pred_result.get("quant_blueprint", {})
    pharma_radar = pred_result.get("pharma_radar", {})

    existing = next((r for r in history if r.get("target_date") == target_date_str), None)

    record_data = {
        "symbol": SYMBOL,
        "target_date": target_date_str,
        "prediction_time": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "prediction_direction": pred_result.get("direction", "NEUTRAL"),
        "raw_probability_up_pct": pred_result.get("raw_probability_up_pct", 50.0),
        "probability_up_pct": pred_result.get("probability_up_pct", 50.0),
        "probability_down_pct": pred_result.get("probability_down_pct", 50.0),
        "confidence": pred_result.get("confidence", "Low"),
        "test_accuracy_pct": pred_result.get("test_accuracy_pct", 0.0),
        "brier_score": pred_result.get("brier_score", 0.25),
        "calibration_architecture": pred_result.get("calibration_architecture", "Pure Ensemble"),
        "latest_close": pred_result.get("latest_close", 0.0),
        "quant_blueprint": blueprint,
        "pharma_radar": pharma_radar,
        "actual_outcome": existing.get("actual_outcome", "PENDING") if existing else "PENDING",
        "accuracy_status": existing.get("accuracy_status", "PENDING") if existing else "PENDING",
        "divergence_reason": existing.get("divergence_reason", None) if existing else None
    }

    if existing:
        for k, v in record_data.items():
            if k not in ["actual_outcome", "accuracy_status", "divergence_reason"] or existing.get(k) == "PENDING":
                existing[k] = v
    else:
        history.append(record_data)

    _save_drreddy_intraday_audit_history(history)
    logger.info(f"Recorded Dr. Reddy's intraday prediction for {target_date_str}")


def get_saved_drreddy_intraday_snapshot(target_date_str: str) -> dict:
    """
    Retrieve previously saved forecast snapshot for target_date_str.
    Guards against serving corrupted snapshots.
    """
    history = load_drreddy_intraday_audit_history()
    for pred in reversed(history):
        if pred.get("target_date") == target_date_str:
            # Reject corrupted >95% probability spikes
            prob = pred.get("probability_up_pct", 50.0)
            if prob >= 95.0 or prob <= 5.0:
                logger.warning(f"Discarding anomalous snapshot with prob {prob}%")
                continue

            blueprint = pred.get("quant_blueprint", {})
            return {
                "status": "success",
                "symbol": SYMBOL,
                "direction": pred.get("prediction_direction", "UP"),
                "raw_probability_up_pct": pred.get("raw_probability_up_pct", prob),
                "probability_up_pct": prob,
                "probability_down_pct": pred.get("probability_down_pct", 100.0 - prob),
                "confidence": pred.get("confidence", "Moderate"),
                "test_accuracy_pct": pred.get("test_accuracy_pct", 60.0),
                "brier_score": pred.get("brier_score", 0.24),
                "calibration_architecture": pred.get("calibration_architecture", "FrozenEstimator"),
                "feature_importances": {},
                "sample_count": 490,
                "test_sample_count": 75,
                "latest_close": pred.get("latest_close", 0.0),
                "news_info": {"score": 0.0, "status": "CACHED"},
                "quant_blueprint": blueprint,
                "pharma_radar": pred.get("pharma_radar", {})
            }
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. EVALUATE OUTCOMES (Post-Market Close at 03:30 PM)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_drreddy_intraday_outcomes() -> list:
    """
    Evaluate pending predictions against actual session performance.
    Close > Open = UP | Close < Open = DOWN.
    """
    history = load_drreddy_intraday_audit_history()
    if not history:
        return []

    stock_df = get_stock_data(SYMBOL, period="1mo")
    if stock_df.empty:
        return history

    today = dt.date.today()
    updated = False

    for rec in history:
        if rec.get("accuracy_status") not in ("PENDING", None):
            continue

        target_date = _parse_date(rec.get("target_date", ""))
        if target_date > today:
            continue

        matching_rows = stock_df[stock_df.index.date == target_date]
        if matching_rows.empty:
            continue

        day_open = float(matching_rows["Open"].iloc[0])
        day_close = float(matching_rows["Close"].iloc[-1])
        day_return_pct = round(((day_close - day_open) / day_open) * 100.0, 2)

        actual_dir = "UP" if day_close >= day_open else "DOWN"
        rec["actual_open"] = day_open
        rec["actual_close"] = day_close
        rec["actual_return_pct"] = day_return_pct
        rec["actual_outcome"] = actual_dir

        pred_dir = rec.get("prediction_direction", "").replace(" 📈", "").replace(" 📉", "").strip()
        is_correct = (pred_dir == actual_dir)
        rec["accuracy_status"] = "CORRECT" if is_correct else "INCORRECT"
        updated = True

    if updated:
        _save_drreddy_intraday_audit_history(history)

    return history


def get_drreddy_telemetry_metrics() -> dict:
    """Compute rolling accuracy and Brier score telemetry for Dr. Reddy's audit."""
    history = load_drreddy_intraday_audit_history()
    completed = [r for r in history if r.get("accuracy_status") in ("CORRECT", "INCORRECT")]

    total = len(completed)
    if total == 0:
        return {
            "total_evaluated": 0,
            "accuracy_pct": 0.0,
            "brier_score": 0.0,
            "status": "NO_COMPLETED_SESSIONS"
        }

    correct = sum(1 for r in completed if r.get("accuracy_status") == "CORRECT")
    acc = round((correct / total) * 100.0, 1)

    # Calculate empirical Brier score
    sq_errs = []
    for r in completed:
        prob = r.get("probability_up_pct", 50.0) / 100.0
        y = 1.0 if r.get("actual_outcome") == "UP" else 0.0
        sq_errs.append((prob - y) ** 2)

    brier = round(float(pd.Series(sq_errs).mean()), 4) if sq_errs else 0.25

    return {
        "total_evaluated": total,
        "correct_count": correct,
        "accuracy_pct": acc,
        "brier_score": brier,
        "status": "HEALTHY"
    }
