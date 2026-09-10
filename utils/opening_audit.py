import os
import json
import logging
import pandas as pd
import datetime as dt
from utils.data_loader import get_stock_data
from utils.macro_factors import get_macro_market_cues

logging.basicConfig(level=logging.INFO)

GAP_AUDIT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "opening_gap_audit.json")

def parse_target_date(date_str: str) -> dt.date:
    """Parse string formatted date into datetime.date object."""
    if not date_str:
        return dt.date.today()
    try:
        return dt.datetime.strptime(date_str, "%a, %d %b %Y").date()
    except Exception:
        try:
            return dt.datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            return dt.date.today()

def load_saved_gap_audit_history() -> list:
    """Load opening gap audit history from JSON file."""
    if os.path.exists(GAP_AUDIT_FILE):
        try:
            with open(GAP_AUDIT_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, list):
                    return saved
        except Exception as e:
            logging.error(f"Error loading opening gap audit history: {e}")
    return []

def save_gap_audit_history(audit_list: list):
    """Save opening gap audit history to JSON file."""
    try:
        with open(GAP_AUDIT_FILE, "w", encoding="utf-8") as f:
            json.dump(audit_list, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving opening gap audit history: {e}")

def record_opening_gap_prediction(symbol: str, target_date_str: str, pred_result: dict):
    """
    Log an opening gap prediction generated at 3:05-3:15 PM for tomorrow's 9:15 AM market open.
    """
    history = load_saved_gap_audit_history()

    existing = next((r for r in history if r.get("symbol") == symbol and r.get("target_date") == target_date_str), None)

    options_call = pred_result.get("options_call", {})
    record_data = {
        "symbol": symbol,
        "target_date": target_date_str,
        "prediction_time": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "predicted_gap_direction": pred_result.get("direction"),
        "probability_up_pct": pred_result.get("probability_up_pct"),
        "probability_down_pct": pred_result.get("probability_down_pct"),
        "confidence": pred_result.get("confidence"),
        "options_action": options_call.get("action", "HOLD"),
        "suggested_strike": options_call.get("suggested_strike", "N/A"),
        "baseline_3pm_close": pred_result.get("current_price"),
        "top_features": list(pred_result.get("feature_importances", {}).items())[:6],
        "actual_915_open": None,
        "actual_gap_rs": None,
        "actual_gap_pct": None,
        "actual_gap_direction": "⏳ Pending Next Market Open (9:15 AM)",
        "is_correct": None,
        "divergence_reasons": ["⏳ Opening gap outcome will be evaluated after market opens tomorrow at 9:15 AM."]
    }

    if existing:
        if existing.get("is_correct") is None:
            existing.update(record_data)
    else:
        history.append(record_data)

    save_gap_audit_history(history)
    return history

def diagnose_gap_divergence(symbol: str, pred_direction: str, actual_gap_pct: float, target_date_str: str) -> list:
    """
    Diagnose why the opening gap diverged from the 3:05 PM prediction.
    """
    reasons = []
    try:
        macro_df = get_macro_market_cues(period="1mo")
        if not macro_df.empty and len(macro_df) >= 2:
            latest_macro = macro_df.iloc[-1]
            sp_change = latest_macro.get('SP500_Ret1', 0.0) * 100.0
            if sp_change < -0.8 and "UP" in pred_direction:
                reasons.append(
                    f"🌐 **Overnight US Tech/S&P Selloff**: S&P 500 sank {sp_change:.2f}% after Indian market close, "
                    "forcing an overnight gap-down despite strong domestic close."
                )
            elif sp_change > 0.8 and "DOWN" in pred_direction:
                reasons.append(
                    f"🌐 **Overnight Global Rally Surprise**: US markets gained +{sp_change:.2f}% overnight, "
                    "prompting strong gap-up opening momentum across Asian equities."
                )

            vix_change = latest_macro.get('VIX_Ret1', 0.0) * 100.0
            if vix_change > 5.0 and "UP" in pred_direction:
                reasons.append(
                    f"🌋 **Overnight Fear / Volatility Spike**: India VIX spiked +{vix_change:.1f}%, causing risk-off opening selling."
                )
    except Exception as e:
        logging.error(f"Error diagnosing gap divergence: {e}")

    if not reasons:
        if "UP" in pred_direction and actual_gap_pct < 0:
            reasons.append("📌 **Overnight News / Sentiment Reversal**: Overnight global/macro cues diluted bullish carry momentum before 9:15 AM.")
        elif "DOWN" in pred_direction and actual_gap_pct > 0:
            reasons.append("📌 **Pre-Market Block Order / Gap Surprise**: Positive institutional pre-market matching overrode late-session selling pressure.")

    return reasons

def evaluate_opening_gap_outcomes():
    """
    Evaluates completed trading sessions for opening gap accuracy.
    Compares next day's 9:15 AM Open against previous day's Close.
    """
    history = load_saved_gap_audit_history()
    if not history:
        return []

    updated = False
    today = dt.date.today()

    for record in history:
        target_date_obj = parse_target_date(record.get("target_date"))

        # Evaluate if target trading date has arrived
        if target_date_obj <= today:
            symbol = record.get("symbol")
            df_stock = get_stock_data(symbol, period="1mo")

            if not df_stock.empty and len(df_stock) >= 2:
                df_stock_dates = pd.to_datetime(df_stock.index).date
                matching_indices = [idx for idx, d in enumerate(df_stock_dates) if d == target_date_obj]

                if matching_indices:
                    target_idx = matching_indices[-1]
                    target_bar = df_stock.iloc[target_idx]
                    prev_bar = df_stock.iloc[target_idx - 1] if target_idx > 0 else target_bar

                    actual_open = round(float(target_bar['Open']), 2)
                    baseline = float(record.get("baseline_3pm_close") or prev_bar['Close'])

                    gap_rs = round(actual_open - baseline, 2)
                    gap_pct = round((gap_rs / baseline) * 100.0, 2) if baseline > 0 else 0.0

                    if gap_rs > 0:
                        actual_gap_dir = "GAP UP 📈"
                    elif gap_rs < 0:
                        actual_gap_dir = "GAP DOWN 📉"
                    else:
                        actual_gap_dir = "FLAT ⚪"

                    pred_dir = record.get("predicted_gap_direction", "")
                    is_correct_bool = bool(
                        ("UP" in pred_dir and gap_rs > 0) or
                        ("DOWN" in pred_dir and gap_rs < 0)
                    )

                    record["actual_915_open"] = actual_open
                    record["actual_gap_rs"] = gap_rs
                    record["actual_gap_pct"] = gap_pct
                    record["actual_gap_direction"] = actual_gap_dir
                    record["is_correct"] = is_correct_bool

                    if is_correct_bool:
                        record["divergence_reasons"] = [
                            f"✅ **Opening Gap Verified**: Stock opened at ₹{actual_open:,.2f} ({gap_pct:+.2f}% gap) matching AI prediction."
                        ]
                    else:
                        record["divergence_reasons"] = diagnose_gap_divergence(
                            symbol, pred_dir, gap_pct, record.get("target_date")
                        )

                    updated = True
        else:
            record["actual_gap_direction"] = "⏳ Pending Next Market Open (9:15 AM)"
            record["is_correct"] = None
            record["divergence_reasons"] = ["⏳ Opening gap outcome will be evaluated after market opens tomorrow at 9:15 AM."]
            updated = True

    if updated:
        save_gap_audit_history(history)

    return history

def calculate_gap_recalibration_offset(symbol: str) -> tuple:
    """
    Calculate dynamic recalibration offset from past opening gap predictions.
    """
    history = load_saved_gap_audit_history()
    if not history:
        return 0.0, "Neutral baseline (No past opening gap audit records)"

    symbol_entries = [e for e in history if e.get("symbol") == symbol and e.get("is_correct") is not None]
    if not symbol_entries:
        return 0.0, "Neutral baseline (First opening gap prediction for this stock)"

    recent = symbol_entries[-10:]
    diverged = [e for e in recent if not e.get("is_correct")]

    if not diverged:
        return 0.0, "✅ 100% Recent Opening Gap Accuracy"

    gap_up_misses = sum(1 for e in diverged if 'DOWN' in e.get('predicted_gap_direction', '') and (e.get('actual_gap_pct', 0) or 0) > 0)
    gap_down_misses = sum(1 for e in diverged if 'UP' in e.get('predicted_gap_direction', '') and (e.get('actual_gap_pct', 0) or 0) < 0)

    net_bias = gap_up_misses - gap_down_misses
    offset = float(max(-10.0, min(10.0, net_bias * 3.5)))

    if offset > 0:
        reason = f"🟢 Historical Gap Bias (+{offset:.1f}%): AI previously understated opening gap-up support in {gap_up_misses} session(s)."
    elif offset < 0:
        reason = f"🔴 Historical Gap Bias ({offset:.1f}%): AI previously understated overnight gap-down selling in {gap_down_misses} session(s)."
    else:
        reason = "⚪ Balanced Historical Gap Error Rate."

    return offset, reason
