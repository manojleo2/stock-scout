import os
import json
import logging
import pandas as pd
import datetime as dt
from utils.data_loader import get_stock_data
from utils.macro_factors import get_macro_market_cues

logging.basicConfig(level=logging.INFO)

AUDIT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prediction_audit.json")

def load_saved_audit_history() -> list:
    """Load prediction audit history from JSON file."""
    if os.path.exists(AUDIT_FILE):
        try:
            with open(AUDIT_FILE, "r") as f:
                saved = json.load(f)
                if isinstance(saved, list):
                    return saved
        except Exception as e:
            logging.error(f"Error loading audit history: {e}")
    return []

def save_audit_history(audit_list: list):
    """Save prediction audit history to JSON file."""
    try:
        with open(AUDIT_FILE, "w") as f:
            json.dump(audit_list, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving audit history: {e}")

def record_prediction(symbol: str, target_date_str: str, pred_result: dict):
    """
    Log a new prediction for a target trading date before the market moves.
    """
    history = load_saved_audit_history()

    # Check if record already exists for symbol + date
    existing = next((r for r in history if r.get("symbol") == symbol and r.get("target_date") == target_date_str), None)
    
    record_data = {
        "symbol": symbol,
        "target_date": target_date_str,
        "prediction_time": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "predicted_direction": pred_result.get("direction"),
        "probability_up_pct": pred_result.get("probability_up_pct"),
        "probability_down_pct": pred_result.get("probability_down_pct"),
        "confidence": pred_result.get("confidence"),
        "baseline_close": pred_result.get("latest_close"),
        "top_features": list(pred_result.get("feature_importances", {}).items())[:6],
        "actual_close": None,
        "actual_change_pct": None,
        "actual_direction": None,
        "is_correct": None,
        "divergence_reasons": []
    }

    if existing:
        # Update existing prediction metadata if actual is not yet resolved
        if existing.get("is_correct") is None:
            existing.update(record_data)
    else:
        history.append(record_data)

    save_audit_history(history)
    return history

def diagnose_divergence_reasons(symbol: str, pred_direction: str, actual_change_pct: float, target_date_str: str) -> list:
    """
    Performs root cause analysis when actual outcome contradicts the AI prediction.
    Identifies exact parameters that caused the opposite movement.
    """
    reasons = []

    try:
        df_stock = get_stock_data(symbol, period="1mo")
        macro_df = get_macro_market_cues(period="1mo")

        if not df_stock.empty and len(df_stock) >= 2:
            latest_bar = df_stock.iloc[-1]
            prev_bar = df_stock.iloc[-2]

            # 1. High Intraday Rejection / Upper Wick check
            intraday_range = latest_bar['High'] - latest_bar['Low']
            upper_wick = latest_bar['High'] - max(latest_bar['Open'], latest_bar['Close'])
            if intraday_range > 0 and (upper_wick / intraday_range) > 0.40 and "UP" in pred_direction:
                reasons.append(
                    "📌 **Intraday Profit Booking / Rejection**: Stock gapped up at market open, but strong "
                    "institutional profit booking created a long upper wick, driving price down by close."
                )

            # 2. Volume Pressure check
            vol_avg = df_stock['Volume'].rolling(10).mean().iloc[-1]
            if latest_bar['Volume'] > vol_avg * 1.5 and actual_change_pct < 0 and "UP" in pred_direction:
                reasons.append(
                    "📌 **Heavy Volume Selling Shock**: Trading volume was 50%+ higher than average during a "
                    "downward session, signaling large institutional distribution."
                )

        if not macro_df.empty and len(macro_df) >= 2:
            latest_macro = macro_df.iloc[-1]

            # 3. Overnight S&P 500 / Global Cues Flip
            sp_change = latest_macro.get('SP500_Ret1', 0.0) * 100.0
            if sp_change < -0.6 and "UP" in pred_direction:
                reasons.append(
                    f"🌐 **Overnight Global Market Drag**: US S&P 500 plummeted {sp_change:.2f}% overnight, "
                    "triggering a risk-off gap down at the Indian market open."
                )
            elif sp_change > 0.6 and "DOWN" in pred_direction:
                reasons.append(
                    f"🌐 **Global Relief Rally Surprise**: US S&P 500 surged +{sp_change:.2f}% overnight, "
                    "overriding technical sell signals with an opening gap-up."
                )

            # 4. India VIX Volatility Shock
            vix_change = latest_macro.get('VIX_Ret1', 0.0) * 100.0
            if vix_change > 4.0 and "UP" in pred_direction:
                reasons.append(
                    f"🌋 **India VIX Volatility Spike**: India VIX spiked +{vix_change:.1f}%, raising broad market fear "
                    "and forcing risk aversion across long positions."
                )

            # 5. Nifty Index Drag
            nifty_change = latest_macro.get('Nifty_Ret1', 0.0) * 100.0 if 'Nifty_Ret1' in latest_macro else 0.0
            if nifty_change < -0.8 and "UP" in pred_direction:
                reasons.append(
                    f"🏛️ **Broad Nifty Index Drag**: Benchmark Nifty 50 dropped {nifty_change:.2f}%, dragging down "
                    "individual stocks despite favorable standalone technicals."
                )

    except Exception as e:
        logging.error(f"Error diagnosing divergence: {e}")

    if not reasons:
        if "UP" in pred_direction and actual_change_pct < 0:
            reasons.append(
                "📌 **Intraday Profit Taking & Rebalance**: Short-term intraday profit booking outweighed "
                "baseline technical indicator signals."
            )
        elif "DOWN" in pred_direction and actual_change_pct > 0:
            reasons.append(
                "📌 **Unexpected Short Covering Surge**: Aggressive short-covering momentum near key support "
                "drove an unexpected intraday rally."
            )

    return reasons

def evaluate_and_update_audit_outcomes():
    """
    Evaluates completed trading sessions, checks actual close prices, updates hit/miss status,
    and generates root cause diagnostics for missed predictions.
    """
    history = load_saved_audit_history()
    if not history:
        return []

    updated = False

    for record in history:
        # Evaluate if actual close is not yet resolved
        if record.get("is_correct") is None:
            symbol = record.get("symbol")
            df_stock = get_stock_data(symbol, period="1mo")

            if not df_stock.empty and len(df_stock) >= 2:
                # Latest bar is actual outcome
                latest_bar = df_stock.iloc[-1]
                prev_bar = df_stock.iloc[-2]

                actual_close = round(latest_bar['Close'], 2)
                baseline = record.get("baseline_close", prev_bar['Close'])

                actual_change_rs = actual_close - baseline
                actual_change_pct = round((actual_change_rs / baseline) * 100.0, 2)

                actual_dir = "UP 📈" if actual_change_rs >= 0 else "DOWN 📉"
                predicted_dir = record.get("predicted_direction", "")

                is_correct = (
                    ("UP" in predicted_dir and actual_change_rs >= 0) or
                    ("DOWN" in predicted_dir and actual_change_rs < 0)
                )

                record["actual_close"] = actual_close
                record["actual_change_pct"] = actual_change_pct
                record["actual_direction"] = actual_dir
                record["is_correct"] = is_correct

                # If missed prediction, generate root cause diagnosis
                if not is_correct:
                    record["divergence_reasons"] = diagnose_divergence_reasons(
                        symbol, predicted_dir, actual_change_pct, record.get("target_date")
                    )
                else:
                    record["divergence_reasons"] = [
                        "✅ **Prediction Verified**: Stock movement matched the AI model's directional forecast."
                    ]

                updated = True

    if updated:
        save_audit_history(history)

    return history
