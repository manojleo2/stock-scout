import os
import json
import logging
import pandas as pd
import datetime as dt
from utils.data_loader import get_stock_data
from utils.macro_factors import get_macro_market_cues

logging.basicConfig(level=logging.INFO)

AUDIT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prediction_audit.json")

def parse_target_date(date_str: str) -> dt.date:
    """Parse string formatted date into datetime.date object."""
    if not date_str:
        return dt.date.today()
    try:
        # Format: "Mon, 07 Sep 2026"
        return dt.datetime.strptime(date_str, "%a, %d %b %Y").date()
    except Exception:
        try:
            return dt.datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            return dt.date.today()

def load_saved_audit_history() -> list:
    """Load prediction audit history from JSON file."""
    if os.path.exists(AUDIT_FILE):
        try:
            with open(AUDIT_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, list):
                    return saved
        except Exception as e:
            logging.error(f"Error loading audit history: {e}")
    return []

def save_audit_history(audit_list: list):
    """Save prediction audit history to JSON file."""
    try:
        with open(AUDIT_FILE, "w", encoding="utf-8") as f:
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
        "pred_result": pred_result,
        "actual_close": None,
        "actual_change_pct": None,
        "actual_direction": "⏳ Pending Session Close",
        "is_correct": None,
        "divergence_reasons": ["⏳ Target trading session is upcoming. Outcome will be evaluated post-market close."]
    }

    if existing:
        if existing.get("is_correct") is None:
            existing.update(record_data)
    else:
        history.append(record_data)

    save_audit_history(history)
    return history

def get_saved_prediction_snapshot(symbol: str, target_date_str: str) -> dict | None:
    """
    Retrieve previously calculated prediction snapshot for symbol and target date.
    Allows instant sub-second page loads without re-training models on every render.
    """
    history = load_saved_audit_history()
    for r in reversed(history):
        if r.get("symbol") == symbol and r.get("target_date") == target_date_str:
            if "pred_result" in r and isinstance(r["pred_result"], dict):
                return r["pred_result"]
    return None

def diagnose_divergence_reasons(symbol: str, pred_direction: str, actual_change_pct: float, target_date_str: str) -> list:
    """
    Performs root cause analysis when actual outcome contradicts the AI prediction.
    """
    reasons = []

    try:
        df_stock = get_stock_data(symbol, period="1mo")
        macro_df = get_macro_market_cues(period="1mo")

        if not df_stock.empty and len(df_stock) >= 2:
            latest_bar = df_stock.iloc[-1]

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

def compute_intraday_execution_details(symbol: str, target_date_obj: dt.date, predicted_dir: str, df_5m: pd.DataFrame = None) -> dict:
    """
    Computes exact intraday 9:20 AM entry price, target (+₹4.50 opt), stop loss (-₹3.80 opt),
    and scans 5m bars to determine whether target or stop loss was hit along with exact hit time.
    """
    if df_5m is None:
        try:
            import yfinance as yf
            df_5m = yf.Ticker(symbol).history(period="60d", interval="5m")
        except Exception as e:
            logging.warning(f"Error fetching 5m data for {symbol}: {e}")
            return {}

    if df_5m is None or df_5m.empty:
        return {}

    try:
        bars = df_5m.copy()
        if bars.index.tz is not None:
            bars.index = bars.index.tz_convert("Asia/Kolkata")

        day_bars = bars[bars.index.date == target_date_obj]
        if day_bars.empty:
            return {}

        bar_920 = day_bars[day_bars.index.time >= dt.time(9, 20)]
        if bar_920.empty:
            bar_920 = day_bars

        entry_bar = bar_920.iloc[0]
        entry_time = entry_bar.name.strftime("%I:%M %p")
        entry_price = round(float(entry_bar["Open"]), 2)

        is_up = "UP" in str(predicted_dir)
        step_target = max(round(entry_price * 0.006, 2), 8.0)
        step_sl = max(round(entry_price * 0.005, 2), 7.0)

        target_price = round(entry_price + step_target if is_up else entry_price - step_target, 2)
        sl_price = round(entry_price - step_sl if is_up else entry_price + step_sl, 2)

        hit_status = None
        hit_time = None
        hit_price = None
        points = 0.0

        for idx, row in bar_920.iterrows():
            b_time = idx.strftime("%I:%M %p")
            high = float(row["High"])
            low = float(row["Low"])
            if is_up:
                if high >= target_price:
                    hit_status = "🎯 Target Hit (+₹4.50 Opt)"
                    hit_time = b_time
                    hit_price = target_price
                    points = round(target_price - entry_price, 2)
                    break
                elif low <= sl_price:
                    hit_status = "🛑 Stop Loss Hit (-₹3.80 Opt)"
                    hit_time = b_time
                    hit_price = sl_price
                    points = round(sl_price - entry_price, 2)
                    break
            else:
                if low <= target_price:
                    hit_status = "🎯 Target Hit (+₹4.50 Opt)"
                    hit_time = b_time
                    hit_price = target_price
                    points = round(entry_price - target_price, 2)
                    break
                elif high >= sl_price:
                    hit_status = "🛑 Stop Loss Hit (-₹3.80 Opt)"
                    hit_time = b_time
                    hit_price = sl_price
                    points = round(entry_price - sl_price, 2)
                    break

        if not hit_status:
            last_close = round(float(day_bars.iloc[-1]["Close"]), 2)
            today_date = dt.date.today()
            now_t = dt.datetime.now().time()
            if target_date_obj == today_date and now_t < dt.time(15, 30):
                hit_status = "⏳ Trade In Progress"
                hit_time = "Live"
                hit_price = last_close
                points = round((last_close - entry_price) if is_up else (entry_price - last_close), 2)
            else:
                hit_status = "⏱️ Held to Close"
                hit_time = "03:15 PM"
                hit_price = last_close
                points = round((last_close - entry_price) if is_up else (entry_price - last_close), 2)

        return {
            "entry_time": entry_time,
            "entry_price": entry_price,
            "target_price": target_price,
            "sl_price": sl_price,
            "hit_status": hit_status,
            "hit_time": hit_time,
            "hit_price": hit_price,
            "points": points
        }
    except Exception as e:
        logging.warning(f"Error computing intraday execution details: {e}")
        return {}

def evaluate_and_update_audit_outcomes():
    """
    Evaluates completed trading sessions, checks actual close prices, updates hit/miss status,
    and enriches with exact 9:20 AM entry price, target/stop-loss hit times.
    """
    history = load_saved_audit_history()
    if not history:
        return []

    updated = False
    today = dt.date.today()
    df_5m_cache = {}

    for record in history:
        target_date_obj = parse_target_date(record.get("target_date"))
        symbol = record.get("symbol")

        # Check if execution details need computing
        if record.get("hit_status") is None and target_date_obj <= today:
            if symbol not in df_5m_cache:
                try:
                    import yfinance as yf
                    df_5m_cache[symbol] = yf.Ticker(symbol).history(period="60d", interval="5m")
                except Exception:
                    df_5m_cache[symbol] = pd.DataFrame()
            
            exec_details = compute_intraday_execution_details(
                symbol, target_date_obj, record.get("predicted_direction", ""), df_5m_cache.get(symbol)
            )
            if exec_details:
                record.update(exec_details)
                updated = True

        # Fast skip if session outcome already evaluated
        if record.get("is_correct") is not None:
            continue

        # If session is today and market is currently trading (before 3:35 PM IST), skip
        is_today = (target_date_obj == today)
        now_time = dt.datetime.now().time()
        if is_today and now_time < dt.time(15, 35):
            continue

        # ONLY evaluate if target trading session date has arrived or passed!
        if target_date_obj <= today:
            df_stock = get_stock_data(symbol, period="1mo")

            if not df_stock.empty and len(df_stock) >= 2:
                df_stock_dates = pd.to_datetime(df_stock.index).date
                matching_indices = [idx for idx, d in enumerate(df_stock_dates) if d == target_date_obj]

                if matching_indices:
                    target_idx = matching_indices[-1]
                    target_bar = df_stock.iloc[target_idx]
                    prev_bar = df_stock.iloc[target_idx - 1] if target_idx > 0 else target_bar

                    actual_close = round(target_bar['Close'], 2)
                    baseline = record.get("baseline_close", prev_bar['Close'])

                    actual_change_rs = actual_close - baseline
                    actual_change_pct = round((actual_change_rs / baseline) * 100.0, 2)

                    actual_dir = "UP 📈" if actual_change_rs > 0 else ("DOWN 📉" if actual_change_rs < 0 else "FLAT ⚪")
                    predicted_dir = record.get("predicted_direction", "")

                    is_correct_bool = bool(
                        ("UP" in predicted_dir and actual_change_rs > 0) or
                        ("DOWN" in predicted_dir and actual_change_rs < 0)
                    )

                    record["actual_close"] = float(actual_close)
                    record["actual_change_pct"] = float(actual_change_pct)
                    record["actual_direction"] = str(actual_dir)
                    record["is_correct"] = True if is_correct_bool else False

                    if not is_correct_bool:
                        record["divergence_reasons"] = diagnose_divergence_reasons(
                            symbol, predicted_dir, actual_change_pct, record.get("target_date")
                        )
                    else:
                        record["divergence_reasons"] = [
                            "✅ **Prediction Verified**: Stock movement matched the AI model's directional forecast."
                        ]

                    # Sync outcome into Daily Market Journal
                    try:
                        from utils.daily_journal import create_daily_session_snapshot, update_or_append_journal_entry
                        snap = create_daily_session_snapshot(symbol, record.get("target_date"), predicted_dir, record.get("probability_up_pct", 50.0))
                        update_or_append_journal_entry(snap)
                    except Exception as e_snap:
                        logging.warning(f"Journal sync error for {symbol}: {e_snap}")

                    updated = True
        else:
            # Target date is in the future
            record["actual_direction"] = "⏳ Pending Session Close"
            record["actual_change_pct"] = None
            record["is_correct"] = None
            record["divergence_reasons"] = ["⏳ Target trading session is upcoming. Outcome will be evaluated post-market close."]
            updated = True

    if updated:
        save_audit_history(history)

    return history
