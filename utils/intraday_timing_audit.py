import os
import json
import logging
import datetime as dt
import pandas as pd

logging.basicConfig(level=logging.INFO)

TIMING_AUDIT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "intraday_timing_audit.json")
PREDICTION_AUDIT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prediction_audit.json")

def parse_target_date(date_str: str) -> dt.date | None:
    """Parse string formatted date into datetime.date object."""
    if not date_str:
        return None
    try:
        return dt.datetime.strptime(date_str, "%a, %d %b %Y").date()
    except Exception:
        try:
            return dt.datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            return None

def load_intraday_timing_audit() -> list:
    """Load intraday entry timing comparison records from JSON file."""
    if os.path.exists(TIMING_AUDIT_FILE):
        try:
            with open(TIMING_AUDIT_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, list):
                    return saved
        except Exception as e:
            logging.error(f"Error loading intraday timing audit: {e}")
    return []

def save_intraday_timing_audit(records: list):
    """Save intraday entry timing comparison records to JSON file."""
    try:
        with open(TIMING_AUDIT_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving intraday timing audit: {e}")

def evaluate_single_timing(day_bars: pd.DataFrame, is_up: bool, h: int, m: int, target_date_obj: dt.date, is_today: bool, now_time: dt.time) -> dict:
    """
    Evaluates execution outcome for a specific entry window (e.g. 09:20, 09:25, 09:30).
    """
    time_label = f"{h:02d}:{m:02d} AM"
    entry_cutoff = dt.time(h, m)

    # If session is today and the entry cutoff time has not arrived yet:
    if is_today and now_time < entry_cutoff:
        return {
            "entry_time": time_label,
            "entry_price": None,
            "target_price": None,
            "sl_price": None,
            "hit_status": "⏳ Awaiting Entry Time",
            "hit_time": "Pending",
            "hit_price": None,
            "minutes_to_hit": None,
            "points": None,
            "is_win": None
        }

    sub_bars = day_bars[day_bars.index.time >= entry_cutoff]
    if sub_bars.empty:
        return {
            "entry_time": time_label,
            "entry_price": None,
            "target_price": None,
            "sl_price": None,
            "hit_status": "⏳ No Bar Data Yet",
            "hit_time": "Pending",
            "hit_price": None,
            "minutes_to_hit": None,
            "points": None,
            "is_win": None
        }

    entry_bar = sub_bars.iloc[0]
    entry_time_str = entry_bar.name.strftime("%I:%M %p")
    entry_price = round(float(entry_bar["Open"]), 2)
    step_tgt = max(round(entry_price * 0.006, 2), 8.0)
    step_sl = max(round(entry_price * 0.005, 2), 7.0)
    tgt = round(entry_price + step_tgt if is_up else entry_price - step_tgt, 2)
    sl = round(entry_price - step_sl if is_up else entry_price + step_sl, 2)

    hit_status = None
    hit_time = None
    hit_price = None
    mins = None
    is_win = None

    entry_dt = entry_bar.name

    for idx, bar in sub_bars.iterrows():
        b_time = idx.strftime("%I:%M %p")
        hi = float(bar["High"])
        lo = float(bar["Low"])
        elapsed = max(int((idx - entry_dt).total_seconds() / 60.0), 0)

        if is_up:
            if hi >= tgt:
                hit_status = "🎯 Target Hit (+₹4.50 Opt)"
                hit_time = b_time
                hit_price = tgt
                mins = elapsed
                is_win = True
                break
            elif lo <= sl:
                hit_status = "🛑 Stop Loss Hit (-₹3.80 Opt)"
                hit_time = b_time
                hit_price = sl
                mins = elapsed
                is_win = False
                break
        else:
            if lo <= tgt:
                hit_status = "🎯 Target Hit (+₹4.50 Opt)"
                hit_time = b_time
                hit_price = tgt
                mins = elapsed
                is_win = True
                break
            elif hi >= sl:
                hit_status = "🛑 Stop Loss Hit (-₹3.80 Opt)"
                hit_time = b_time
                hit_price = sl
                mins = elapsed
                is_win = False
                break

    if not hit_status:
        last_close = round(float(day_bars.iloc[-1]["Close"]), 2)
        if is_today and now_time < dt.time(15, 30):
            hit_status = "⏳ Trade In Progress"
            hit_time = "Live"
            hit_price = last_close
            mins = max(int((day_bars.index[-1] - entry_dt).total_seconds() / 60.0), 0)
            pts = round((last_close - entry_price) if is_up else (entry_price - last_close), 2)
            is_win = None
        else:
            hit_status = "⏱️ Held to Close"
            hit_time = "03:15 PM"
            hit_price = last_close
            mins = max(int((day_bars.index[-1] - entry_dt).total_seconds() / 60.0), 0)
            pts = round((last_close - entry_price) if is_up else (entry_price - last_close), 2)
            is_win = pts > 0
    else:
        pts = round((tgt - entry_price) if is_win else (sl - entry_price), 2)
        if not is_up and is_win is not None:
            pts = round(-pts, 2) if is_win else round(abs(entry_price - sl) * -1, 2)

    return {
        "entry_time": entry_time_str,
        "entry_price": entry_price,
        "target_price": tgt,
        "sl_price": sl,
        "hit_status": hit_status,
        "hit_time": hit_time,
        "hit_price": hit_price,
        "minutes_to_hit": mins,
        "points": pts,
        "is_win": is_win
    }

def sync_intraday_timing_audit(symbol: str = "CDSL.NS") -> list:
    """
    Syncs the intraday timing comparison records across all sessions for the given symbol.
    Fetches 5m candlestick data and evaluates 09:20, 09:25, and 09:30 AM entries.
    Automatically updates and persists to intraday_timing_audit.json.
    """
    try:
        import yfinance as yf
    except Exception as e:
        logging.error(f"yfinance not available: {e}")
        return load_intraday_timing_audit()

    if not os.path.exists(PREDICTION_AUDIT_FILE):
        return []

    try:
        with open(PREDICTION_AUDIT_FILE, "r", encoding="utf-8") as f:
            pred_history = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read prediction audit: {e}")
        return load_intraday_timing_audit()

    existing_timing_records = load_intraday_timing_audit()
    timing_map = {r.get("target_date"): r for r in existing_timing_records if r.get("symbol") == symbol}

    # Fetch 5m intraday bars
    try:
        df_5m = yf.Ticker(symbol).history(period="60d", interval="5m")
        if df_5m.index.tz is not None:
            df_5m.index = df_5m.index.tz_convert("Asia/Kolkata")
    except Exception as e:
        logging.warning(f"Error fetching 5m data for {symbol}: {e}")
        df_5m = pd.DataFrame()

    today = dt.date.today()
    now_time = dt.datetime.now().time()
    updated_records = []

    # Filter pred_history for the symbol
    symbol_sessions = [r for r in pred_history if r.get("symbol") == symbol]

    for rec in symbol_sessions:
        target_date_str = rec.get("target_date")
        target_date_obj = parse_target_date(target_date_str)
        pred_dir = rec.get("predicted_direction", "UP 📈")
        is_up = "UP" in pred_dir
        conf = rec.get("confidence", "High Confidence")

        # Future date guard
        if not target_date_obj or target_date_obj > today:
            updated_records.append({
                "symbol": symbol,
                "target_date": target_date_str,
                "predicted_direction": pred_dir,
                "confidence": conf,
                "entries": {
                    "9:20": {
                        "entry_time": "09:20 AM",
                        "entry_price": None,
                        "target_price": None,
                        "sl_price": None,
                        "hit_status": "⏳ Pending Next Session Open (9:20 AM)",
                        "hit_time": "Pending",
                        "hit_price": None,
                        "minutes_to_hit": None,
                        "points": None,
                        "is_win": None
                    },
                    "9:25": {
                        "entry_time": "09:25 AM",
                        "entry_price": None,
                        "target_price": None,
                        "sl_price": None,
                        "hit_status": "⏳ Pending Next Session Open (9:25 AM)",
                        "hit_time": "Pending",
                        "hit_price": None,
                        "minutes_to_hit": None,
                        "points": None,
                        "is_win": None
                    },
                    "9:30": {
                        "entry_time": "09:30 AM",
                        "entry_price": None,
                        "target_price": None,
                        "sl_price": None,
                        "hit_status": "⏳ Pending Next Session Open (9:30 AM)",
                        "hit_time": "Pending",
                        "hit_price": None,
                        "minutes_to_hit": None,
                        "points": None,
                        "is_win": None
                    }
                },
                "best_entry": "⏳ Pending Session Open"
            })
            continue

        is_today = (target_date_obj == today)

        # Check if already fully evaluated past session
        cached = timing_map.get(target_date_str)
        if cached and not is_today:
            cached_entries = cached.get("entries", {})
            if (
                cached_entries.get("9:20", {}).get("is_win") is not None and
                cached_entries.get("9:25", {}).get("is_win") is not None and
                cached_entries.get("9:30", {}).get("is_win") is not None
            ):
                updated_records.append(cached)
                continue

        # Evaluate from df_5m
        if df_5m.empty:
            if cached:
                updated_records.append(cached)
            continue

        day_bars = df_5m[df_5m.index.date == target_date_obj]
        if day_bars.empty:
            if cached:
                updated_records.append(cached)
            continue

        e_920 = evaluate_single_timing(day_bars, is_up, 9, 20, target_date_obj, is_today, now_time)
        e_925 = evaluate_single_timing(day_bars, is_up, 9, 25, target_date_obj, is_today, now_time)
        e_930 = evaluate_single_timing(day_bars, is_up, 9, 30, target_date_obj, is_today, now_time)

        # Determine best entry
        # Prefer target hit, then fewest minutes to hit, then lowest entry price (or highest for puts)
        valid_hits = [
            ("09:20 AM", e_920),
            ("09:25 AM", e_925),
            ("09:30 AM", e_930)
        ]
        winners = [t for t in valid_hits if t[1].get("is_win") is True]
        if winners:
            winners.sort(key=lambda x: x[1].get("minutes_to_hit", 999))
            fastest = winners[0]
            best_entry_label = f"🏆 {fastest[0]} ({fastest[1].get('minutes_to_hit', 0)}m to Target)"
        elif any(t[1].get("is_win") is False for t in valid_hits):
            best_entry_label = "❌ Stop Loss across entries"
        else:
            best_entry_label = "⏳ In Progress"

        updated_records.append({
            "symbol": symbol,
            "target_date": target_date_str,
            "predicted_direction": pred_dir,
            "confidence": conf,
            "entries": {
                "9:20": e_920,
                "9:25": e_925,
                "9:30": e_930
            },
            "best_entry": best_entry_label
        })

    # Preserve any other symbols in the audit file
    other_records = [r for r in existing_timing_records if r.get("symbol") != symbol]
    all_records = other_records + updated_records

    save_intraday_timing_audit(all_records)
    return updated_records

def get_timing_kpis(records: list) -> dict:
    """
    Computes summary KPI statistics for 09:20, 09:25, and 09:30 entries across evaluated sessions.
    """
    stats = {
        "9:20": {"wins": 0, "losses": 0, "total": 0, "minutes": [], "points": 0.0, "fastest": 0},
        "9:25": {"wins": 0, "losses": 0, "total": 0, "minutes": [], "points": 0.0, "fastest": 0},
        "9:30": {"wins": 0, "losses": 0, "total": 0, "minutes": [], "points": 0.0, "fastest": 0}
    }

    evaluated_sessions = 0

    for r in records:
        entries = r.get("entries", {})
        # Only count if evaluated
        is_eval = False
        for k in ["9:20", "9:25", "9:30"]:
            entry_data = entries.get(k, {})
            is_win = entry_data.get("is_win")
            if is_win is not None:
                is_eval = True
                stats[k]["total"] += 1
                if is_win:
                    stats[k]["wins"] += 1
                    m = entry_data.get("minutes_to_hit")
                    if m is not None:
                        stats[k]["minutes"].append(m)
                        if m <= 10:
                            stats[k]["fastest"] += 1
                else:
                    stats[k]["losses"] += 1
                pts = entry_data.get("points")
                if pts is not None:
                    stats[k]["points"] += pts
        if is_eval:
            evaluated_sessions += 1

    summary = {}
    for k, title in [("9:20", "09:20 AM Entry"), ("9:25", "09:25 AM Entry"), ("9:30", "09:30 AM Entry")]:
        tot = stats[k]["total"]
        w = stats[k]["wins"]
        l = stats[k]["losses"]
        wr = round((w / tot * 100.0), 1) if tot > 0 else 0.0
        avg_m = round(sum(stats[k]["minutes"]) / len(stats[k]["minutes"]), 1) if stats[k]["minutes"] else 0.0
        summary[k] = {
            "title": title,
            "total": tot,
            "wins": w,
            "losses": l,
            "win_rate_pct": wr,
            "avg_minutes": avg_m,
            "fastest_hits": stats[k]["fastest"],
            "total_points": round(stats[k]["points"], 2)
        }

    return {
        "evaluated_sessions": evaluated_sessions,
        "details": summary
    }
