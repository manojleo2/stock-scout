import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
import json

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.paper_trading import calculate_bsm_option_price
from utils.opening_predictor import get_days_to_monthly_expiry

def load_genuine_model_predictions():
    """Load genuine walk-forward model predictions from backtest CSV and prediction audit."""
    predictions = {}

    # 1. Walk-Forward Backtest Predictions
    spec_csv = os.path.join(PROJECT_ROOT, "backtest_results", "cdsl_specialist_backtest_20260918_190321.csv")
    if os.path.exists(spec_csv):
        df_spec = pd.read_csv(spec_csv)
        for _, row in df_spec.iterrows():
            d_str = str(row['date']).strip()
            prob = float(row['prob_spec'])
            conv = float(row['conv_spec'])
            trade = bool(row['trade_spec'])
            is_call = prob >= 0.50
            predictions[d_str] = {
                "trade": trade,
                "is_call": is_call,
                "conviction_pct": round(conv * 100.0, 1),
                "prob_up_pct": round(prob * 100.0, 1)
            }

    # 2. Real-Time Prediction Audit Ledger for Recent Days
    audit_file = os.path.join(PROJECT_ROOT, "prediction_audit.json")
    if os.path.exists(audit_file):
        with open(audit_file, "r", encoding="utf-8") as f:
            audit_data = json.load(f)
        for r in audit_data:
            if r.get("symbol") == "CDSL.NS":
                t_str = r.get("target_date")
                try:
                    d_obj = dt.datetime.strptime(t_str, "%a, %d %b %Y").date()
                    d_iso = d_obj.strftime("%Y-%m-%d")
                    pred_dir = r.get("predicted_direction", "")
                    is_call = "UP" in pred_dir
                    p_up = r.get("probability_up_pct", 50.0)
                    conv = max(p_up, 100.0 - p_up)
                    # Friday threshold 70%, weekday 65%
                    is_fri = (d_obj.weekday() == 4)
                    trade = conv >= (70.0 if is_fri else 65.0)
                    predictions[d_iso] = {
                        "trade": trade,
                        "is_call": is_call,
                        "conviction_pct": round(conv, 1),
                        "prob_up_pct": round(p_up, 1)
                    }
                except Exception:
                    pass

    return predictions

def run_intraday_backtest_with_genuine_model(interval="5m", period="60d"):
    print(f"Fetching CDSL.NS intraday data ({interval}, {period})...")
    ticker = yf.Ticker("CDSL.NS")
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        print("No intraday data returned.")
        return

    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata")
    
    unique_days = sorted(list(set(df.index.date)))
    print(f"Total trading days loaded: {len(unique_days)} ({unique_days[0]} to {unique_days[-1]})")

    # EXACT USER POSITION SIZING
    LOT_QTY = 475
    NUM_LOTS = 2
    TOTAL_QTY = LOT_QTY * NUM_LOTS  # 950 Shares
    TARGET_OPT_DELTA = 10.0         # +Rs 10.00 Option Premium
    SL_OPT_DELTA = 5.0              # -Rs 5.00 Option Premium
    BROKERAGE = 100.0               # Rs 100 per executed trade

    model_preds = load_genuine_model_predictions()

    # Daily close reference for gap calculation
    daily_df = ticker.history(period="1y", interval="1d")
    if daily_df.index.tz is not None:
        daily_df.index = daily_df.index.tz_convert("Asia/Kolkata")
    daily_close_map = {idx.date(): row['Close'] for idx, row in daily_df.iterrows()}

    trade_records = []

    for t_date in unique_days:
        d_str = t_date.strftime("%Y-%m-%d")
        day_bars = df[df.index.date == t_date]
        if len(day_bars) < 10:
            continue

        prev_dates = [d for d in daily_close_map.keys() if d < t_date]
        if not prev_dates:
            continue
        prev_close = daily_close_map[max(prev_dates)]

        day_open = day_bars.iloc[0]['Open']
        gap_points = day_open - prev_close
        is_gap_day = abs(gap_points) > 12.0  # Gap > Rs 12

        # Retrieve Genuine Model Prediction for this specific day
        pred_info = model_preds.get(d_str)
        if not pred_info:
            # Fallback to previous close vs open trend if not in history
            is_call = day_open >= prev_close
            should_trade = True
            conviction = 65.0
        else:
            should_trade = pred_info["trade"]
            is_call = pred_info["is_call"]
            conviction = pred_info["conviction_pct"]

        # Check Gap-Day Entry Rule
        entry_time_target = dt.time(9, 35) if is_gap_day else dt.time(9, 20)
        sub_bars = day_bars[day_bars.index.time >= entry_time_target]
        if sub_bars.empty:
            continue

        entry_bar = sub_bars.iloc[0]
        entry_time_actual = entry_bar.name.strftime("%H:%M")
        entry_spot = float(entry_bar['Open'])

        atm_strike = round(entry_spot / 20.0) * 20.0
        contract_type = "CE" if is_call else "PE"
        contract_str = f"CDSL {int(atm_strike)} {contract_type}"

        # If model did not meet conviction threshold -> NO TRADE
        if not should_trade:
            week_str = f"{t_date.isocalendar()[0]}-W{t_date.isocalendar()[1]:02d}"
            trade_records.append({
                "date": d_str,
                "day_name": t_date.strftime("%A"),
                "week": week_str,
                "is_gap_day": is_gap_day,
                "gap_points": round(gap_points, 2),
                "model_action": "NO_TRADE",
                "contract": contract_str,
                "entry_time": entry_time_actual,
                "entry_spot": entry_spot,
                "entry_prem": 0.0,
                "exit_prem": 0.0,
                "opt_pnl_pt": 0.0,
                "hit_status": "⚪ SIDELINED (Conviction < Threshold)",
                "hit_time": "N/A",
                "net_pnl": 0.0,
                "lots": 0,
                "qty": 0
            })
            continue

        dte = max(float(get_days_to_monthly_expiry(t_date)), 1.0)
        entry_prem = calculate_bsm_option_price(entry_spot, atm_strike, days_to_expiry=dte, is_call=is_call)
        target_prem = entry_prem + TARGET_OPT_DELTA
        sl_prem = max(entry_prem - SL_OPT_DELTA, 1.0)

        hit_status = "⏱️ CUTOFF (03:05 PM)"
        hit_time = "15:05"
        exit_prem = entry_prem
        early_spike_sl = False

        eval_bars = sub_bars[(sub_bars.index.time >= entry_time_target) & (sub_bars.index.time <= dt.time(15, 5))]

        for b_idx, bar in eval_bars.iterrows():
            b_time = b_idx.time()
            b_time_str = b_idx.strftime("%H:%M")
            hi = float(bar['High'])
            lo = float(bar['Low'])

            if not is_call:  # PUT OPTION
                prem_at_lo = calculate_bsm_option_price(lo, atm_strike, days_to_expiry=dte, is_call=False)
                prem_at_hi = calculate_bsm_option_price(hi, atm_strike, days_to_expiry=dte, is_call=False)

                if prem_at_hi <= sl_prem:
                    hit_status = "🛑 STOP LOSS HIT (-₹5.00)"
                    hit_time = b_time_str
                    exit_prem = sl_prem
                    if b_time <= dt.time(9, 45):
                        early_spike_sl = True
                    break

                if prem_at_lo >= target_prem:
                    hit_status = "🎯 TARGET HIT (+₹10.00)"
                    hit_time = b_time_str
                    exit_prem = target_prem
                    break
            else:  # CALL OPTION
                prem_at_hi = calculate_bsm_option_price(hi, atm_strike, days_to_expiry=dte, is_call=True)
                prem_at_lo = calculate_bsm_option_price(lo, atm_strike, days_to_expiry=dte, is_call=True)

                if prem_at_lo <= sl_prem:
                    hit_status = "🛑 STOP LOSS HIT (-₹5.00)"
                    hit_time = b_time_str
                    exit_prem = sl_prem
                    if b_time <= dt.time(9, 45):
                        early_spike_sl = True
                    break

                if prem_at_hi >= target_prem:
                    hit_status = "🎯 TARGET HIT (+₹10.00)"
                    hit_time = b_time_str
                    exit_prem = target_prem
                    break

        if "CUTOFF" in hit_status:
            last_spot = float(eval_bars.iloc[-1]['Close'])
            exit_prem = calculate_bsm_option_price(last_spot, atm_strike, days_to_expiry=dte, is_call=is_call)

        opt_pnl_pt = round(exit_prem - entry_prem, 2)
        gross_pnl = round(opt_pnl_pt * TOTAL_QTY, 2)
        net_pnl = round(gross_pnl - BROKERAGE, 2)

        week_str = f"{t_date.isocalendar()[0]}-W{t_date.isocalendar()[1]:02d}"
        trade_records.append({
            "date": d_str,
            "day_name": t_date.strftime("%A"),
            "week": week_str,
            "is_gap_day": is_gap_day,
            "gap_points": round(gap_points, 2),
            "model_action": f"BUY {contract_type} ({conviction}%)",
            "contract": contract_str,
            "entry_time": entry_time_actual,
            "entry_spot": round(entry_spot, 2),
            "entry_prem": round(entry_prem, 2),
            "exit_prem": round(exit_prem, 2),
            "opt_pnl_pt": opt_pnl_pt,
            "hit_status": hit_status,
            "hit_time": hit_time,
            "net_pnl": net_pnl,
            "lots": NUM_LOTS,
            "qty": TOTAL_QTY
        })

    return pd.DataFrame(trade_records)

if __name__ == "__main__":
    df_trades = run_intraday_backtest_with_genuine_model(interval="5m", period="60d")
    csv_out = os.path.join(PROJECT_ROOT, "backtest_results", "cdsl_intraday_2lots_weekly.csv")
    df_trades.to_csv(csv_out, index=False)
    print(f"Saved full trade ledger to {csv_out}")

    # Weekly Aggregations
    weekly_summary = []
    for week_id, w_df in df_trades.groupby("week"):
        active_trades = w_df[w_df['model_action'] != 'NO_TRADE']
        total_active = len(active_trades)
        wins = len(active_trades[active_trades['hit_status'].str.contains("TARGET")])
        losses = len(active_trades[active_trades['hit_status'].str.contains("STOP LOSS")])
        cutoffs = len(active_trades[active_trades['hit_status'].str.contains("CUTOFF")])
        sidelined = len(w_df[w_df['model_action'] == 'NO_TRADE'])
        week_pnl = w_df['net_pnl'].sum()
        start_d = w_df['date'].iloc[0]
        end_d = w_df['date'].iloc[-1]

        weekly_summary.append({
            "Week": week_id,
            "Date Range": f"{start_d} to {end_d}",
            "Trades": total_active,
            "Wins": wins,
            "Losses": losses,
            "Cutoff": cutoffs,
            "Sidelined": sidelined,
            "Net PnL (Rs)": round(week_pnl, 2)
        })

    df_weekly = pd.DataFrame(weekly_summary)
    weekly_csv = os.path.join(PROJECT_ROOT, "backtest_results", "cdsl_weekly_summary.csv")
    df_weekly.to_csv(weekly_csv, index=False)
    print("\n================== WEEKLY PERFORMANCE BREAKDOWN (2 LOTS / 950 QTY) ==================")
    print(df_weekly.to_string(index=False))
    print(f"\nGRAND TOTAL NET P&L (2 LOTS): Rs {df_trades['net_pnl'].sum():,.2f}")
