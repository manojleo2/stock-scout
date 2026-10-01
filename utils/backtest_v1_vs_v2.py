"""
V2 Rulebook Backtest — CDSL Bearish Intraday
Compares V1 (flat +10/-5 on 2 lots) vs V2 (Tranche entry + Runner)
Uses actual model predictions + 5-min intraday data from Sep 1, 2026
"""
import os, sys, datetime as dt, json
import pandas as pd
import numpy as np
import yfinance as yf

if sys.stdout.encoding != 'utf-8':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.paper_trading import calculate_bsm_option_price
from utils.market_calendar import get_days_to_monthly_expiry

# ─── Constants ───
LOT_QTY = 475
NUM_LOTS = 2
TARGET_OPT = 10.0   # +Rs 10 option premium
SL_OPT = 5.0        # -Rs 5 option premium
BROKERAGE = 100.0
STRIKE_INTERVAL = 20.0

def load_model_predictions():
    """Load genuine model predictions from backtest CSV + prediction audit."""
    preds = {}
    spec_csv = os.path.join(PROJECT_ROOT, "backtest_results", "cdsl_specialist_backtest_20260918_190321.csv")
    if os.path.exists(spec_csv):
        df = pd.read_csv(spec_csv)
        for _, row in df.iterrows():
            d_str = str(row['date']).strip()
            prob = float(row['prob_spec'])
            conv = float(row['conv_spec'])
            trade = bool(row['trade_spec'])
            preds[d_str] = {"trade": trade, "is_call": prob >= 0.50, "conviction_pct": round(conv*100,1), "prob_up_pct": round(prob*100,1)}
    
    audit_file = os.path.join(PROJECT_ROOT, "prediction_audit.json")
    if os.path.exists(audit_file):
        with open(audit_file, "r", encoding="utf-8") as f:
            audit = json.load(f)
        for r in audit:
            if r.get("symbol") == "CDSL.NS":
                try:
                    d_obj = dt.datetime.strptime(r.get("target_date"), "%a, %d %b %Y").date()
                    d_iso = d_obj.strftime("%Y-%m-%d")
                    pred_dir = r.get("predicted_direction", "")
                    is_call = "UP" in pred_dir
                    p_up = r.get("probability_up_pct", 50.0)
                    conv = max(p_up, 100.0 - p_up)
                    is_fri = d_obj.weekday() == 4
                    trade = conv >= (70.0 if is_fri else 65.0)
                    preds[d_iso] = {"trade": trade, "is_call": is_call, "conviction_pct": round(conv,1), "prob_up_pct": round(p_up,1)}
                except: pass
    return preds

def compute_vwap_bands(bars_1m):
    """Compute VWAP and standard deviation bands from 1-min bars."""
    tp = (bars_1m['High'] + bars_1m['Low'] + bars_1m['Close']) / 3.0
    vol = bars_1m['Volume'].astype(float)
    cum_tp_vol = (tp * vol).cumsum()
    cum_vol = vol.cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    # Standard deviation band
    sq_diff = ((tp - vwap) ** 2 * vol).cumsum()
    variance = sq_diff / cum_vol.replace(0, np.nan)
    std = np.sqrt(variance)
    return vwap, std

def is_rejection_candle(bar):
    """V2 mechanical rejection candle: close < open, upper wick >= 1.5x body, close in lower 40%."""
    o, h, l, c = float(bar['Open']), float(bar['High']), float(bar['Low']), float(bar['Close'])
    body = abs(c - o)
    if body < 0.05:
        return False
    if c >= o:  # Must be bearish (close < open)
        return False
    upper_wick = h - o  # For bearish candle: wick = high - open
    candle_range = h - l
    if candle_range < 0.10:
        return False
    close_position = (c - l) / candle_range  # 0 = bottom, 1 = top
    return upper_wick >= 1.5 * body and close_position <= 0.40

def run_v1_v2_backtest():
    print("Fetching CDSL.NS 5-min data (60d)...")
    ticker = yf.Ticker("CDSL.NS")
    df_5m = ticker.history(period="60d", interval="5m")
    if df_5m.index.tz is not None:
        df_5m.index = df_5m.index.tz_convert("Asia/Kolkata")
    
    # Also fetch 1-min data for rejection candle detection (max 7 days from yfinance)
    # For older dates, we'll use 5-min candles as proxy
    print("Fetching CDSL.NS 1-min data (7d) for recent rejection candle detection...")
    try:
        df_1m = ticker.history(period="7d", interval="1m")
        if df_1m.index.tz is not None:
            df_1m.index = df_1m.index.tz_convert("Asia/Kolkata")
    except:
        df_1m = pd.DataFrame()
    
    daily_df = ticker.history(period="1y", interval="1d")
    if daily_df.index.tz is not None:
        daily_df.index = daily_df.index.tz_convert("Asia/Kolkata")
    daily_close_map = {idx.date(): float(row['Close']) for idx, row in daily_df.iterrows()}
    
    model_preds = load_model_predictions()
    
    # Filter to Sep 1 onwards
    start_date = dt.date(2026, 9, 1)
    unique_days = sorted(set(d for d in df_5m.index.date if d >= start_date))
    print(f"Trading days from Sep 1: {len(unique_days)} ({unique_days[0]} to {unique_days[-1]})")
    
    results = []
    
    for t_date in unique_days:
        d_str = t_date.strftime("%Y-%m-%d")
        day_5m = df_5m[df_5m.index.date == t_date]
        if len(day_5m) < 10:
            continue
        
        prev_dates = [d for d in daily_close_map.keys() if d < t_date]
        if not prev_dates:
            continue
        prev_close = daily_close_map[max(prev_dates)]
        day_open = float(day_5m.iloc[0]['Open'])
        gap_pts = day_open - prev_close
        is_gap = abs(gap_pts) > 12.0
        
        # Model prediction
        pred = model_preds.get(d_str)
        if not pred:
            continue
        if not pred["trade"]:
            results.append({
                "date": d_str, "day": t_date.strftime("%a"),
                "status": "SIDELINED", "reason": f"Conviction {pred['conviction_pct']}% < threshold",
                "v1_pnl": 0, "v2_pnl": 0, "v2_edge": 0
            })
            continue
        
        is_call = pred["is_call"]
        conv = pred["conviction_pct"]
        
        # Entry time
        entry_time_target = dt.time(9, 35) if is_gap else dt.time(9, 20)
        
        # ─── V1: Standard VWAP Entry ───
        v1_bars = day_5m[day_5m.index.time >= entry_time_target]
        if v1_bars.empty:
            continue
        
        v1_entry_bar = v1_bars.iloc[0]
        v1_entry_spot = float(v1_entry_bar['Open'])
        v1_entry_time = v1_entry_bar.name.strftime("%H:%M")
        atm_strike = round(v1_entry_spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
        dte = max(float(get_days_to_monthly_expiry(t_date)), 1.0)
        
        v1_entry_prem = calculate_bsm_option_price(v1_entry_spot, atm_strike, days_to_expiry=dte, is_call=is_call)
        v1_target_prem = v1_entry_prem + TARGET_OPT
        v1_sl_prem = max(v1_entry_prem - SL_OPT, 1.0)
        
        # Simulate V1
        v1_exit_prem = v1_entry_prem
        v1_hit = "CUTOFF"
        v1_hit_time = "15:05"
        eval_bars = day_5m[(day_5m.index.time >= entry_time_target) & (day_5m.index.time <= dt.time(15, 5))]
        
        for _, bar in eval_bars.iterrows():
            hi, lo = float(bar['High']), float(bar['Low'])
            if not is_call:  # PE
                prem_hi = calculate_bsm_option_price(hi, atm_strike, days_to_expiry=dte, is_call=False)
                prem_lo = calculate_bsm_option_price(lo, atm_strike, days_to_expiry=dte, is_call=False)
                if prem_hi <= v1_sl_prem:
                    v1_exit_prem = v1_sl_prem
                    v1_hit = "SL"
                    v1_hit_time = bar.name.strftime("%H:%M")
                    break
                if prem_lo >= v1_target_prem:
                    v1_exit_prem = v1_target_prem
                    v1_hit = "TARGET"
                    v1_hit_time = bar.name.strftime("%H:%M")
                    break
            else:  # CE
                prem_hi = calculate_bsm_option_price(hi, atm_strike, days_to_expiry=dte, is_call=True)
                prem_lo = calculate_bsm_option_price(lo, atm_strike, days_to_expiry=dte, is_call=True)
                if prem_lo <= v1_sl_prem:
                    v1_exit_prem = v1_sl_prem
                    v1_hit = "SL"
                    v1_hit_time = bar.name.strftime("%H:%M")
                    break
                if prem_hi >= v1_target_prem:
                    v1_exit_prem = v1_target_prem
                    v1_hit = "TARGET"
                    v1_hit_time = bar.name.strftime("%H:%M")
                    break
        
        if v1_hit == "CUTOFF":
            last_spot = float(eval_bars.iloc[-1]['Close'])
            v1_exit_prem = calculate_bsm_option_price(last_spot, atm_strike, days_to_expiry=dte, is_call=is_call)
        
        v1_opt_pnl = round(v1_exit_prem - v1_entry_prem, 2)
        v1_net = round(v1_opt_pnl * LOT_QTY * NUM_LOTS - BROKERAGE, 0)
        
        # ─── V2: Tranche Entry + Runner ───
        # Phase A: Check for +1.5sigma rejection in 09:15-09:20 window (using 5-min bars)
        morning_bars = day_5m[(day_5m.index.time >= dt.time(9, 15)) & (day_5m.index.time < dt.time(9, 25))]
        
        # Compute VWAP from morning bars
        tranche1_entry = None
        tranche1_entry_prem = None
        tranche1_entry_time = None
        tranche1_spot = None
        
        if len(morning_bars) >= 2 and not is_call:  # Only for PE trades (DOWN model)
            tp = (morning_bars['High'] + morning_bars['Low'] + morning_bars['Close']) / 3.0
            vol = morning_bars['Volume'].astype(float)
            cum_tp_vol = (tp * vol).cumsum()
            cum_vol = vol.cumsum()
            vwap_series = cum_tp_vol / cum_vol.replace(0, np.nan)
            
            if len(vwap_series) >= 2:
                last_vwap = vwap_series.iloc[-1]
                # Compute std dev
                sq_diff = ((tp - vwap_series) ** 2 * vol).cumsum()
                variance = sq_diff / cum_vol.replace(0, np.nan)
                std_series = np.sqrt(variance)
                last_std = std_series.iloc[-1] if not np.isnan(std_series.iloc[-1]) else 0
                
                upper_1_5sigma = last_vwap + 1.5 * last_std
                
                # Check if any 5-min bar reached upper band and formed rejection
                for _, mbar in morning_bars.iterrows():
                    hi = float(mbar['High'])
                    if hi >= upper_1_5sigma and is_rejection_candle(mbar):
                        tranche1_spot = float(mbar['Low'])  # Enter at rejection low break
                        tranche1_entry_prem = calculate_bsm_option_price(tranche1_spot, atm_strike, days_to_expiry=dte, is_call=False)
                        tranche1_entry_time = mbar.name.strftime("%H:%M")
                        tranche1_entry = True
                        break
        
        # Phase B: Tranche 2 at VWAP breakdown (same as V1 entry or later)
        v2_lot1_entry_prem = tranche1_entry_prem if tranche1_entry else v1_entry_prem
        v2_lot2_entry_prem = v1_entry_prem  # VWAP breakdown entry
        v2_lot1_entry_time = tranche1_entry_time if tranche1_entry else v1_entry_time
        v2_lot2_entry_time = v1_entry_time
        
        if not tranche1_entry:
            # No rejection found: both lots enter at VWAP breakdown (same as V1 entry)
            v2_lot1_entry_prem = v1_entry_prem
        
        # ─── V2 Lot 1: Target +10 flat ───
        v2_lot1_target = v2_lot1_entry_prem + TARGET_OPT
        v2_lot1_sl = max(v2_lot1_entry_prem - SL_OPT, 1.0)
        v2_lot1_exit = v2_lot1_entry_prem
        v2_lot1_hit = "CUTOFF"
        v2_lot1_hit_time = "15:05"
        
        for _, bar in eval_bars.iterrows():
            hi, lo = float(bar['High']), float(bar['Low'])
            if not is_call:
                prem_hi = calculate_bsm_option_price(hi, atm_strike, days_to_expiry=dte, is_call=False)
                prem_lo = calculate_bsm_option_price(lo, atm_strike, days_to_expiry=dte, is_call=False)
                if prem_hi <= v2_lot1_sl:
                    v2_lot1_exit = v2_lot1_sl
                    v2_lot1_hit = "SL"
                    v2_lot1_hit_time = bar.name.strftime("%H:%M")
                    break
                if prem_lo >= v2_lot1_target:
                    v2_lot1_exit = v2_lot1_target
                    v2_lot1_hit = "TARGET"
                    v2_lot1_hit_time = bar.name.strftime("%H:%M")
                    break
            else:
                prem_hi = calculate_bsm_option_price(hi, atm_strike, days_to_expiry=dte, is_call=True)
                prem_lo = calculate_bsm_option_price(lo, atm_strike, days_to_expiry=dte, is_call=True)
                if prem_lo <= v2_lot1_sl:
                    v2_lot1_exit = v2_lot1_sl
                    v2_lot1_hit = "SL"
                    v2_lot1_hit_time = bar.name.strftime("%H:%M")
                    break
                if prem_hi >= v2_lot1_target:
                    v2_lot1_exit = v2_lot1_target
                    v2_lot1_hit = "TARGET"
                    v2_lot1_hit_time = bar.name.strftime("%H:%M")
                    break
        
        if v2_lot1_hit == "CUTOFF":
            last_spot = float(eval_bars.iloc[-1]['Close'])
            v2_lot1_exit = calculate_bsm_option_price(last_spot, atm_strike, days_to_expiry=dte, is_call=is_call)
        
        # ─── V2 Lot 2: Runner (trail 5-min 20 EMA) ───
        v2_lot2_sl = max(v2_lot2_entry_prem - SL_OPT, 1.0)
        v2_lot2_exit = v2_lot2_entry_prem
        v2_lot2_hit = "CUTOFF"
        v2_lot2_hit_time = "15:05"
        
        if v2_lot1_hit == "SL":
            # If Lot 1 hit SL, Lot 2 also hits SL (same direction)
            v2_lot2_exit = v2_lot2_sl
            v2_lot2_hit = "SL"
            v2_lot2_hit_time = v2_lot1_hit_time
        else:
            # Runner: compute 20 EMA on 5-min spot data
            day_5m_close = day_5m['Close'].astype(float)
            ema_20 = day_5m_close.ewm(span=20, adjust=False).mean()
            
            lot1_target_reached = False
            runner_trail_active = False
            
            for idx, bar in eval_bars.iterrows():
                hi, lo, cl = float(bar['High']), float(bar['Low']), float(bar['Close'])
                b_time_str = idx.strftime("%H:%M")
                
                if not is_call:  # PE runner
                    current_prem = calculate_bsm_option_price(cl, atm_strike, days_to_expiry=dte, is_call=False)
                    prem_gain = current_prem - v2_lot2_entry_prem
                    
                    # Check SL first
                    prem_at_hi = calculate_bsm_option_price(hi, atm_strike, days_to_expiry=dte, is_call=False)
                    if prem_at_hi <= v2_lot2_sl:
                        v2_lot2_exit = v2_lot2_sl
                        v2_lot2_hit = "SL"
                        v2_lot2_hit_time = b_time_str
                        break
                    
                    if prem_gain >= TARGET_OPT:
                        lot1_target_reached = True
                        runner_trail_active = True
                    
                    # Once runner trail active: exit on 5-min close above 20 EMA
                    if runner_trail_active and idx in ema_20.index:
                        ema_val = ema_20[idx]
                        if cl > ema_val:  # For PE (bearish): close ABOVE EMA = exit
                            v2_lot2_exit = current_prem
                            v2_lot2_hit = "EMA_EXIT"
                            v2_lot2_hit_time = b_time_str
                            break
                else:  # CE runner
                    current_prem = calculate_bsm_option_price(cl, atm_strike, days_to_expiry=dte, is_call=True)
                    prem_gain = current_prem - v2_lot2_entry_prem
                    
                    prem_at_lo = calculate_bsm_option_price(lo, atm_strike, days_to_expiry=dte, is_call=True)
                    if prem_at_lo <= v2_lot2_sl:
                        v2_lot2_exit = v2_lot2_sl
                        v2_lot2_hit = "SL"
                        v2_lot2_hit_time = b_time_str
                        break
                    
                    if prem_gain >= TARGET_OPT:
                        lot1_target_reached = True
                        runner_trail_active = True
                    
                    if runner_trail_active and idx in ema_20.index:
                        ema_val = ema_20[idx]
                        if cl < ema_val:  # For CE (bullish): close BELOW EMA = exit
                            v2_lot2_exit = current_prem
                            v2_lot2_hit = "EMA_EXIT"
                            v2_lot2_hit_time = b_time_str
                            break
            
            if v2_lot2_hit == "CUTOFF":
                last_spot = float(eval_bars.iloc[-1]['Close'])
                v2_lot2_exit = calculate_bsm_option_price(last_spot, atm_strike, days_to_expiry=dte, is_call=is_call)
        
        v2_lot1_pnl = round((v2_lot1_exit - v2_lot1_entry_prem) * LOT_QTY, 0)
        v2_lot2_pnl = round((v2_lot2_exit - v2_lot2_entry_prem) * LOT_QTY, 0)
        v2_net = round(v2_lot1_pnl + v2_lot2_pnl - BROKERAGE, 0)
        
        v2_edge = round(v2_net - v1_net, 0)
        
        contract_type = "CE" if is_call else "PE"
        
        results.append({
            "date": d_str,
            "day": t_date.strftime("%a"),
            "status": "TRADED",
            "direction": "UP" if is_call else "DOWN",
            "conviction": conv,
            "contract": f"CDSL {int(atm_strike)} {contract_type}",
            "gap": round(gap_pts, 1),
            # V1
            "v1_entry_time": v1_entry_time,
            "v1_entry_prem": round(v1_entry_prem, 2),
            "v1_exit_prem": round(v1_exit_prem, 2),
            "v1_opt_pnl": v1_opt_pnl,
            "v1_hit": v1_hit,
            "v1_hit_time": v1_hit_time,
            "v1_net": v1_net,
            # V2 Tranche
            "v2_tranche1": "YES" if tranche1_entry else "NO",
            "v2_lot1_entry_prem": round(v2_lot1_entry_prem, 2),
            "v2_lot1_exit_prem": round(v2_lot1_exit, 2),
            "v2_lot1_hit": v2_lot1_hit,
            "v2_lot1_pnl": v2_lot1_pnl,
            "v2_lot2_entry_prem": round(v2_lot2_entry_prem, 2),
            "v2_lot2_exit_prem": round(v2_lot2_exit, 2),
            "v2_lot2_hit": v2_lot2_hit,
            "v2_lot2_hit_time": v2_lot2_hit_time,
            "v2_lot2_pnl": v2_lot2_pnl,
            "v2_net": v2_net,
            "v2_edge": v2_edge,
            "reason": ""
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = run_v1_v2_backtest()
    
    # Save CSV
    csv_out = os.path.join(PROJECT_ROOT, "backtest_results", "v1_vs_v2_backtest.csv")
    df.to_csv(csv_out, index=False, encoding='utf-8')
    print(f"\nSaved to {csv_out}")
    
    # Print trade-by-trade
    traded = df[df['status'] == 'TRADED']
    sidelined = df[df['status'] == 'SIDELINED']
    
    print(f"\n{'='*120}")
    print(f"V1 vs V2 BACKTEST RESULTS — CDSL.NS (Sep 2026)")
    print(f"{'='*120}")
    
    for _, r in traded.iterrows():
        print(f"\n--- {r['date']} ({r['day']}) | {r['contract']} | Conv: {r['conviction']}% | Gap: {r['gap']:+.1f} ---")
        print(f"  V1: Entry {r['v1_entry_time']} @ Rs{r['v1_entry_prem']} -> Exit Rs{r['v1_exit_prem']} [{r['v1_hit']} @ {r['v1_hit_time']}] | Net: Rs{r['v1_net']:+,.0f}")
        t1_tag = " (SPIKE REJECTION)" if r['v2_tranche1'] == "YES" else ""
        print(f"  V2 Lot1{t1_tag}: Entry Rs{r['v2_lot1_entry_prem']} -> Exit Rs{r['v2_lot1_exit_prem']} [{r['v2_lot1_hit']}] | PnL: Rs{r['v2_lot1_pnl']:+,.0f}")
        print(f"  V2 Lot2 (RUNNER): Entry Rs{r['v2_lot2_entry_prem']} -> Exit Rs{r['v2_lot2_exit_prem']} [{r['v2_lot2_hit']} @ {r['v2_lot2_hit_time']}] | PnL: Rs{r['v2_lot2_pnl']:+,.0f}")
        print(f"  V2 Total: Rs{r['v2_net']:+,.0f} | V2 Edge over V1: Rs{r['v2_edge']:+,.0f}")
    
    print(f"\n{'='*120}")
    print(f"SIDELINED DAYS: {len(sidelined)}")
    for _, r in sidelined.iterrows():
        print(f"  {r['date']} ({r['day']}) - {r['reason']}")
    
    v1_total = traded['v1_net'].sum()
    v2_total = traded['v2_net'].sum()
    edge_total = traded['v2_edge'].sum()
    
    v1_wins = len(traded[traded['v1_net'] > 0])
    v2_wins = len(traded[traded['v2_net'] > 0])
    
    print(f"\n{'='*120}")
    print(f"SUMMARY")
    print(f"{'='*120}")
    print(f"  Total Traded Days:  {len(traded)}")
    print(f"  V1 Total Net P&L:  Rs{v1_total:+,.0f} (Wins: {v1_wins})")
    print(f"  V2 Total Net P&L:  Rs{v2_total:+,.0f} (Wins: {v2_wins})")
    print(f"  V2 Edge over V1:   Rs{edge_total:+,.0f} ({round(edge_total/abs(v1_total)*100 if v1_total else 0, 1)}%)")
    print(f"  V1 Avg/trade:      Rs{v1_total/len(traded):+,.0f}")
    print(f"  V2 Avg/trade:      Rs{v2_total/len(traded):+,.0f}")
