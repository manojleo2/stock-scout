import sys, os
import pandas as pd
import numpy as np
import yfinance as yf

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.paper_trading import calculate_bsm_option_price
from utils.opening_predictor import get_days_to_monthly_expiry
from utils.backtest_intraday_spikes import load_genuine_model_predictions

ticker = yf.Ticker('CDSL.NS')
df = ticker.history(period='60d', interval='5m')
if df.index.tz is not None:
    df.index = df.index.tz_convert('Asia/Kolkata')

unique_days = sorted(list(set(df.index.date)))
model_preds = load_genuine_model_predictions()

LOT_QTY = 475
NUM_LOTS = 2
TOTAL_QTY = LOT_QTY * NUM_LOTS # 950
BROKERAGE = 100.0

baseline_trades = []
upgraded_trades = []

for t_date in unique_days:
    d_str = t_date.strftime('%Y-%m-%d')
    day_bars = df[df.index.date == t_date]
    if len(day_bars) < 10:
        continue

    pred = model_preds.get(d_str)
    if not pred or not pred['trade']:
        continue
    is_call = pred['is_call']
    dte = max(float(get_days_to_monthly_expiry(t_date)), 1.0)

    # 1. BASELINE SETUP (09:20 AM entry, Fixed Target +10, Fixed SL -5)
    sub_bars = day_bars[day_bars.index.time >= pd.to_datetime('09:20').time()]
    if not sub_bars.empty:
        e_bar = sub_bars.iloc[0]
        e_spot = float(e_bar['Open'])
        atm_strike = round(e_spot / 20.0) * 20.0
        e_prem = calculate_bsm_option_price(e_spot, atm_strike, dte, is_call=is_call)
        
        hit_status = 'CUTOFF'
        exit_prem = e_prem
        for b_idx, bar in sub_bars.iterrows():
            if b_idx.time() > pd.to_datetime('15:05').time():
                break
            hi = float(bar['High'])
            lo = float(bar['Low'])
            if not is_call: # PUT
                p_hi = calculate_bsm_option_price(hi, atm_strike, dte, is_call=False)
                p_lo = calculate_bsm_option_price(lo, atm_strike, dte, is_call=False)
                if p_hi <= (e_prem - 5.0):
                    hit_status = 'SL_HIT'
                    exit_prem = e_prem - 5.0
                    break
                if p_lo >= (e_prem + 10.0):
                    hit_status = 'TARGET_HIT'
                    exit_prem = e_prem + 10.0
                    break
            else: # CALL
                p_hi = calculate_bsm_option_price(hi, atm_strike, dte, is_call=True)
                p_lo = calculate_bsm_option_price(lo, atm_strike, dte, is_call=True)
                if p_lo <= (e_prem - 5.0):
                    hit_status = 'SL_HIT'
                    exit_prem = e_prem - 5.0
                    break
                if p_hi >= (e_prem + 10.0):
                    hit_status = 'TARGET_HIT'
                    exit_prem = e_prem + 10.0
                    break
        if hit_status == 'CUTOFF':
            last_s = float(sub_bars[sub_bars.index.time <= pd.to_datetime('15:05').time()].iloc[-1]['Close'])
            exit_prem = calculate_bsm_option_price(last_s, atm_strike, dte, is_call=is_call)
        
        pnl = round((exit_prem - e_prem) * TOTAL_QTY - BROKERAGE, 2)
        baseline_trades.append({'date': d_str, 'pnl': pnl, 'status': hit_status})

    # 2. UPGRADED SETUP: VWAP Confirmation Entry + Dynamic Trailing SL
    first_bars = day_bars[day_bars.index.time <= pd.to_datetime('09:20').time()]
    vwap_0920 = ((first_bars['Close'] * first_bars['Volume']).sum()) / (first_bars['Volume'].sum() + 1e-9)

    # Search for valid entry between 09:20 and 09:40 with VWAP confirmation
    valid_entry_bar = None
    search_bars = day_bars[(day_bars.index.time >= pd.to_datetime('09:20').time()) & (day_bars.index.time <= pd.to_datetime('09:40').time())]
    for b_idx, bar in search_bars.iterrows():
        b_close = float(bar['Close'])
        # For PUT: must be BELOW VWAP; For CALL: must be ABOVE VWAP
        if (not is_call and b_close < vwap_0920) or (is_call and b_close > vwap_0920):
            valid_entry_bar = bar
            break

    if valid_entry_bar is not None:
        e_spot_up = float(valid_entry_bar['Close'])
        atm_strike_up = round(e_spot_up / 20.0) * 20.0
        e_prem_up = calculate_bsm_option_price(e_spot_up, atm_strike_up, dte, is_call=is_call)
        
        eval_bars_up = day_bars[(day_bars.index > valid_entry_bar.name) & (day_bars.index.time <= pd.to_datetime('15:05').time())]
        
        current_sl = e_prem_up - 5.0
        max_favorable_gain = 0.0
        hit_status_up = 'CUTOFF'
        exit_prem_up = e_prem_up
        
        for b_idx, bar in eval_bars_up.iterrows():
            hi = float(bar['High'])
            lo = float(bar['Low'])
            
            if not is_call: # PUT
                p_hi = calculate_bsm_option_price(hi, atm_strike_up, dte, is_call=False)
                p_lo = calculate_bsm_option_price(lo, atm_strike_up, dte, is_call=False)
                candle_max_gain = p_lo - e_prem_up
                
                # Check trailing ratchets
                if candle_max_gain > max_favorable_gain:
                    max_favorable_gain = candle_max_gain
                    if max_favorable_gain >= 8.0:
                        current_sl = max(current_sl, e_prem_up + 4.0) # Lock +4
                    elif max_favorable_gain >= 5.0:
                        current_sl = max(current_sl, e_prem_up) # Move to cost
                
                # Check if SL hit
                if p_hi <= current_sl:
                    if current_sl > e_prem_up:
                        hit_status_up = 'TRAILED_WIN (+4 Locked)'
                    elif current_sl == e_prem_up:
                        hit_status_up = 'TRAILED_BE (Cost 0)'
                    else:
                        hit_status_up = 'SL_HIT (-5)'
                    exit_prem_up = current_sl
                    break
                
                # Check Target (+10)
                if p_lo >= (e_prem_up + 10.0):
                    hit_status_up = 'TARGET_HIT (+10)'
                    exit_prem_up = e_prem_up + 10.0
                    break
            else: # CALL
                p_hi = calculate_bsm_option_price(hi, atm_strike_up, dte, is_call=True)
                p_lo = calculate_bsm_option_price(lo, atm_strike_up, dte, is_call=True)
                candle_max_gain = p_hi - e_prem_up
                
                if candle_max_gain > max_favorable_gain:
                    max_favorable_gain = candle_max_gain
                    if max_favorable_gain >= 8.0:
                        current_sl = max(current_sl, e_prem_up + 4.0)
                    elif max_favorable_gain >= 5.0:
                        current_sl = max(current_sl, e_prem_up)
                
                if p_lo <= current_sl:
                    if current_sl > e_prem_up:
                        hit_status_up = 'TRAILED_WIN (+4 Locked)'
                    elif current_sl == e_prem_up:
                        hit_status_up = 'TRAILED_BE (Cost 0)'
                    else:
                        hit_status_up = 'SL_HIT (-5)'
                    exit_prem_up = current_sl
                    break
                
                if p_hi >= (e_prem_up + 10.0):
                    hit_status_up = 'TARGET_HIT (+10)'
                    exit_prem_up = e_prem_up + 10.0
                    break
                    
        if hit_status_up == 'CUTOFF' and not eval_bars_up.empty:
            last_s = float(eval_bars_up.iloc[-1]['Close'])
            exit_prem_up = calculate_bsm_option_price(last_s, atm_strike_up, dte, is_call=is_call)
            
        pnl_up = round((exit_prem_up - e_prem_up) * TOTAL_QTY - BROKERAGE, 2)
        upgraded_trades.append({'date': d_str, 'pnl': pnl_up, 'status': hit_status_up})
    else:
        upgraded_trades.append({'date': d_str, 'pnl': 0.0, 'status': 'FILTERED_OUT (VWAP)'})

# 3. Test VWAP Confirmation ALONE (No Trailing, Fixed Target +10, SL -5)
vwap_only_trades = []
calibrated_trailing_trades = []

for t_date in unique_days:
    d_str = t_date.strftime('%Y-%m-%d')
    day_bars = df[df.index.date == t_date]
    if len(day_bars) < 10:
        continue
    pred = model_preds.get(d_str)
    if not pred or not pred['trade']:
        continue
    is_call = pred['is_call']
    dte = max(float(get_days_to_monthly_expiry(t_date)), 1.0)
    
    first_bars = day_bars[day_bars.index.time <= pd.to_datetime('09:20').time()]
    vwap_0920 = ((first_bars['Close'] * first_bars['Volume']).sum()) / (first_bars['Volume'].sum() + 1e-9)

    valid_entry_bar = None
    search_bars = day_bars[(day_bars.index.time >= pd.to_datetime('09:20').time()) & (day_bars.index.time <= pd.to_datetime('09:40').time())]
    for b_idx, bar in search_bars.iterrows():
        b_close = float(bar['Close'])
        if (not is_call and b_close < vwap_0920) or (is_call and b_close > vwap_0920):
            valid_entry_bar = bar
            break
            
    if valid_entry_bar is not None:
        e_spot = float(valid_entry_bar['Close'])
        atm_strike = round(e_spot / 20.0) * 20.0
        e_prem = calculate_bsm_option_price(e_spot, atm_strike, dte, is_call=is_call)
        eval_bars = day_bars[(day_bars.index > valid_entry_bar.name) & (day_bars.index.time <= pd.to_datetime('15:05').time())]
        
        # Test A: VWAP Only
        hit_status_vo = 'CUTOFF'
        exit_prem_vo = e_prem
        for b_idx, bar in eval_bars.iterrows():
            hi = float(bar['High'])
            lo = float(bar['Low'])
            if not is_call:
                p_hi = calculate_bsm_option_price(hi, atm_strike, dte, is_call=False)
                p_lo = calculate_bsm_option_price(lo, atm_strike, dte, is_call=False)
                if p_hi <= (e_prem - 5.0):
                    hit_status_vo = 'SL_HIT'
                    exit_prem_vo = e_prem - 5.0
                    break
                if p_lo >= (e_prem + 10.0):
                    hit_status_vo = 'TARGET_HIT'
                    exit_prem_vo = e_prem + 10.0
                    break
            else:
                p_hi = calculate_bsm_option_price(hi, atm_strike, dte, is_call=True)
                p_lo = calculate_bsm_option_price(lo, atm_strike, dte, is_call=True)
                if p_lo <= (e_prem - 5.0):
                    hit_status_vo = 'SL_HIT'
                    exit_prem_vo = e_prem - 5.0
                    break
                if p_hi >= (e_prem + 10.0):
                    hit_status_vo = 'TARGET_HIT'
                    exit_prem_vo = e_prem + 10.0
                    break
        if hit_status_vo == 'CUTOFF' and not eval_bars.empty:
            last_s = float(eval_bars.iloc[-1]['Close'])
            exit_prem_vo = calculate_bsm_option_price(last_s, atm_strike, dte, is_call=is_call)
        pnl_vo = round((exit_prem_vo - e_prem) * TOTAL_QTY - BROKERAGE, 2)
        vwap_only_trades.append({'date': d_str, 'pnl': pnl_vo, 'status': hit_status_vo})

        # Test B: Calibrated Trailing (Move to Cost only after +7.00, Lock +5 after +8.50)
        current_sl_cal = e_prem - 5.0
        max_favorable_gain_cal = 0.0
        hit_status_cal = 'CUTOFF'
        exit_prem_cal = e_prem
        for b_idx, bar in eval_bars.iterrows():
            hi = float(bar['High'])
            lo = float(bar['Low'])
            if not is_call:
                p_hi = calculate_bsm_option_price(hi, atm_strike, dte, is_call=False)
                p_lo = calculate_bsm_option_price(lo, atm_strike, dte, is_call=False)
                c_gain = p_lo - e_prem
                if c_gain > max_favorable_gain_cal:
                    max_favorable_gain_cal = c_gain
                    if max_favorable_gain_cal >= 8.5:
                        current_sl_cal = max(current_sl_cal, e_prem + 5.0)
                    elif max_favorable_gain_cal >= 7.0:
                        current_sl_cal = max(current_sl_cal, e_prem + 1.0) # Move to cost + 1
                if p_hi <= current_sl_cal:
                    hit_status_cal = 'TRAILED_EXIT'
                    exit_prem_cal = current_sl_cal
                    break
                if p_lo >= (e_prem + 10.0):
                    hit_status_cal = 'TARGET_HIT'
                    exit_prem_cal = e_prem + 10.0
                    break
            else:
                p_hi = calculate_bsm_option_price(hi, atm_strike, dte, is_call=True)
                p_lo = calculate_bsm_option_price(lo, atm_strike, dte, is_call=True)
                c_gain = p_hi - e_prem
                if c_gain > max_favorable_gain_cal:
                    max_favorable_gain_cal = c_gain
                    if max_favorable_gain_cal >= 8.5:
                        current_sl_cal = max(current_sl_cal, e_prem + 5.0)
                    elif max_favorable_gain_cal >= 7.0:
                        current_sl_cal = max(current_sl_cal, e_prem + 1.0)
                if p_lo <= current_sl_cal:
                    hit_status_cal = 'TRAILED_EXIT'
                    exit_prem_cal = current_sl_cal
                    break
                if p_hi >= (e_prem + 10.0):
                    hit_status_cal = 'TARGET_HIT'
                    exit_prem_cal = e_prem + 10.0
                    break
        if hit_status_cal == 'CUTOFF' and not eval_bars.empty:
            last_s = float(eval_bars.iloc[-1]['Close'])
            exit_prem_cal = calculate_bsm_option_price(last_s, atm_strike, dte, is_call=is_call)
        pnl_cal = round((exit_prem_cal - e_prem) * TOTAL_QTY - BROKERAGE, 2)
        calibrated_trailing_trades.append({'date': d_str, 'pnl': pnl_cal, 'status': hit_status_cal})

vo_df = pd.DataFrame(vwap_only_trades)
cal_df = pd.DataFrame(calibrated_trailing_trades)

print('=== 3. VWAP CONFIRMATION ALONE (Target +10, SL -5) ===')
print('Total Trades:', len(vo_df))
print('Target (+10):', len(vo_df[vo_df['status'] == 'TARGET_HIT']))
print('SL Hit (-5):', len(vo_df[vo_df['status'] == 'SL_HIT']))
print('Cutoffs (03:05 PM):', len(vo_df[vo_df['status'] == 'CUTOFF']))
vo_pnl = vo_df['pnl'].sum()
print(f'Total Net PnL: Rs {vo_pnl:,.2f}')

print('\n=== 4. VWAP + CALIBRATED TRAILING (+7 -> Cost+1, +8.5 -> +5) ===')
print('Total Trades:', len(cal_df))
print('Target (+10):', len(cal_df[cal_df['status'] == 'TARGET_HIT']))
print('Trailed / SL Exits:', len(cal_df[cal_df['status'].str.contains('TRAILED|SL')]))
print('Cutoffs (03:05 PM):', len(cal_df[cal_df['status'] == 'CUTOFF']))
cal_pnl = cal_df['pnl'].sum()
print(f'Total Net PnL: Rs {cal_pnl:,.2f}')

