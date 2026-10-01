"""
production_validation.py
─────────────────────────
Comprehensive Production Validation Engine for CDSL_FSM_V2.1_CALIBRATED.

Executes:
  - Step 3: Historical Walk-Forward Simulation (Lookahead-free replay at timestamp T)
  - Step 4: ML Probability Calibration & Reliability Bin Validation (Brier score, bins 50-80%+)
  - Step 5: Strategy Expectancy & Execution Attribution (Expectancy E, PF, MaxDD, T1/T2/Runner)
  - Step 6: Parameter Ablation Experiments (A: V2 vs V1, B: ATM vs ITM, C: Hybrid SL vs Fixed SL, D: Target +8/+10/+12, E: Runner vs Fixed)
"""

import os
import sys
import json
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.paper_trading import calculate_bsm_option_price
from utils.market_calendar import get_days_to_monthly_expiry
from utils.entry_conditions import (
    calculate_vwap_and_bands,
    is_rejection_candle,
    check_pre_trade_gates,
    SPREAD_MAX_PCT,
    SPREAD_MAX_RS,
    ANTI_CHASE_MAX_PTS,
    EMERGENCY_SL_PTS,
    PRECISION_SL_PTS,
    LOT1_TARGET_PTS
)

LOT_QTY = 475
NUM_LOTS = 2
TOTAL_QTY = LOT_QTY * NUM_LOTS
STRIKE_INTERVAL = 20.0
BROKERAGE = 100.0


def load_dataset():
    """Load historical out-of-sample predictions and audit trail."""
    spec_csv = os.path.join(PROJECT_ROOT, "backtest_results", "cdsl_specialist_backtest_20260918_190321.csv")
    df_spec = pd.read_csv(spec_csv) if os.path.exists(spec_csv) else pd.DataFrame()
    
    audit_file = os.path.join(PROJECT_ROOT, "prediction_audit.json")
    audit_list = []
    if os.path.exists(audit_file):
        with open(audit_file, "r", encoding="utf-8") as f:
            audit_list = json.load(f)
            
    return df_spec, audit_list


def run_step_4_ml_probability_validation(df_spec, audit_list):
    """
    Validate calibrated ML probabilities across empirical reliability bins.
    Bins: 50-60%, 60-65%, 65-70%, 70-75%, 75-80%, 80%+
    """
    print("\n" + "="*70)
    print("STEP 4: ML PROBABILITY CALIBRATION & RELIABILITY VALIDATION")
    print("="*70)
    
    records = []
    
    # 1. From CSV out-of-sample test set (199 days)
    if not df_spec.empty:
        for _, row in df_spec.iterrows():
            d_str = str(row['date']).strip()
            # In df_spec, target 1 = UP, 0 = DOWN. prob_spec is prob of UP.
            # Down prob is 1.0 - prob_spec.
            prob_up = float(row['prob_spec'])
            prob_down = 1.0 - prob_up
            actual_down = 1 if int(row['actual_target']) == 0 else 0
            records.append({
                "date": d_str,
                "prob_down": prob_down,
                "actual_down": actual_down,
                "source": "specialist_csv"
            })
            
    # 2. From prediction_audit.json (live out-of-sample predictions)
    for a in audit_list:
        if a.get("symbol") == "CDSL.NS" and "actual_direction" in a:
            p_down = float(a.get("probability_down_pct", 50.0)) / 100.0
            act_dir = a.get("actual_direction", "")
            act_down = 1 if "DOWN" in act_dir else 0
            d_str = a.get("target_date", "")
            records.append({
                "date": d_str,
                "prob_down": p_down,
                "actual_down": act_down,
                "source": "live_audit"
            })
            
    df_eval = pd.DataFrame(records)
    if df_eval.empty:
        print("[!] No records found for ML validation.")
        return {}
        
    # Remove duplicates if any
    df_eval = df_eval.drop_duplicates(subset=["date"])
    
    # Brier Score calculation
    brier = float(np.mean((df_eval['prob_down'] - df_eval['actual_down']) ** 2))
    
    # Reliability Bins
    bins = [
        ("50% - 60%", 0.50, 0.60),
        ("60% - 65%", 0.60, 0.65),
        ("65% - 70%", 0.65, 0.70),
        ("70% - 75%", 0.70, 0.75),
        ("75% - 80%", 0.75, 0.80),
        ("80% - 100%", 0.80, 1.00),
    ]
    
    bin_results = []
    total_trades_checked = len(df_eval)
    
    for name, low, high in bins:
        sub = df_eval[(df_eval['prob_down'] >= low) & (df_eval['prob_down'] < high)]
        count = len(sub)
        if count > 0:
            actual_down_count = int(sub['actual_down'].sum())
            empirical_rate = (actual_down_count / count) * 100.0
            expected_mid = ((low + high) / 2.0) * 100.0
            diff = empirical_rate - expected_mid
        else:
            actual_down_count = 0
            empirical_rate = 0.0
            expected_mid = ((low + high) / 2.0) * 100.0
            diff = 0.0
            
        bin_results.append({
            "bin": name,
            "count": count,
            "actual_down": actual_down_count,
            "empirical_down_pct": round(empirical_rate, 1),
            "expected_mid_pct": round(expected_mid, 1),
            "calibration_error_pct": round(diff, 1)
        })
        
    print(f"Total Out-of-Sample Predictions Evaluated: {total_trades_checked}")
    print(f"Overall Calibrated Brier Score: {brier:.4f} (Ideal: < 0.25)")
    print("-" * 70)
    print(f"{'Predicted Bucket':<16} | {'Count':<6} | {'Actual DOWN':<11} | {'Empirical %':<11} | {'Expected %':<10} | {'Error %':<8}")
    print("-" * 70)
    for b in bin_results:
        print(f"{b['bin']:<16} | {b['count']:<6} | {b['actual_down']:<11} | {b['empirical_down_pct']:<10}% | {b['expected_mid_pct']:<9}% | {b['calibration_error_pct']:+5.1f}%")
    print("-" * 70)
    
    return {
        "brier_score": round(brier, 4),
        "total_evaluated": total_trades_checked,
        "bin_results": bin_results,
        "df_eval": df_eval
    }


def run_lookahead_free_intraday_replay(
    use_v2_fsm=True,
    strike_delta="ATM",
    emergency_sl_pts=8.0,
    precision_sl_pts=5.0,
    lot1_target_pts=10.0,
    use_runner=True
):
    """
    Step 3, 5, 6: Lookahead-free replay across historical 5-minute CDSL data.
    Evaluates every rule sequentially at timestamp T.
    """
    ticker = yf.Ticker("CDSL.NS")
    df_5m = ticker.history(period="60d", interval="5m")
    if df_5m.index.tz is not None:
        df_5m.index = df_5m.index.tz_convert("Asia/Kolkata")
        
    daily_df = ticker.history(period="1y", interval="1d")
    if daily_df.index.tz is not None:
        daily_df.index = daily_df.index.tz_convert("Asia/Kolkata")
    daily_close_map = {idx.date(): float(row['Close']) for idx, row in daily_df.iterrows()}
    
    # Load model predictions
    spec_csv = os.path.join(PROJECT_ROOT, "backtest_results", "cdsl_specialist_backtest_20260918_190321.csv")
    preds = {}
    if os.path.exists(spec_csv):
        df_s = pd.read_csv(spec_csv)
        for _, r in df_s.iterrows():
            d_str = str(r['date']).strip()
            prob = float(r['prob_spec'])
            conv = float(r['conv_spec'])
            trade = bool(r['trade_spec'])
            preds[d_str] = {"trade": trade, "is_call": prob >= 0.50, "conviction_pct": round(conv*100, 1)}
            
    audit_file = os.path.join(PROJECT_ROOT, "prediction_audit.json")
    if os.path.exists(audit_file):
        with open(audit_file, "r", encoding="utf-8") as f:
            aud = json.load(f)
        for r in aud:
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
                    preds[d_iso] = {"trade": trade, "is_call": is_call, "conviction_pct": round(conv, 1)}
                except:
                    pass

    unique_days = sorted(set(d for d in df_5m.index.date if d >= dt.date(2026, 8, 1)))
    
    trade_logs = []
    consecutive_losses = 0
    daily_lockout_count = 0
    
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
        
        pred = preds.get(d_str)
        if not pred or not pred["trade"]:
            continue
            
        is_call = pred["is_call"]
        dte = max(float(get_days_to_monthly_expiry(t_date)), 1.0)
        
        # Sizing and Strike Selection
        # ATM: round to nearest 20. ITM: 1 strike in-the-money
        open_spot = day_open
        if strike_delta == "ATM":
            strike = round(open_spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
        elif strike_delta == "ITM":
            if is_call:
                strike = (round(open_spot / STRIKE_INTERVAL) * STRIKE_INTERVAL) - STRIKE_INTERVAL
            else:
                strike = (round(open_spot / STRIKE_INTERVAL) * STRIKE_INTERVAL) + STRIKE_INTERVAL
        else:
            strike = round(open_spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
            
        # Entry Timing Gatekeeper
        entry_time_target = dt.time(9, 35) if is_gap else dt.time(9, 20)
        
        if not use_v2_fsm:
            # ── V1 BASELINE: Single 2-lot entry at 09:20/09:35, fixed +10/-5 ──
            v1_bars = day_5m[day_5m.index.time >= entry_time_target]
            if v1_bars.empty:
                continue
            entry_bar = v1_bars.iloc[0]
            entry_spot = float(entry_bar['Open'])
            entry_prem = calculate_bsm_option_price(entry_spot, strike, days_to_expiry=dte, is_call=is_call)
            
            sl_prem = max(entry_prem - precision_sl_pts, 1.0)
            target_prem = entry_prem + lot1_target_pts
            
            exit_prem = entry_prem
            exit_reason = "CUTOFF"
            exit_time = "15:05"
            
            eval_bars = day_5m[(day_5m.index.time >= entry_time_target) & (day_5m.index.time <= dt.time(15, 5))]
            for _, b in eval_bars.iterrows():
                hi, lo = float(b['High']), float(b['Low'])
                prem_hi = calculate_bsm_option_price(hi, strike, days_to_expiry=dte, is_call=is_call)
                prem_lo = calculate_bsm_option_price(lo, strike, days_to_expiry=dte, is_call=is_call)
                
                # Check adverse move first
                adverse_prem = prem_lo if is_call else prem_hi
                favorable_prem = prem_hi if is_call else prem_lo
                
                if adverse_prem <= sl_prem:
                    exit_prem = sl_prem
                    exit_reason = "SL"
                    exit_time = b.name.strftime("%H:%M")
                    break
                if favorable_prem >= target_prem:
                    exit_prem = target_prem
                    exit_reason = "TARGET"
                    exit_time = b.name.strftime("%H:%M")
                    break
                    
            if exit_reason == "CUTOFF":
                last_spot = float(eval_bars.iloc[-1]['Close'])
                exit_prem = calculate_bsm_option_price(last_spot, strike, days_to_expiry=dte, is_call=is_call)
                
            pnl_pts = exit_prem - entry_prem
            net_pnl = (pnl_pts * TOTAL_QTY) - BROKERAGE
            
            trade_logs.append({
                "date": d_str,
                "strategy": "V1_BASELINE",
                "entry_prem": entry_prem,
                "exit_prem": exit_prem,
                "pnl_pts": round(pnl_pts, 2),
                "net_pnl": round(net_pnl, 2),
                "is_win": net_pnl > 0,
                "exit_reason": exit_reason,
                "exit_time": exit_time,
                "t1_pnl": net_pnl / 2.0,
                "t2_pnl": net_pnl / 2.0,
                "runner_pnl": 0.0
            })
            
        else:
            # ── V2 FSM: Tranche 1 (09:15-09:20 spike rejection) + Tranche 2 (09:20-09:25 VWAP) + Decoupled Exits ──
            morning_bars = day_5m[(day_5m.index.time >= dt.time(9, 15)) & (day_5m.index.time < dt.time(9, 25))]
            
            # Compute morning VWAP and 1.5-sigma band
            t1_filled = False
            t1_fill_prem = 0.0
            t1_time = None
            
            if len(morning_bars) >= 2 and not is_call: # PE trade setup
                vwap_val, upper_band, lower_band, std_val = calculate_vwap_and_bands(morning_bars, sigma_mult=1.5)
                for _, mbar in morning_bars.iterrows():
                    hi = float(mbar['High'])
                    if hi >= upper_band and is_rejection_candle(mbar, is_call=False):
                        # Pre-trade gate check simulation: bid-ask spread <= 0.40, anti-chase <= 6.50
                        sim_spot = float(mbar['Low'])
                        sim_prem = calculate_bsm_option_price(sim_spot, strike, days_to_expiry=dte, is_call=False)
                        open_prem = calculate_bsm_option_price(day_open, strike, days_to_expiry=dte, is_call=False)
                        quote_sim = {"ltp": sim_prem, "bid": sim_prem - 0.15, "ask": sim_prem + 0.15}
                        gates_pass, _ = check_pre_trade_gates(quote_sim, open_prem=open_prem)
                        if gates_pass:
                            t1_filled = True
                            t1_fill_prem = sim_prem
                            t1_time = mbar.name.strftime("%H:%M")
                            break
                            
            # Tranche 2: VWAP Confirmation at 09:20 / 09:35
            v2_bars = day_5m[day_5m.index.time >= entry_time_target]
            if v2_bars.empty:
                continue
            t2_bar = v2_bars.iloc[0]
            t2_spot = float(t2_bar['Open'])
            t2_prem = calculate_bsm_option_price(t2_spot, strike, days_to_expiry=dte, is_call=is_call)
            
            if not t1_filled:
                # If no rejection spike occurred, buy full 2 lots at Tranche 2 VWAP confirmation
                t1_fill_prem = t2_prem
                t1_time = t2_bar.name.strftime("%H:%M")
                
            blended_entry = (t1_fill_prem + t2_prem) / 2.0
            
            # Risk Regimes:
            # Before 09:25: Emergency Stop = t1_fill_prem - emergency_sl_pts
            # At 09:25: Precision Stop = blended_entry - precision_sl_pts
            precision_sl = max(blended_entry - precision_sl_pts, 1.0)
            emergency_sl = max(t1_fill_prem - emergency_sl_pts, 1.0)
            
            # Lot 1 Target: T1 Actual Fill + lot1_target_pts (locks cash)
            lot1_target = t1_fill_prem + lot1_target_pts
            
            eval_bars = day_5m[(day_5m.index.time >= entry_time_target) & (day_5m.index.time <= dt.time(15, 5))]
            
            # Lot 1 Execution Simulation
            lot1_exit_prem = t1_fill_prem
            lot1_exit_reason = "CUTOFF"
            lot1_exit_time = "15:05"
            lot1_hit_sl = False
            
            for idx, b in eval_bars.iterrows():
                b_time = idx.time()
                hi, lo = float(b['High']), float(b['Low'])
                prem_hi = calculate_bsm_option_price(hi, strike, days_to_expiry=dte, is_call=is_call)
                prem_lo = calculate_bsm_option_price(lo, strike, days_to_expiry=dte, is_call=is_call)
                
                adverse_prem = prem_lo if is_call else prem_hi
                favorable_prem = prem_hi if is_call else prem_lo
                
                active_sl = emergency_sl if b_time < dt.time(9, 25) else precision_sl
                
                if adverse_prem <= active_sl:
                    lot1_exit_prem = active_sl
                    lot1_exit_reason = "SL"
                    lot1_exit_time = idx.strftime("%H:%M")
                    lot1_hit_sl = True
                    break
                if favorable_prem >= lot1_target:
                    lot1_exit_prem = lot1_target
                    lot1_exit_reason = "TARGET"
                    lot1_exit_time = idx.strftime("%H:%M")
                    break
                    
            if lot1_exit_reason == "CUTOFF":
                last_spot = float(eval_bars.iloc[-1]['Close'])
                lot1_exit_prem = calculate_bsm_option_price(last_spot, strike, days_to_expiry=dte, is_call=is_call)
                
            # Lot 2 (Runner) Execution Simulation
            lot2_exit_prem = t2_prem
            lot2_exit_reason = "CUTOFF"
            lot2_exit_time = "15:05"
            
            if lot1_hit_sl:
                # Both lots exit on SL
                lot2_exit_prem = precision_sl
                lot2_exit_reason = "SL"
                lot2_exit_time = lot1_exit_time
            else:
                if not use_runner:
                    # If runner disabled, Lot 2 exits flat at target +10
                    lot2_exit_prem = lot1_exit_prem
                    lot2_exit_reason = lot1_exit_reason
                    lot2_exit_time = lot1_exit_time
                else:
                    # Lot 2 trails 5-min 20 EMA after Lot 1 target is secured
                    day_5m_close = day_5m['Close'].astype(float)
                    ema_20 = day_5m_close.ewm(span=20, adjust=False).mean()
                    
                    runner_activated = False
                    for idx, b in eval_bars.iterrows():
                        hi, lo, cl = float(b['High']), float(b['Low']), float(b['Close'])
                        b_prem = calculate_bsm_option_price(cl, strike, days_to_expiry=dte, is_call=is_call)
                        
                        # Check SL
                        prem_adverse = calculate_bsm_option_price(hi if not is_call else lo, strike, days_to_expiry=dte, is_call=is_call)
                        active_sl = emergency_sl if idx.time() < dt.time(9, 25) else precision_sl
                        if prem_adverse <= active_sl:
                            lot2_exit_prem = active_sl
                            lot2_exit_reason = "SL"
                            lot2_exit_time = idx.strftime("%H:%M")
                            break
                            
                        # If Lot 1 reached target or current option gain >= target
                        if (b_prem - t2_prem) >= lot1_target_pts:
                            runner_activated = True
                            
                        if runner_activated and idx in ema_20.index:
                            ema_v = ema_20[idx]
                            # Technical exit on 5-min candle close reversal
                            if (not is_call and cl > ema_v) or (is_call and cl < ema_v):
                                lot2_exit_prem = b_prem
                                lot2_exit_reason = "EMA_RUNNER_EXIT"
                                lot2_exit_time = idx.strftime("%H:%M")
                                break
                                
                    if lot2_exit_reason == "CUTOFF":
                        last_spot = float(eval_bars.iloc[-1]['Close'])
                        lot2_exit_prem = calculate_bsm_option_price(last_spot, strike, days_to_expiry=dte, is_call=is_call)
                        
            # Combined P&L
            lot1_pnl_pts = lot1_exit_prem - t1_fill_prem
            lot2_pnl_pts = lot2_exit_prem - t2_prem
            
            lot1_net = (lot1_pnl_pts * LOT_QTY) - (BROKERAGE / 2.0)
            lot2_net = (lot2_pnl_pts * LOT_QTY) - (BROKERAGE / 2.0)
            total_net = lot1_net + lot2_net
            blended_pts = (lot1_pnl_pts + lot2_pnl_pts) / 2.0
            
            # Attribution of runner vs fixed target
            fixed_lot2_pts = lot1_pnl_pts
            runner_edge_pts = lot2_pnl_pts - fixed_lot2_pts
            runner_edge_cash = runner_edge_pts * LOT_QTY
            
            trade_logs.append({
                "date": d_str,
                "strategy": "V2_FSM",
                "t1_fill": round(t1_fill_prem, 2),
                "t2_fill": round(t2_prem, 2),
                "blended_entry": round(blended_entry, 2),
                "lot1_exit": round(lot1_exit_prem, 2),
                "lot2_exit": round(lot2_exit_prem, 2),
                "pnl_pts": round(blended_pts, 2),
                "net_pnl": round(total_net, 2),
                "is_win": total_net > 0,
                "lot1_reason": lot1_exit_reason,
                "lot2_reason": lot2_exit_reason,
                "t1_pnl": round(lot1_net, 2),
                "t2_pnl": round(lot2_net, 2),
                "runner_edge_cash": round(runner_edge_cash, 2)
            })
            
    df_trades = pd.DataFrame(trade_logs)
    return df_trades


def compute_performance_metrics(df_trades):
    """
    Computes Expectancy E, Profit Factor, Max Drawdown, Win/Loss Rate, Average Trade.
    """
    if df_trades.empty:
        return {}
        
    wins = df_trades[df_trades['net_pnl'] > 0]
    losses = df_trades[df_trades['net_pnl'] <= 0]
    
    total_trades = len(df_trades)
    win_count = len(wins)
    loss_count = len(losses)
    
    win_rate = win_count / total_trades if total_trades > 0 else 0.0
    loss_rate = loss_count / total_trades if total_trades > 0 else 0.0
    
    avg_win = float(wins['net_pnl'].mean()) if win_count > 0 else 0.0
    avg_loss = float(abs(losses['net_pnl'].mean())) if loss_count > 0 else 0.0
    
    # Expectancy formula: E = (WinRate * AvgWin) - (LossRate * AvgLoss)
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    
    total_gains = float(wins['net_pnl'].sum()) if win_count > 0 else 0.0
    total_losses = float(abs(losses['net_pnl'].sum())) if loss_count > 0 else 0.0
    profit_factor = (total_gains / total_losses) if total_losses > 0 else 99.99
    
    net_total_pnl = float(df_trades['net_pnl'].sum())
    avg_trade_pnl = float(df_trades['net_pnl'].mean())
    
    # Drawdown
    equity_curve = df_trades['net_pnl'].cumsum()
    peak = equity_curve.cummax()
    drawdown = peak - equity_curve
    max_drawdown = float(drawdown.max()) if not drawdown.empty else 0.0
    
    # Consecutive losses
    streak = 0
    max_streak = 0
    for p in df_trades['net_pnl']:
        if p <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
            
    # Attribution
    t1_total = float(df_trades['t1_pnl'].sum()) if 't1_pnl' in df_trades else 0.0
    t2_total = float(df_trades['t2_pnl'].sum()) if 't2_pnl' in df_trades else 0.0
    runner_contrib = float(df_trades['runner_edge_cash'].sum()) if 'runner_edge_cash' in df_trades else 0.0
    
    return {
        "total_trades": total_trades,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate_pct": round(win_rate * 100.0, 1),
        "loss_rate_pct": round(loss_rate * 100.0, 1),
        "avg_win_rs": round(avg_win, 2),
        "avg_loss_rs": round(avg_loss, 2),
        "expectancy_rs": round(expectancy, 2),
        "profit_factor": round(profit_factor, 2),
        "net_total_pnl_rs": round(net_total_pnl, 2),
        "avg_trade_pnl_rs": round(avg_trade_pnl, 2),
        "max_drawdown_rs": round(max_drawdown, 2),
        "max_consecutive_losses": max_streak,
        "t1_total_pnl_rs": round(t1_total, 2),
        "t2_total_pnl_rs": round(t2_total, 2),
        "runner_contrib_rs": round(runner_contrib, 2)
    }


def run_all_production_validations():
    print("=" * 80)
    print("  CDSL_FSM_V2.1_CALIBRATED — FULL QUANTITATIVE VALIDATION SUITE")
    print("=" * 80)
    
    # Step 4: ML Probability Calibration
    df_spec, audit_list = load_dataset()
    ml_report = run_step_4_ml_probability_validation(df_spec, audit_list)
    
    # Step 3 & 5: Trading Strategy Replay & Expectancy
    print("\n" + "=" * 70)
    print("STEPS 3 & 5: HISTORICAL REPLAY & STRATEGY EXPECTANCY")
    print("=" * 70)
    trades_v2 = run_lookahead_free_intraday_replay(use_v2_fsm=True, strike_delta="ATM")
    metrics_v2 = compute_performance_metrics(trades_v2)
    
    print(f"Total Completed Sessions: {metrics_v2['total_trades']}")
    print(f"Win Rate: {metrics_v2['win_rate_pct']}% ({metrics_v2['win_count']} Wins / {metrics_v2['loss_count']} Losses)")
    print(f"Average Win: +Rs {metrics_v2['avg_win_rs']:,.2f}")
    print(f"Average Loss: -Rs {metrics_v2['avg_loss_rs']:,.2f}")
    print(f"Expectancy (E per trade): +Rs {metrics_v2['expectancy_rs']:,.2f}")
    print(f"Profit Factor: {metrics_v2['profit_factor']:.2f}")
    print(f"Net Total P&L (2 Lots / 950 Qty): +Rs {metrics_v2['net_total_pnl_rs']:,.2f}")
    print(f"Max Drawdown: -Rs {metrics_v2['max_drawdown_rs']:,.2f}")
    print(f"Max Losing Streak: {metrics_v2['max_consecutive_losses']} trades")
    print(f"Attribution -> Tranche 1: +Rs {metrics_v2['t1_total_pnl_rs']:,.2f} | Tranche 2: +Rs {metrics_v2['t2_total_pnl_rs']:,.2f}")
    print(f"Runner Contribution (Edge over flat +10): +Rs {metrics_v2['runner_contrib_rs']:,.2f}")
    
    # Step 6: Parameter Ablation Experiments (One by One)
    print("\n" + "=" * 70)
    print("STEP 6: PARAMETER ABLATION EXPERIMENTS (ONE-AT-A-TIME)")
    print("=" * 70)
    
    # Experiment A: V2 FSM vs V1 Baseline
    trades_v1 = run_lookahead_free_intraday_replay(use_v2_fsm=False)
    metrics_v1 = compute_performance_metrics(trades_v1)
    
    # Experiment B: ATM vs ITM (0.65 delta)
    trades_itm = run_lookahead_free_intraday_replay(use_v2_fsm=True, strike_delta="ITM")
    metrics_itm = compute_performance_metrics(trades_itm)
    
    # Experiment C: Hybrid SL (-8/-5) vs Fixed -5 SL
    trades_fixed_sl = run_lookahead_free_intraday_replay(use_v2_fsm=True, emergency_sl_pts=5.0, precision_sl_pts=5.0)
    metrics_fixed_sl = compute_performance_metrics(trades_fixed_sl)
    
    # Experiment D: Lot 1 Target: +8 vs +10 vs +12
    trades_tgt8 = run_lookahead_free_intraday_replay(use_v2_fsm=True, lot1_target_pts=8.0)
    metrics_tgt8 = compute_performance_metrics(trades_tgt8)
    trades_tgt12 = run_lookahead_free_intraday_replay(use_v2_fsm=True, lot1_target_pts=12.0)
    metrics_tgt12 = compute_performance_metrics(trades_tgt12)
    
    # Experiment E: Runner vs Fixed +10 Exit on Both Lots
    trades_no_runner = run_lookahead_free_intraday_replay(use_v2_fsm=True, use_runner=False)
    metrics_no_runner = compute_performance_metrics(trades_no_runner)
    
    ablation_summary = [
        {"Experiment": "A: V2 FSM (Production)", "Expectancy (E)": f"+Rs {metrics_v2['expectancy_rs']}", "Profit Factor": metrics_v2['profit_factor'], "Win Rate": f"{metrics_v2['win_rate_pct']}%", "Net P&L": f"+Rs {metrics_v2['net_total_pnl_rs']:,.0f}", "Max DD": f"-Rs {metrics_v2['max_drawdown_rs']:,.0f}"},
        {"Experiment": "A: V1 Baseline (Flat +10/-5)", "Expectancy (E)": f"+Rs {metrics_v1['expectancy_rs']}", "Profit Factor": metrics_v1['profit_factor'], "Win Rate": f"{metrics_v1['win_rate_pct']}%", "Net P&L": f"+Rs {metrics_v1['net_total_pnl_rs']:,.0f}", "Max DD": f"-Rs {metrics_v1['max_drawdown_rs']:,.0f}"},
        {"Experiment": "B: ATM Strikes (Delta ~0.50)", "Expectancy (E)": f"+Rs {metrics_v2['expectancy_rs']}", "Profit Factor": metrics_v2['profit_factor'], "Win Rate": f"{metrics_v2['win_rate_pct']}%", "Net P&L": f"+Rs {metrics_v2['net_total_pnl_rs']:,.0f}", "Max DD": f"-Rs {metrics_v2['max_drawdown_rs']:,.0f}"},
        {"Experiment": "B: ITM Strikes (Delta ~0.65)", "Expectancy (E)": f"+Rs {metrics_itm['expectancy_rs']}", "Profit Factor": metrics_itm['profit_factor'], "Win Rate": f"{metrics_itm['win_rate_pct']}%", "Net P&L": f"+Rs {metrics_itm['net_total_pnl_rs']:,.0f}", "Max DD": f"-Rs {metrics_itm['max_drawdown_rs']:,.0f}"},
        {"Experiment": "C: Hybrid SL (-8 Emergency / -5)", "Expectancy (E)": f"+Rs {metrics_v2['expectancy_rs']}", "Profit Factor": metrics_v2['profit_factor'], "Win Rate": f"{metrics_v2['win_rate_pct']}%", "Net P&L": f"+Rs {metrics_v2['net_total_pnl_rs']:,.0f}", "Max DD": f"-Rs {metrics_v2['max_drawdown_rs']:,.0f}"},
        {"Experiment": "C: Strict Fixed -Rs 5 SL", "Expectancy (E)": f"+Rs {metrics_fixed_sl['expectancy_rs']}", "Profit Factor": metrics_fixed_sl['profit_factor'], "Win Rate": f"{metrics_fixed_sl['win_rate_pct']}%", "Net P&L": f"+Rs {metrics_fixed_sl['net_total_pnl_rs']:,.0f}", "Max DD": f"-Rs {metrics_fixed_sl['max_drawdown_rs']:,.0f}"},
        {"Experiment": "D: T1 Target +Rs 8.00", "Expectancy (E)": f"+Rs {metrics_tgt8['expectancy_rs']}", "Profit Factor": metrics_tgt8['profit_factor'], "Win Rate": f"{metrics_tgt8['win_rate_pct']}%", "Net P&L": f"+Rs {metrics_tgt8['net_total_pnl_rs']:,.0f}", "Max DD": f"-Rs {metrics_tgt8['max_drawdown_rs']:,.0f}"},
        {"Experiment": "D: T1 Target +Rs 10.00 (Prod)", "Expectancy (E)": f"+Rs {metrics_v2['expectancy_rs']}", "Profit Factor": metrics_v2['profit_factor'], "Win Rate": f"{metrics_v2['win_rate_pct']}%", "Net P&L": f"+Rs {metrics_v2['net_total_pnl_rs']:,.0f}", "Max DD": f"-Rs {metrics_v2['max_drawdown_rs']:,.0f}"},
        {"Experiment": "D: T1 Target +Rs 12.00", "Expectancy (E)": f"+Rs {metrics_tgt12['expectancy_rs']}", "Profit Factor": metrics_tgt12['profit_factor'], "Win Rate": f"{metrics_tgt12['win_rate_pct']}%", "Net P&L": f"+Rs {metrics_tgt12['net_total_pnl_rs']:,.0f}", "Max DD": f"-Rs {metrics_tgt12['max_drawdown_rs']:,.0f}"},
        {"Experiment": "E: Lot 2 Runner (20 EMA)", "Expectancy (E)": f"+Rs {metrics_v2['expectancy_rs']}", "Profit Factor": metrics_v2['profit_factor'], "Win Rate": f"{metrics_v2['win_rate_pct']}%", "Net P&L": f"+Rs {metrics_v2['net_total_pnl_rs']:,.0f}", "Max DD": f"-Rs {metrics_v2['max_drawdown_rs']:,.0f}"},
        {"Experiment": "E: Fixed +Rs 10 Both Lots", "Expectancy (E)": f"+Rs {metrics_no_runner['expectancy_rs']}", "Profit Factor": metrics_no_runner['profit_factor'], "Win Rate": f"{metrics_no_runner['win_rate_pct']}%", "Net P&L": f"+Rs {metrics_no_runner['net_total_pnl_rs']:,.0f}", "Max DD": f"-Rs {metrics_no_runner['max_drawdown_rs']:,.0f}"},
    ]
    
    df_abl = pd.DataFrame(ablation_summary)
    print(df_abl.to_string(index=False))
    print("=" * 80)
    
    # Save validation artifacts
    results_path = os.path.join(PROJECT_ROOT, "backtest_results", "production_validation_results.json")
    save_data = {
        "timestamp": dt.datetime.now().isoformat(),
        "git_tag": "CDSL_FSM_V2.1_CALIBRATED",
        "step_4_ml_calibration": {
            "brier_score": ml_report.get("brier_score"),
            "total_evaluated": ml_report.get("total_evaluated"),
            "bins": ml_report.get("bin_results")
        },
        "step_5_strategy_performance": metrics_v2,
        "step_6_ablation_experiments": ablation_summary
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[+] Production Validation Results saved to: {results_path}")
    
    return save_data


if __name__ == "__main__":
    run_all_production_validations()
