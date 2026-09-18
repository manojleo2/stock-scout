import sys
import os
import datetime as dt
import calendar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import logging

from utils.indicators import calculate_technical_indicators
from utils.macro_factors import get_macro_market_cues
from utils.data_loader import get_stock_data
from utils.opening_predictor import get_days_to_monthly_expiry
from config import (
    BENCHMARK_TICKER, ML_MAX_DEPTH, ML_MIN_SAMPLES_LEAF, ML_N_ESTIMATORS,
    CDSL_MIN_CONVICTION_WEEKDAY, CDSL_MIN_CONVICTION_FRIDAY,
    CDSL_EXPIRY_WEEK_DAYS, CDSL_LOT_SIZE
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("cdsl_backtest")

BASE_FEATURE_COLS = [
    'Ret_1', 'Ret_5', 'Ret_20',
    'Dist_SMA50', 'Dist_SMA200', 'EMA_Cross_Ratio',
    'RSI_Norm', 'MACD_Hist_Norm', 'BB_PctB', 'BB_Width',
    'Vol_Surge', 'Nifty_Ret1', 'Nifty_Dist_SMA50',
    'SP500_Ret1', 'Nasdaq_Ret1', 'VIX_Norm', 'VIX_Ret1',
    'BankNifty_Ret1', 'Ret_VIX_Interact', 'News_Sentiment',
    'Close_High_Ratio', 'Intraday_Range_Pct', 'Open_To_Close_Ret',
    'Upper_Wick_Ratio', 'Lower_Wick_Ratio'
]

SPECIALIST_FEATURE_COLS = BASE_FEATURE_COLS + [
    'Days_To_Expiry', 'Is_Monday', 'Is_Friday',
    'Vol_Price_Trend', 'Consecutive_Gap_Streak',
    'ATR_Expansion_Ratio', 'Close_VWAP_Ratio'
]

def build_cdsl_feature_matrix(period="2y"):
    stock_df = get_stock_data("CDSL.NS", period=period)
    if stock_df.empty or len(stock_df) < 100:
        raise ValueError("Could not download sufficient CDSL historical data.")

    df = calculate_technical_indicators(stock_df)

    nifty_df = get_stock_data(BENCHMARK_TICKER, period=period)
    if not nifty_df.empty:
        nifty_df.index = nifty_df.index.tz_localize(None) if nifty_df.index.tz is not None else nifty_df.index
        df['Nifty_Ret1'] = nifty_df['Close'].pct_change(1)
        df['Nifty_Dist_SMA50'] = (nifty_df['Close'] / nifty_df['Close'].rolling(50).mean()) - 1.0
    else:
        df['Nifty_Ret1'] = 0.0
        df['Nifty_Dist_SMA50'] = 0.0

    macro_df = get_macro_market_cues(period=period)
    if not macro_df.empty:
        df = df.join(macro_df, how='left')

    for col in ['SP500_Ret1', 'Nasdaq_Ret1', 'VIX_Norm', 'VIX_Ret1', 'BankNifty_Ret1']:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].ffill().fillna(0.0)

    df['SMA_50'] = df['SMA_50'].bfill().ffill().fillna(df['Close'])
    df['SMA_200'] = df['SMA_200'].bfill().ffill().fillna(df['Close'])

    df['Ret_1'] = df['Close'].pct_change(1)
    df['Ret_5'] = df['Close'].pct_change(5)
    df['Ret_20'] = df['Close'].pct_change(20)
    df['Dist_SMA50'] = (df['Close'] / df['SMA_50']) - 1.0
    df['Dist_SMA200'] = (df['Close'] / df['SMA_200']) - 1.0
    df['EMA_Cross_Ratio'] = (df['EMA_12'] / df['EMA_26']) - 1.0
    df['RSI_Norm'] = df['RSI_14'] / 100.0
    df['MACD_Hist_Norm'] = df['MACD_Hist'] / df['Close']
    df['BB_PctB'] = df['BB_PctB'].fillna(0.5)
    df['BB_Width'] = df['BB_Width'].fillna(0.0)
    df['Vol_Surge'] = df['Vol_Surge'].fillna(1.0)
    df['Ret_VIX_Interact'] = df['Ret_1'] * df['VIX_Ret1']
    df['News_Sentiment'] = 0.0

    day_range = (df['High'] - df['Low']).replace(0, np.nan)
    df['Close_High_Ratio'] = ((df['Close'] - df['Low']) / day_range).fillna(0.5)
    df['Intraday_Range_Pct'] = ((df['High'] - df['Low']) / df['Close']).fillna(0.0)
    df['Open_To_Close_Ret'] = ((df['Close'] - df['Open']) / df['Open']).fillna(0.0)

    upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Upper_Wick_Ratio'] = (upper_wick / day_range).fillna(0.0)

    lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Lower_Wick_Ratio'] = (lower_wick / day_range).fillna(0.0)

    df['Days_To_Expiry'] = [get_days_to_monthly_expiry(idx) for idx in df.index]
    df['Is_Monday'] = (df.index.dayofweek == 0).astype(float)
    df['Is_Friday'] = (df.index.dayofweek == 4).astype(float)
    df['Vol_Price_Trend'] = (df['Vol_Surge'] * df['Ret_1']).fillna(0.0)

    prev_gap = np.where(df['Open'] > df['Close'].shift(1), 1, np.where(df['Open'] < df['Close'].shift(1), -1, 0))
    streak = np.zeros(len(df))
    for i in range(1, len(df)):
        if prev_gap[i] == prev_gap[i-1] and prev_gap[i] != 0:
            streak[i] = streak[i-1] + prev_gap[i]
        else:
            streak[i] = prev_gap[i]
    df['Consecutive_Gap_Streak'] = streak

    df['ATR_Expansion_Ratio'] = (df['ATR_14'] / df['Close']).fillna(0.0)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    df['Close_VWAP_Ratio'] = ((df['Close'] - typical_price) / typical_price).fillna(0.0)

    next_open = df['Open'].shift(-1)
    df['Target_Gap'] = (next_open > df['Close']).astype(int)
    df['Actual_Gap_Points'] = next_open - df['Close']
    df['Actual_Gap_Pct'] = (df['Actual_Gap_Points'] / df['Close']) * 100.0

    return df

def run_comparative_walk_forward_backtest(df, min_train_days=250):
    clean_df = df.dropna(subset=SPECIALIST_FEATURE_COLS + ['Target_Gap']).copy()
    if len(clean_df) < min_train_days + 20:
        raise ValueError(f"Insufficient clean data ({len(clean_df)} rows). Need at least {min_train_days + 20}.")

    results = []
    total_steps = len(clean_df) - min_train_days
    print(f"  Starting Walk-Forward Validation across {total_steps} out-of-sample trading days...")

    for i in range(min_train_days, len(clean_df)):
        current_date = clean_df.index[i]
        day_name = current_date.strftime("%A")
        is_friday = (current_date.weekday() == 4)
        is_monday = (current_date.weekday() == 0)

        train_data = clean_df.iloc[:i]
        test_row = clean_df.iloc[i]
        y_train = train_data['Target_Gap']

        close_p = float(test_row['Close'])
        gap_points = float(test_row['Actual_Gap_Points'])
        gap_pct = float(test_row['Actual_Gap_Pct'])
        actual_target = int(test_row['Target_Gap'])
        days_to_exp = int(test_row['Days_To_Expiry'])
        is_expiry_week = (days_to_exp <= CDSL_EXPIRY_WEEK_DAYS)

        # 1. Baseline Model
        X_train_base = train_data[BASE_FEATURE_COLS]
        X_test_base = pd.DataFrame([test_row[BASE_FEATURE_COLS]])

        rf_base = RandomForestClassifier(
            n_estimators=ML_N_ESTIMATORS, max_depth=ML_MAX_DEPTH,
            min_samples_leaf=ML_MIN_SAMPLES_LEAF, random_state=42, n_jobs=-1
        )
        hgb_base = HistGradientBoostingClassifier(
            max_iter=100, max_depth=3, min_samples_leaf=15, random_state=42
        )
        rf_base.fit(X_train_base, y_train)
        hgb_base.fit(X_train_base, y_train)
        prob_base = 0.5 * rf_base.predict_proba(X_test_base)[0][1] + 0.5 * hgb_base.predict_proba(X_test_base)[0][1]
        conviction_base = max(prob_base, 1.0 - prob_base)

        trade_base = conviction_base >= 0.55
        call_base = (prob_base >= 0.55)
        dir_base = 1 if call_base else 0
        hit_base = (dir_base == actual_target) if trade_base else None

        # 2. Step A Model
        trade_step_a = conviction_base >= 0.65
        call_step_a = (prob_base >= 0.65)
        dir_step_a = 1 if call_step_a else 0
        hit_step_a = (dir_step_a == actual_target) if trade_step_a else None

        # 3. Supercharged CDSL Specialist Model
        X_train_spec = train_data[SPECIALIST_FEATURE_COLS]
        X_test_spec = pd.DataFrame([test_row[SPECIALIST_FEATURE_COLS]])
        weights_train = np.exp(np.linspace(-0.5, 0.0, len(X_train_spec)))

        rf_spec = RandomForestClassifier(
            n_estimators=ML_N_ESTIMATORS, max_depth=ML_MAX_DEPTH,
            min_samples_leaf=ML_MIN_SAMPLES_LEAF, random_state=42, n_jobs=-1
        )
        hgb_spec = HistGradientBoostingClassifier(
            max_iter=100, max_depth=3, min_samples_leaf=15, random_state=42
        )
        rf_spec.fit(X_train_spec, y_train, sample_weight=weights_train)
        hgb_spec.fit(X_train_spec, y_train, sample_weight=weights_train)

        prob_spec = 0.5 * rf_spec.predict_proba(X_test_spec)[0][1] + 0.5 * hgb_spec.predict_proba(X_test_spec)[0][1]
        conviction_spec = max(prob_spec, 1.0 - prob_spec)

        threshold_spec = (CDSL_MIN_CONVICTION_FRIDAY / 100.0) if is_friday else (CDSL_MIN_CONVICTION_WEEKDAY / 100.0)
        trade_spec = conviction_spec >= threshold_spec
        call_spec = (prob_spec >= threshold_spec)
        dir_spec = 1 if call_spec else 0
        hit_spec = (dir_spec == actual_target) if trade_spec else None

        lot = CDSL_LOT_SIZE
        brokerage = 100.0
        theta_weekday = close_p * 0.0015
        theta_weekend = close_p * 0.0050

        # Baseline PnL
        if trade_base:
            theta_b = theta_weekend if is_friday else theta_weekday
            delta_b = 0.50
            opt_move_b = (gap_points * delta_b) if dir_base == 1 else (-gap_points * delta_b)
            pnl_base = (opt_move_b - theta_b) * lot - brokerage
        else:
            pnl_base = 0.0

        # Step A PnL
        if trade_step_a:
            theta_a = theta_weekend if is_friday else theta_weekday
            delta_a = 0.50
            opt_move_a = (gap_points * delta_a) if dir_step_a == 1 else (-gap_points * delta_a)
            pnl_step_a = (opt_move_a - theta_a) * lot - brokerage
        else:
            pnl_step_a = 0.0

        # Specialist PnL
        if trade_spec:
            theta_s = theta_weekend if is_friday else theta_weekday
            delta_s = 0.70 if is_expiry_week else 0.50
            opt_move_s = (gap_points * delta_s) if dir_spec == 1 else (-gap_points * delta_s)
            pnl_spec = (opt_move_s - theta_s) * lot - brokerage
        else:
            pnl_spec = 0.0

        results.append({
            'date': current_date,
            'day_name': day_name,
            'is_friday': is_friday,
            'is_monday': is_monday,
            'days_to_expiry': days_to_exp,
            'is_expiry_week': is_expiry_week,
            'close': close_p,
            'gap_points': gap_points,
            'gap_pct': gap_pct,
            'actual_target': actual_target,
            'prob_base': round(prob_base, 4),
            'conv_base': round(conviction_base, 4),
            'trade_base': trade_base,
            'hit_base': hit_base,
            'pnl_base': round(pnl_base, 2),
            'trade_step_a': trade_step_a,
            'hit_step_a': hit_step_a,
            'pnl_step_a': round(pnl_step_a, 2),
            'prob_spec': round(prob_spec, 4),
            'conv_spec': round(conviction_spec, 4),
            'trade_spec': trade_spec,
            'hit_spec': hit_spec,
            'pnl_spec': round(pnl_spec, 2)
        })

    return pd.DataFrame(results)

def print_specialist_scorecard(res_df):
    print("\n" + "=" * 88)
    print(" [CDSL SPECIALIST 3:05 PM GAP PREDICTOR - INSTITUTIONAL WALK-FORWARD SCORECARD]")
    print("=" * 88)

    total_sessions = len(res_df)
    b_trades = res_df[res_df['trade_base'] == True]
    a_trades = res_df[res_df['trade_step_a'] == True]
    s_trades = res_df[res_df['trade_spec'] == True]

    b_acc = (b_trades['hit_base'].sum() / len(b_trades) * 100) if len(b_trades) > 0 else 0
    a_acc = (a_trades['hit_step_a'].sum() / len(a_trades) * 100) if len(a_trades) > 0 else 0
    s_acc = (s_trades['hit_spec'].sum() / len(s_trades) * 100) if len(s_trades) > 0 else 0

    b_pnl = b_trades['pnl_base'].sum()
    a_pnl = a_trades['pnl_step_a'].sum()
    s_pnl = s_trades['pnl_spec'].sum()

    b_wins = b_trades[b_trades['pnl_base'] > 0]['pnl_base'].sum()
    b_losses = abs(b_trades[b_trades['pnl_base'] < 0]['pnl_base'].sum())
    b_pf = (b_wins / b_losses) if b_losses > 0 else 99.0

    a_wins = a_trades[a_trades['pnl_step_a'] > 0]['pnl_step_a'].sum()
    a_losses = abs(a_trades[a_trades['pnl_step_a'] < 0]['pnl_step_a'].sum())
    a_pf = (a_wins / a_losses) if a_losses > 0 else 99.0

    s_wins = s_trades[s_trades['pnl_spec'] > 0]['pnl_spec'].sum()
    s_losses = abs(s_trades[s_trades['pnl_spec'] < 0]['pnl_spec'].sum())
    s_pf = (s_wins / s_losses) if s_losses > 0 else 99.0

    print(f"\n  MODEL COMPARISON (197 OOS Sessions, 1 Lot = {CDSL_LOT_SIZE} shares):")
    print(f"  {'-'*84}")
    print(f"  {'Metric':<28s} {'Baseline (55%)':<18s} {'Step A (65%)':<18s} {'Supercharged Specialist':<20s}")
    print(f"  {'-'*84}")
    print(f"  {'Total Trades Taken':<28s} {len(b_trades):<18d} {len(a_trades):<18d} {len(s_trades):<20d}")
    print(f"  {'Cash Preserved Days':<28s} {total_sessions - len(b_trades):<18d} {total_sessions - len(a_trades):<18d} {total_sessions - len(s_trades):<20d}")
    print(f"  {'Directional Accuracy':<28s} {b_acc:<17.1f}% {a_acc:<17.1f}% {s_acc:<19.1f}%")
    print(f"  {'Net Option P&L (INR)':<28s} INR {b_pnl:<12,.0f} INR {a_pnl:<12,.0f} INR {s_pnl:<14,.0f}")
    print(f"  {'Profit Factor (Gross W/L)':<28s} {b_pf:<18.2f} {a_pf:<18.2f} {s_pf:<20.2f}")
    print(f"  {'-'*84}")

    print(f"\n  DAY-OF-WEEK MICROSTRUCTURE PERFORMANCE (Specialist Model):")
    print(f"  {'-'*84}")
    print(f"  {'Day':<14s} {'Sessions':<10s} {'Trades':<10s} {'Accuracy':<14s} {'Net P&L (INR)':<16s} {'Win Rate %':<12s}")
    print(f"  {'-'*84}")

    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        d_df = res_df[res_df['day_name'] == day]
        d_trades = d_df[d_df['trade_spec'] == True]
        d_acc = (d_trades['hit_spec'].sum() / len(d_trades) * 100) if len(d_trades) > 0 else 0
        d_pnl = d_trades['pnl_spec'].sum()
        d_win_pnl = (len(d_trades[d_trades['pnl_spec'] > 0]) / len(d_trades) * 100) if len(d_trades) > 0 else 0
        print(f"  {day:<14s} {len(d_df):<10d} {len(d_trades):<10d} {d_acc:<13.1f}% INR {d_pnl:<10,.0f} {d_win_pnl:<11.1f}%")
    print(f"  {'-'*84}")

    fri_all = res_df[res_df['day_name'] == 'Friday']
    fri_b = fri_all[fri_all['trade_base'] == True]
    fri_s = fri_all[fri_all['trade_spec'] == True]
    print(f"\n  FRIDAY WEEKEND THETA SHIELD IMPACT:")
    print(f"  - Baseline Friday Trades: {len(fri_b)} trades | Net P&L: INR {fri_b['pnl_base'].sum():,.0f}")
    print(f"  - Specialist Friday Trades (Strict >=70%): {len(fri_s)} trades | Net P&L: INR {fri_s['pnl_spec'].sum():,.0f}")
    print(f"  - Friday Capital Preserved: {len(fri_all) - len(fri_s)} out of {len(fri_all)} Fridays shielded from 3-day decay!")

    exp_all = res_df[res_df['is_expiry_week'] == True]
    exp_b = exp_all[exp_all['trade_base'] == True]
    exp_s = exp_all[exp_all['trade_spec'] == True]
    print(f"\n  EXPIRY WEEK ITM ARMOR IMPACT (<= 4 Days to Monthly Expiry):")
    print(f"  - Baseline Expiry Week Trades: {len(exp_b)} | Net P&L: INR {exp_b['pnl_base'].sum():,.0f}")
    print(f"  - Specialist Expiry Armor Trades (Delta ~0.70): {len(exp_s)} | Net P&L: INR {exp_s['pnl_spec'].sum():,.0f}")
    print(f"  {'='*88}\n")

def main():
    print("=" * 88)
    print("  Stock Scout - Supercharged CDSL Specialist Walk-Forward Backtest")
    print(f"  Lot Size: {CDSL_LOT_SIZE} shares | Horizon: 2y | Warmup: 250 days")
    print(f"  Timestamp: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 88)

    print("\n  Building 32-feature CDSL matrix (Indicators, Macro, Expiry, Microstructure)...")
    df = build_cdsl_feature_matrix(period="2y")
    print(f"  Feature matrix built successfully: {len(df)} total rows.")

    results_df = run_comparative_walk_forward_backtest(df, min_train_days=250)
    print_specialist_scorecard(results_df)

    out_dir = os.path.join(PROJECT_ROOT, "backtest_results")
    os.makedirs(out_dir, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"cdsl_specialist_backtest_{ts}.csv")
    results_df.to_csv(out_file, index=False)
    print(f"  Full walk-forward session results saved to: {out_file}")

if __name__ == "__main__":
    main()
