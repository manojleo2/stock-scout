# -*- coding: utf-8 -*-
"""
=============================================================================
 Stock Scout — Supercharged AI Up/Down Forecast Predictor Backtest
=============================================================================
 PURPOSE:
   Walk-forward quantitative verification of the Supercharged AI Directional
   Predictor (Close-to-Close) vs Baseline over 197+ historical trading sessions.

 PILLARS EVALUATED:
   1. 28-Feature Institutional Alpha & Flow Dataset (Alpha vs Nifty, BPI, Trend)
   2. Recency Exponential Sample Weighting (prioritizes recent market regime)
   3. Conviction Gatekeeper (>= 60% threshold for swing / breakout entries)
   4. Dynamic ATR Take-Profit (1.5x) and Stop-Loss (1.0x) Simulation

 RUN:
   python -m utils.backtest_supercharged_ai
=============================================================================
"""

import sys
import os
import datetime as dt

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
from config import (
    BENCHMARK_TICKER, ML_MAX_DEPTH, ML_MIN_SAMPLES_LEAF, ML_N_ESTIMATORS,
    AI_MIN_CONVICTION_THRESHOLD, AI_ATR_SL_MULT, AI_ATR_TP1_MULT
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("ai_backtest")

BASE_FEATURE_COLS = [
    'Ret_1', 'Ret_5', 'Ret_20',
    'Dist_SMA50', 'Dist_SMA200', 'EMA_Cross_Ratio',
    'RSI_Norm', 'MACD_Hist_Norm', 'BB_PctB', 'BB_Width',
    'Vol_Surge', 'Nifty_Ret1', 'Nifty_Dist_SMA50',
    'SP500_Ret1', 'Nasdaq_Ret1', 'VIX_Norm', 'VIX_Ret1',
    'BankNifty_Ret1', 'Ret_VIX_Interact', 'News_Sentiment'
]

INSTITUTIONAL_FEATURE_COLS = BASE_FEATURE_COLS + [
    'Alpha_Nifty_5d', 'Alpha_Nifty_20d', 'Buying_Pressure_Index',
    'Trend_Convergence_Score', 'Bollinger_Z_Score',
    'ATR_Normalized_Ratio', 'Intraday_Efficiency_Ratio',
    'Consecutive_Direction_Streak'
]

def build_feature_matrix(symbol: str, period: str = "2y"):
    stock_df = get_stock_data(symbol, period=period)
    if stock_df.empty or len(stock_df) < 100:
        return None

    df = calculate_technical_indicators(stock_df)

    nifty_df = get_stock_data(BENCHMARK_TICKER, period=period)
    if not nifty_df.empty:
        nifty_df.index = nifty_df.index.tz_localize(None) if nifty_df.index.tz is not None else nifty_df.index
        df['Nifty_Ret1'] = nifty_df['Close'].pct_change(1)
        df['Nifty_Ret5'] = nifty_df['Close'].pct_change(5)
        df['Nifty_Ret20'] = nifty_df['Close'].pct_change(20)
        df['Nifty_Dist_SMA50'] = (nifty_df['Close'] / nifty_df['Close'].rolling(50).mean()) - 1.0
    else:
        df['Nifty_Ret1'] = 0.0
        df['Nifty_Ret5'] = 0.0
        df['Nifty_Ret20'] = 0.0
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

    # 8 Institutional Upgrades
    df['Alpha_Nifty_5d'] = (df['Ret_5'] - df['Nifty_Ret5']).fillna(0.0)
    df['Alpha_Nifty_20d'] = (df['Ret_20'] - df['Nifty_Ret20']).fillna(0.0)

    day_range = (df['High'] - df['Low']).replace(0, np.nan)
    df['Buying_Pressure_Index'] = (((df['Close'] - df['Low']) / day_range).fillna(0.5) * df['Vol_Surge']).fillna(1.0)

    sig_sma50 = np.sign(df['Close'] - df['SMA_50'])
    sig_ema = np.sign(df['EMA_12'] - df['EMA_26'])
    sig_sma200 = np.sign(df['Close'] - df['SMA_200'])
    df['Trend_Convergence_Score'] = ((sig_sma50 + sig_ema + sig_sma200) / 3.0).fillna(0.0)

    rolling_mean_20 = df['Close'].rolling(20).mean()
    rolling_std_20 = df['Close'].rolling(20).std().replace(0, np.nan)
    df['Bollinger_Z_Score'] = ((df['Close'] - rolling_mean_20) / rolling_std_20).fillna(0.0)

    df['ATR_Normalized_Ratio'] = (df['ATR_14'] / df['Close']).fillna(0.0)
    df['Intraday_Efficiency_Ratio'] = ((df['Close'] - df['Open']).abs() / day_range).fillna(0.5)

    direction_sign = np.where(df['Close'] > df['Close'].shift(1), 1, np.where(df['Close'] < df['Close'].shift(1), -1, 0))
    streak = np.zeros(len(df))
    for i in range(1, len(df)):
        if direction_sign[i] == direction_sign[i-1] and direction_sign[i] != 0:
            streak[i] = streak[i-1] + direction_sign[i]
        else:
            streak[i] = direction_sign[i]
    df['Consecutive_Direction_Streak'] = streak

    # Target: Close[t+1] > Close[t]
    next_close = df['Close'].shift(-1)
    df['Target'] = (next_close > df['Close']).astype(int)
    df['Actual_Day_Change_Pct'] = ((next_close - df['Close']) / df['Close']) * 100.0
    df['Actual_Day_Points'] = next_close - df['Close']

    return df

def run_ai_walk_forward_backtest(symbol: str, df: pd.DataFrame, min_train_days: int = 250):
    clean_df = df.dropna(subset=INSTITUTIONAL_FEATURE_COLS + ['Target']).copy()
    if len(clean_df) < min_train_days + 20:
        return pd.DataFrame()

    results = []
    total_steps = len(clean_df) - min_train_days

    for i in range(min_train_days, len(clean_df)):
        current_date = clean_df.index[i]
        train_data = clean_df.iloc[:i]
        test_row = clean_df.iloc[i]
        y_train = train_data['Target']

        close_p = float(test_row['Close'])
        actual_target = int(test_row['Target'])
        day_points = float(test_row['Actual_Day_Points'])
        day_pct = float(test_row['Actual_Day_Change_Pct'])
        atr_14 = float(test_row['ATR_14']) if 'ATR_14' in test_row else (close_p * 0.015)

        # 1. Baseline Model (20 features, unweighted, always trade)
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
        conv_base = max(prob_base, 1.0 - prob_base)
        dir_base = 1 if prob_base >= 0.50 else 0
        hit_base = (dir_base == actual_target)

        # 2. Supercharged AI Model (28 features, recency weighted, >= 60% conviction)
        X_train_spec = train_data[INSTITUTIONAL_FEATURE_COLS]
        X_test_spec = pd.DataFrame([test_row[INSTITUTIONAL_FEATURE_COLS]])
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
        conv_spec = max(prob_spec, 1.0 - prob_spec)

        trade_spec = conv_spec >= (AI_MIN_CONVICTION_THRESHOLD / 100.0)
        dir_spec = 1 if prob_spec >= 0.50 else 0
        hit_spec = (dir_spec == actual_target) if trade_spec else None

        # Simulated Equity Swing P&L with 1.5x ATR TP and 1.0x ATR SL
        sl_points = atr_14 * AI_ATR_SL_MULT
        tp_points = atr_14 * AI_ATR_TP1_MULT

        # Baseline P&L (always holds 1 day)
        pnl_base_pts = day_points if dir_base == 1 else -day_points

        # Specialist P&L (filtered with SL/TP bounds)
        if trade_spec:
            raw_pts = day_points if dir_spec == 1 else -day_points
            # Clip between Stop-Loss (-1.0x ATR) and Target (+1.5x ATR)
            pnl_spec_pts = np.clip(raw_pts, -sl_points, tp_points)
        else:
            pnl_spec_pts = 0.0

        results.append({
            'date': current_date,
            'symbol': symbol,
            'close': close_p,
            'day_points': day_points,
            'day_pct': day_pct,
            'actual_target': actual_target,
            'prob_base': round(prob_base, 4),
            'conv_base': round(conv_base, 4),
            'hit_base': hit_base,
            'pnl_base_pts': round(pnl_base_pts, 2),
            'prob_spec': round(prob_spec, 4),
            'conv_spec': round(conv_spec, 4),
            'trade_spec': trade_spec,
            'hit_spec': hit_spec,
            'pnl_spec_pts': round(pnl_spec_pts, 2)
        })

    return pd.DataFrame(results)

def main():
    print("=" * 88)
    print("  Stock Scout — Supercharged AI Up/Down Forecast Predictor Backtest")
    print(f"  Target: Next-Day Close-to-Close | Min Conviction: {AI_MIN_CONVICTION_THRESHOLD}%")
    print(f"  Timestamp: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 88)

    symbols = ["CDSL.NS", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    all_res = []

    for sym in symbols:
        print(f"\n  Building 28-feature matrix for {sym}...", end=" ", flush=True)
        df = build_feature_matrix(sym, period="2y")
        if df is not None:
            print(f"OK ({len(df)} rows)")
            print(f"  Running Walk-Forward Backtest for {sym}...", end=" ", flush=True)
            res = run_ai_walk_forward_backtest(sym, df, min_train_days=250)
            if len(res) > 0:
                all_res.append(res)
                total = len(res)
                base_acc = res['hit_base'].mean() * 100
                spec_trades = res[res['trade_spec'] == True]
                spec_acc = (spec_trades['hit_spec'].sum() / len(spec_trades) * 100) if len(spec_trades) > 0 else 0
                print(f"DONE: Baseline = {base_acc:.1f}% ({total} days) | Supercharged = {spec_acc:.1f}% ({len(spec_trades)} trades, {total-len(spec_trades)} chop days filtered)")
        else:
            print("FAILED")

    if not all_res:
        return

    combined = pd.concat(all_res, ignore_index=True)

    print("\n" + "=" * 88)
    print(" [COMBINED AI UP/DOWN FORECAST PREDICTOR SCORECARD - Across All Tested Stocks]")
    print("=" * 88)

    total_days = len(combined)
    base_acc = combined['hit_base'].mean() * 100
    spec_trades = combined[combined['trade_spec'] == True]
    spec_acc = (spec_trades['hit_spec'].sum() / len(spec_trades) * 100) if len(spec_trades) > 0 else 0
    chop_days = total_days - len(spec_trades)

    print(f"\n  Total Evaluated Sessions: {total_days}")
    print(f"  Chop / Low-Conviction Days Filtered: {chop_days} ({chop_days/total_days*100:.1f}% capital preserved)")
    print(f"  Baseline Accuracy (All Days, Unweighted): {base_acc:.1f}%")
    print(f"  Supercharged AI Accuracy (>=60% Conviction): {spec_acc:.1f}% (Outperformance: {spec_acc - base_acc:+.1f}%)")

    # Save to backtest_results
    out_dir = os.path.join(PROJECT_ROOT, "backtest_results")
    os.makedirs(out_dir, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"ai_predictor_backtest_{ts}.csv")
    combined.to_csv(out_file, index=False)
    print(f"\n  Detailed results saved to: {out_file}")
    print("=" * 88 + "\n")

if __name__ == "__main__":
    main()
