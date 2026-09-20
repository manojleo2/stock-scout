"""
=============================================================================
 Stock Scout — Multi-Stock Gap Predictor Backtest
=============================================================================
 PURPOSE:
   Answer the key question: "If I scan 8 stocks daily and only trade the
   ones above a certain conviction threshold, does the win rate improve
   compared to trading CDSL every day?"

 METHODOLOGY:
   - Walk-forward backtest: train on the first N days, predict day N+1,
     then slide the window forward by 1 day.
   - Uses the EXACT same features & model as opening_predictor.py
   - Tests conviction thresholds: 55%, 60%, 65%, 70%, 75%, 80%
   - Compares: (A) Always trade CDSL vs (B) Pick the best stock each day

 RUN:
   cd c:\\Users\\manoj\\scout\\stock_scout
   python -m utils.backtest_scanner
=============================================================================
"""

import sys
import os

# Ensure project root is on sys.path so imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import logging
from datetime import datetime

# We import the feature engineering functions but NOT the full predict pipeline
# (to avoid live news sentiment / audit calls during backtesting)
from utils.indicators import calculate_technical_indicators
from utils.macro_factors import get_macro_market_cues
from utils.data_loader import get_stock_data
from config import BENCHMARK_TICKER, ML_TEST_SIZE, ML_MAX_DEPTH, ML_MIN_SAMPLES_LEAF, ML_N_ESTIMATORS

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("backtest")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Stocks to backtest (F&O eligible, liquid)
BACKTEST_STOCKS = [
    "CDSL.NS",
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "TATAMOTORS.NS",
    "SBIN.NS",
    "ITC.NS",
]

# Conviction thresholds to test
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# Walk-forward settings
DATA_PERIOD = "2y"
MIN_TRAIN_DAYS = 250       # ~1 year of training data before first prediction
WALK_FORWARD_STEP = 1      # Predict 1 day at a time


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (mirrors opening_predictor.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def build_gap_features(symbol: str, period: str = "2y"):
    """
    Build the full feature dataset for a stock, identical to
    opening_predictor.prepare_opening_gap_dataset() but without
    Streamlit caching or live news sentiment.
    """
    try:
        stock_df = get_stock_data(symbol, period=period)
    except Exception:
        stock_df = pd.DataFrame()

    if stock_df.empty or len(stock_df) < 100:
        return None

    df = calculate_technical_indicators(stock_df)

    # Benchmark (Nifty 50) momentum
    try:
        nifty_df = get_stock_data(BENCHMARK_TICKER, period=period)
    except Exception:
        nifty_df = pd.DataFrame()

    if not nifty_df.empty:
        nifty_df.index = nifty_df.index.tz_localize(None) if nifty_df.index.tz is not None else nifty_df.index
        df['Nifty_Ret1'] = nifty_df['Close'].pct_change(1)
        df['Nifty_Dist_SMA50'] = (nifty_df['Close'] / nifty_df['Close'].rolling(50).mean()) - 1.0
    else:
        df['Nifty_Ret1'] = 0.0
        df['Nifty_Dist_SMA50'] = 0.0

    # Macro cues (S&P 500, Nasdaq, VIX, Bank Nifty)
    try:
        macro_df = get_macro_market_cues(period=period)
    except Exception:
        macro_df = pd.DataFrame()

    if not macro_df.empty:
        df = df.join(macro_df, how='left')

    for col in ['SP500_Ret1', 'Nasdaq_Ret1', 'VIX_Norm', 'VIX_Ret1', 'BankNifty_Ret1']:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].ffill().fillna(0.0)

    # Fill moving average NaNs
    df['SMA_50'] = df['SMA_50'].bfill().ffill().fillna(df['Close'])
    df['SMA_200'] = df['SMA_200'].bfill().ffill().fillna(df['Close'])

    # Stationary features
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
    df['News_Sentiment'] = 0.0  # No live news in backtest

    # Intraday microstructure proxies
    day_range = (df['High'] - df['Low']).replace(0, np.nan)
    df['Close_High_Ratio'] = ((df['Close'] - df['Low']) / day_range).fillna(0.5)
    df['Intraday_Range_Pct'] = ((df['High'] - df['Low']) / df['Close']).fillna(0.0)
    df['Open_To_Close_Ret'] = ((df['Close'] - df['Open']) / df['Open']).fillna(0.0)

    upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Upper_Wick_Ratio'] = (upper_wick / day_range).fillna(0.0)

    lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Lower_Wick_Ratio'] = (lower_wick / day_range).fillna(0.0)

    # TARGET: Gap Up = 1 if Open[t+1] > Close[t]
    next_open = df['Open'].shift(-1)
    df['Target_Gap'] = (next_open > df['Close']).astype(int)

    # Also store the actual gap % for P&L estimation
    df['Actual_Gap_Pct'] = ((next_open - df['Close']) / df['Close'] * 100)

    return df


FEATURE_COLS = [
    'Ret_1', 'Ret_5', 'Ret_20',
    'Dist_SMA50', 'Dist_SMA200', 'EMA_Cross_Ratio',
    'RSI_Norm', 'MACD_Hist_Norm', 'BB_PctB', 'BB_Width',
    'Vol_Surge', 'Nifty_Ret1', 'Nifty_Dist_SMA50',
    'SP500_Ret1', 'Nasdaq_Ret1', 'VIX_Norm', 'VIX_Ret1',
    'BankNifty_Ret1', 'Ret_VIX_Interact', 'News_Sentiment',
    'Close_High_Ratio', 'Intraday_Range_Pct', 'Open_To_Close_Ret',
    'Upper_Wick_Ratio', 'Lower_Wick_Ratio'
]


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_backtest(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run a walk-forward backtest for a single stock.
    For each day from MIN_TRAIN_DAYS to len(df)-1:
      1. Train on all data up to day i
      2. Predict probability for day i
      3. Record prediction vs actual outcome
    Returns a DataFrame of daily predictions.
    """
    clean_df = df.dropna(subset=FEATURE_COLS + ['Target_Gap']).copy()

    if len(clean_df) < MIN_TRAIN_DAYS + 20:
        logger.warning(f"{symbol}: Only {len(clean_df)} clean rows, need {MIN_TRAIN_DAYS + 20}. Skipping.")
        return pd.DataFrame()

    results = []
    total_steps = len(clean_df) - MIN_TRAIN_DAYS
    last_pct = -1

    for i in range(MIN_TRAIN_DAYS, len(clean_df)):
        # Progress indicator
        pct = int((i - MIN_TRAIN_DAYS) / total_steps * 100)
        if pct % 20 == 0 and pct != last_pct:
            print(f"    {pct}%...", end=" ", flush=True)
            last_pct = pct

        train_data = clean_df.iloc[:i]
        test_row = clean_df.iloc[i]

        X_train = train_data[FEATURE_COLS]
        y_train = train_data['Target_Gap']

        X_test = pd.DataFrame([test_row[FEATURE_COLS]])

        try:
            rf = RandomForestClassifier(
                n_estimators=ML_N_ESTIMATORS,
                max_depth=ML_MAX_DEPTH,
                min_samples_leaf=ML_MIN_SAMPLES_LEAF,
                random_state=42, n_jobs=-1
            )
            hgb = HistGradientBoostingClassifier(
                max_iter=100, max_depth=3,
                min_samples_leaf=15, random_state=42
            )
            ensemble = VotingClassifier(
                estimators=[('rf', rf), ('hgb', hgb)],
                voting='soft'
            )
            ensemble.fit(X_train, y_train)

            prob_up = float(ensemble.predict_proba(X_test)[0][1])
        except Exception as e:
            continue

        actual_gap = int(test_row['Target_Gap'])
        actual_gap_pct = float(test_row.get('Actual_Gap_Pct', 0.0))
        predicted_direction = 1 if prob_up >= 0.50 else 0
        conviction = max(prob_up, 1 - prob_up)  # Distance from 50-50

        results.append({
            'date': clean_df.index[i],
            'symbol': symbol,
            'prob_up': round(prob_up, 4),
            'prob_down': round(1 - prob_up, 4),
            'conviction': round(conviction, 4),
            'predicted_direction': predicted_direction,
            'predicted_label': 'GAP UP' if predicted_direction == 1 else 'GAP DOWN',
            'actual_direction': actual_gap,
            'actual_label': 'GAP UP' if actual_gap == 1 else 'GAP DOWN',
            'actual_gap_pct': round(actual_gap_pct, 4),
            'is_correct': predicted_direction == actual_gap,
            'close_price': round(float(test_row['Close']), 2),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS & REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def analyze_threshold_performance(all_predictions: pd.DataFrame):
    """Analyze accuracy at different conviction thresholds."""

    print("\n" + "=" * 80)
    print("  BACKTEST RESULTS: Multi-Stock Gap Predictor Analysis")
    print("=" * 80)

    # ── Per-Stock Baseline (no threshold filtering) ──
    print("\n  SECTION 1: Per-Stock Accuracy (All Predictions)")
    print("  " + "-" * 55 + "\n")

    stock_stats = []
    for sym in all_predictions['symbol'].unique():
        sym_df = all_predictions[all_predictions['symbol'] == sym]
        acc = sym_df['is_correct'].mean() * 100
        total = len(sym_df)
        wins = sym_df['is_correct'].sum()
        stock_stats.append({
            'Stock': sym,
            'Total Days': total,
            'Correct': wins,
            'Accuracy %': f"{acc:.1f}%"
        })
        print(f"  {sym:<18s}  {wins}/{total} correct = {acc:.1f}%")

    # ── Threshold Analysis ──
    print("\n  SECTION 2: Accuracy at Different Conviction Levels")
    print("  " + "-" * 55 + "\n")

    print(f"  {'Threshold':<14s} {'Trades':<10s} {'Correct':<10s} {'Accuracy':<12s} {'Avg |Gap%|':<12s} {'Days w/ Signal':<16s}")
    print(f"  {'---':<14s} {'---':<10s} {'---':<10s} {'---':<12s} {'---':<12s} {'---':<16s}")

    threshold_results = []
    for threshold in THRESHOLDS:
        filtered = all_predictions[all_predictions['conviction'] >= threshold]
        if len(filtered) == 0:
            continue

        acc = filtered['is_correct'].mean() * 100
        avg_gap = filtered['actual_gap_pct'].abs().mean()
        unique_dates = filtered['date'].nunique()
        total_dates = all_predictions['date'].nunique()
        pct_days_with_signal = (unique_dates / total_dates * 100) if total_dates > 0 else 0

        threshold_results.append({
            'threshold': threshold,
            'trades': len(filtered),
            'correct': int(filtered['is_correct'].sum()),
            'accuracy': acc,
            'avg_gap_pct': avg_gap,
            'days_with_signal': unique_dates,
            'pct_days_active': pct_days_with_signal
        })

        print(f"  >= {threshold*100:.0f}%{'':<9s} {len(filtered):<10d} {int(filtered['is_correct'].sum()):<10d} {acc:<12.1f} {avg_gap:<12.2f} {unique_dates}/{total_dates} ({pct_days_with_signal:.0f}%)")

    # ── Single Stock (CDSL) vs Best-Pick Strategy ──
    print("\n  SECTION 3: CDSL-Only vs Multi-Stock Best Pick")
    print("  " + "-" * 55 + "\n")

    cdsl_df = all_predictions[all_predictions['symbol'] == 'CDSL.NS']

    for threshold in THRESHOLDS:
        # Strategy A: CDSL only, filtered by threshold
        cdsl_filtered = cdsl_df[cdsl_df['conviction'] >= threshold]

        # Strategy B: Best pick across all stocks (highest conviction per day)
        above_threshold = all_predictions[all_predictions['conviction'] >= threshold]
        if len(above_threshold) == 0:
            continue

        best_pick = above_threshold.loc[above_threshold.groupby('date')['conviction'].idxmax()]

        cdsl_acc = cdsl_filtered['is_correct'].mean() * 100 if len(cdsl_filtered) > 0 else 0
        best_acc = best_pick['is_correct'].mean() * 100 if len(best_pick) > 0 else 0

        cdsl_trades = len(cdsl_filtered)
        best_trades = len(best_pick)

        improvement = best_acc - cdsl_acc

        marker = "[+]" if improvement > 0 else ("[ ]" if improvement == 0 else "[-]")
        print(f"  Threshold >= {threshold*100:.0f}%:")
        print(f"    CDSL Only:        {cdsl_trades:>4d} trades -> {cdsl_acc:.1f}% accuracy")
        print(f"    Best of 8 Stocks: {best_trades:>4d} trades -> {best_acc:.1f}% accuracy")
        print(f"    {marker} Improvement: {improvement:+.1f}%\n")

    # ── Weekly Profitability Estimate ──
    print("\n  SECTION 4: Weekly Win-Day Distribution")
    print("  " + "-" * 55 + "\n")

    for threshold in [0.65, 0.70, 0.75]:
        above = all_predictions[all_predictions['conviction'] >= threshold]
        if len(above) == 0:
            continue

        best_daily = above.loc[above.groupby('date')['conviction'].idxmax()]
        best_daily = best_daily.copy()
        best_daily['week'] = best_daily['date'].dt.isocalendar().week.astype(int)
        best_daily['year'] = best_daily['date'].dt.year

        weekly = best_daily.groupby(['year', 'week']).agg(
            trading_days=('is_correct', 'count'),
            winning_days=('is_correct', 'sum')
        ).reset_index()

        weekly['win_rate'] = (weekly['winning_days'] / weekly['trading_days'] * 100).round(1)

        # Only count full weeks (3+ trading days with signals)
        full_weeks = weekly[weekly['trading_days'] >= 3]
        if len(full_weeks) == 0:
            continue

        avg_wins_per_week = full_weeks['winning_days'].mean()
        avg_days_per_week = full_weeks['trading_days'].mean()
        weeks_with_3plus_wins = len(full_weeks[full_weeks['winning_days'] >= 3])
        total_full_weeks = len(full_weeks)

        print(f"  Threshold >= {threshold*100:.0f}% (Best Pick strategy):")
        print(f"    Avg trading days per week:  {avg_days_per_week:.1f}")
        print(f"    Avg winning days per week:  {avg_wins_per_week:.1f}")
        print(f"    Weeks with 3+ winning days: {weeks_with_3plus_wins}/{total_full_weeks} ({weeks_with_3plus_wins/total_full_weeks*100:.0f}%)")
        print()

    return threshold_results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  Stock Scout — Walk-Forward Backtest Starting")
    print(f"   Stocks: {', '.join(BACKTEST_STOCKS)}")
    print(f"   Data Period: {DATA_PERIOD}")
    print(f"   Min Training Window: {MIN_TRAIN_DAYS} days")
    print(f"   Thresholds: {[f'{t*100:.0f}%' for t in THRESHOLDS]}")
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Step 1: Download & build features for all stocks
    print("\n  Downloading data and building features...")
    stock_datasets = {}
    for sym in BACKTEST_STOCKS:
        print(f"  -> {sym}...", end=" ", flush=True)
        df = build_gap_features(sym, period=DATA_PERIOD)
        if df is not None and len(df) > MIN_TRAIN_DAYS:
            stock_datasets[sym] = df
            print(f"OK ({len(df)} rows)")
        else:
            print(f"SKIP (insufficient data)")

    if not stock_datasets:
        print("\n  ERROR: No stocks had sufficient data. Exiting.")
        return

    # Step 2: Run walk-forward backtest for each stock
    print(f"\n  Running walk-forward backtest across {len(stock_datasets)} stocks...")
    print(f"  (This may take 10-30 minutes depending on your CPU)\n")

    all_predictions = []
    for sym, df in stock_datasets.items():
        print(f"  -> Backtesting {sym}...", end=" ", flush=True)
        results = walk_forward_backtest(sym, df)
        if len(results) > 0:
            all_predictions.append(results)
            wins = results['is_correct'].sum()
            total = len(results)
            print(f"  DONE: {total} predictions ({wins} correct = {wins/total*100:.1f}%)")
        else:
            print(f"  WARNING: No predictions generated")

    if not all_predictions:
        print("\n  ERROR: No predictions generated. Exiting.")
        return

    combined = pd.concat(all_predictions, ignore_index=True)

    # Step 3: Analyze results
    analyze_threshold_performance(combined)

    # Step 4: Save detailed results to CSV
    output_dir = os.path.join(PROJECT_ROOT, "backtest_results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"gap_backtest_{timestamp}.csv")
    combined.to_csv(csv_path, index=False)

    print(f"\n  Detailed results saved to: {csv_path}")
    print(f"\n  Backtest complete at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
