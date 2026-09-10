import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging

from utils.data_loader import get_stock_data
from utils.indicators import calculate_technical_indicators
from utils.macro_factors import get_macro_market_cues
from utils.news_sentiment import get_stock_news_sentiment_score
from config import BENCHMARK_TICKER, ML_TEST_SIZE, ML_MAX_DEPTH, ML_MIN_SAMPLES_LEAF, ML_N_ESTIMATORS, STOCK_NAME_MAP

logging.basicConfig(level=logging.INFO)

def prepare_opening_gap_dataset(symbol: str, period: str = "2y") -> tuple:
    """
    Construct stationary feature dataset targeting the next-morning opening gap:
    Target_Gap = 1 if Open[t+1] > Close[t] (GAP UP), else 0 (GAP DOWN/FLAT)
    """
    stock_df = get_stock_data(symbol, period=period)
    if stock_df.empty or len(stock_df) < 100:
        return None, None, None, "Insufficient stock historical data (need >= 100 trading days)."

    # Compute standard technical indicators
    df = calculate_technical_indicators(stock_df)

    # Benchmark momentum
    nifty_df = get_stock_data(BENCHMARK_TICKER, period=period)
    if not nifty_df.empty:
        nifty_df.index = nifty_df.index.tz_localize(None) if nifty_df.index.tz is not None else nifty_df.index
        df['Nifty_Ret1'] = nifty_df['Close'].pct_change(1)
        df['Nifty_Dist_SMA50'] = (nifty_df['Close'] / nifty_df['Close'].rolling(50).mean()) - 1.0
    else:
        df['Nifty_Ret1'] = 0.0
        df['Nifty_Dist_SMA50'] = 0.0

    # Macro & Overnight Cues
    macro_df = get_macro_market_cues(period=period)
    if not macro_df.empty:
        df = df.join(macro_df, how='left')

    for col in ['SP500_Ret1', 'Nasdaq_Ret1', 'VIX_Norm', 'VIX_Ret1', 'BankNifty_Ret1']:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].ffill().fillna(0.0)

    # Fill moving average NaNs safely
    df['SMA_50'] = df['SMA_50'].bfill().ffill().fillna(df['Close'])
    df['SMA_200'] = df['SMA_200'].bfill().ffill().fillna(df['Close'])

    # 1. Base Stationary Returns & Oscillator Features
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

    # 2. Specialized Overnight Gap Driving Features (Intraday Microstructure Proxies)
    # Feature 21: Close Position within Day's Range (0 = closed at Low, 1 = closed at High)
    day_range = (df['High'] - df['Low']).replace(0, np.nan)
    df['Close_High_Ratio'] = ((df['Close'] - df['Low']) / day_range).fillna(0.5)

    # Feature 22: Intraday Range as percentage of close (Volatile days produce larger overnight imbalances)
    df['Intraday_Range_Pct'] = ((df['High'] - df['Low']) / df['Close']).fillna(0.0)

    # Feature 23: Return from Open to Close (Day's trend strength)
    df['Open_To_Close_Ret'] = ((df['Close'] - df['Open']) / df['Open']).fillna(0.0)

    # Feature 24: Upper Wick Ratio (Strong rejection at high implies overnight gap-down risk)
    upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Upper_Wick_Ratio'] = (upper_wick / day_range).fillna(0.0)

    # Feature 25: Lower Wick Ratio (Strong defense at low implies overnight gap-up potential)
    lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Lower_Wick_Ratio'] = (lower_wick / day_range).fillna(0.0)

    # TARGET: Next day's 9:15 AM OPEN compared to today's 3:30 PM CLOSE
    # 1 if Open[t+1] > Close[t], else 0
    next_open = df['Open'].shift(-1)
    df['Target_Gap'] = (next_open > df['Close']).astype(int)

    feature_cols = [
        'Ret_1', 'Ret_5', 'Ret_20',
        'Dist_SMA50', 'Dist_SMA200', 'EMA_Cross_Ratio',
        'RSI_Norm', 'MACD_Hist_Norm', 'BB_PctB', 'BB_Width',
        'Vol_Surge', 'Nifty_Ret1', 'Nifty_Dist_SMA50',
        'SP500_Ret1', 'Nasdaq_Ret1', 'VIX_Norm', 'VIX_Ret1',
        'BankNifty_Ret1', 'Ret_VIX_Interact', 'News_Sentiment',
        'Close_High_Ratio', 'Intraday_Range_Pct', 'Open_To_Close_Ret',
        'Upper_Wick_Ratio', 'Lower_Wick_Ratio'
    ]

    clean_df = df.dropna(subset=feature_cols).copy()
    train_test_df = clean_df.iloc[:-1].dropna(subset=['Target_Gap'])

    return train_test_df, clean_df.iloc[-1], feature_cols, None

def generate_options_trading_call(symbol: str, current_price: float, prob_up: float, confidence: str) -> dict:
    """
    Generate an actionable Put/Call options trading recommendation for entry at 3:05-3:15 PM.
    """
    strike_round = 10 if current_price < 500 else (20 if current_price < 2000 else 50)
    atm_strike = round(current_price / strike_round) * strike_round
    
    if prob_up >= 65.0:
        action = "🟢 BUY CALL (CE)"
        strategy = "Bullish Overnight Gap Carry"
        suggested_strike = f"₹{atm_strike} CE (At-The-Money Call)"
        alt_strike = f"₹{atm_strike + strike_round} CE (Slightly OTM)"
        entry_window = "3:08 PM - 3:20 PM IST Today"
        exit_window = "9:15 AM - 9:25 AM IST Tomorrow (First 5-10 min)"
        risk_guideline = "Exit immediately if market fails to gap up by 9:20 AM. Do not carry into intraday chop."
        bias_color = "#00E676"
    elif prob_up >= 55.0:
        action = "🟢 LEAN CALL (CE)"
        strategy = "Moderate Bullish Gap Tilt"
        suggested_strike = f"₹{atm_strike} CE (ATM Call)"
        alt_strike = "Small Quantity / Half Size"
        entry_window = "3:10 PM - 3:25 PM IST Today"
        exit_window = "9:15 AM - 9:20 AM IST Tomorrow"
        risk_guideline = "Keep risk controlled with strict overnight stop if gap does not materialize."
        bias_color = "#4ade80"
    elif prob_up <= 35.0:
        action = "🔴 BUY PUT (PE)"
        strategy = "Bearish Overnight Gap Down Carry"
        suggested_strike = f"₹{atm_strike} PE (At-The-Money Put)"
        alt_strike = f"₹{atm_strike - strike_round} PE (Slightly OTM)"
        entry_window = "3:08 PM - 3:20 PM IST Today"
        exit_window = "9:15 AM - 9:25 AM IST Tomorrow (First 5-10 min)"
        risk_guideline = "Lock profits on opening drop. Exit if stock opens flat or green."
        bias_color = "#FF5252"
    elif prob_up <= 45.0:
        action = "🔴 LEAN PUT (PE)"
        strategy = "Moderate Bearish Gap Tilt"
        suggested_strike = f"₹{atm_strike} PE (ATM Put)"
        alt_strike = "Small Quantity / Half Size"
        entry_window = "3:10 PM - 3:25 PM IST Today"
        exit_window = "9:15 AM - 9:20 AM IST Tomorrow"
        risk_guideline = "Cautious overnight short. Book quickly on 9:15 AM dip."
        bias_color = "#f87171"
    else:
        action = "⚪ NEUTRAL / NO TRADE"
        strategy = "High Overnight Uncertainty"
        suggested_strike = "None (Avoid Options Overnight)"
        alt_strike = "Wait for 9:15 AM Cash Open"
        entry_window = "No Entry Recommended"
        exit_window = "N/A"
        risk_guideline = "Gap odds are ~50-50. Premium decay (theta) will hurt both CE and PE buyers."
        bias_color = "#FFB300"

    return {
        "action": action,
        "strategy": strategy,
        "suggested_strike": suggested_strike,
        "alt_strike": alt_strike,
        "entry_window": entry_window,
        "exit_window": exit_window,
        "risk_guideline": risk_guideline,
        "bias_color": bias_color,
        "atm_strike": atm_strike
    }

def predict_opening_gap(symbol: str, period: str = "2y") -> dict:
    """
    Train and predict tomorrow's 9:15-9:20 AM Opening Gap direction (GAP UP vs GAP DOWN).
    Designed specifically for 3:05-3:10 PM options trading.
    """
    try:
        data, latest_row, feature_cols, err = prepare_opening_gap_dataset(symbol, period=period)
        if err or data is None or len(data) < 25:
            return {
                "status": "error",
                "message": err or "Not enough clean rows after indicator processing."
            }

        # Fetch real-time news sentiment
        clean_stock_name = STOCK_NAME_MAP.get(symbol, symbol.replace(".NS", "").replace(".BO", ""))
        news_info = get_stock_news_sentiment_score(clean_stock_name)

        # Inject news score into latest feature row
        latest_row_dict = latest_row[feature_cols].to_dict()
        latest_row_dict['News_Sentiment'] = news_info.get("score", 0.0)
        latest_features = pd.DataFrame([latest_row_dict])[feature_cols]

        X = data[feature_cols]
        y = data['Target_Gap']

        # Chronological train/test split
        split_idx = int(len(X) * (1 - ML_TEST_SIZE))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        rf_model = RandomForestClassifier(
            n_estimators=ML_N_ESTIMATORS,
            max_depth=ML_MAX_DEPTH,
            min_samples_leaf=ML_MIN_SAMPLES_LEAF,
            random_state=42,
            n_jobs=-1
        )

        hgb_model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=3,
            min_samples_leaf=15,
            random_state=42
        )

        ensemble = VotingClassifier(
            estimators=[('rf', rf_model), ('hgb', hgb_model)],
            voting='soft'
        )

        ensemble.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)

        # Out-of-sample evaluation on gap prediction
        y_preds = ensemble.predict(X_test)
        test_acc = accuracy_score(y_test, y_preds)
        test_prec = precision_score(y_test, y_preds, zero_division=0)
        test_rec = recall_score(y_test, y_preds, zero_division=0)

        # Predict probability incorporating live news bias
        raw_prob_up = ensemble.predict_proba(latest_features)[0][1]
        news_bias = news_info.get("score", 0.0) * 0.05
        prob_up_raw = float(np.clip(raw_prob_up + news_bias, 0.05, 0.95))

        # Recalibration from past opening gap memory if available
        from utils.opening_audit import calculate_gap_recalibration_offset
        gap_offset, gap_reason = calculate_gap_recalibration_offset(symbol)
        prob_up = float(np.clip(prob_up_raw + (gap_offset / 100.0), 0.05, 0.95))

        direction = "GAP UP 📈" if prob_up >= 0.50 else "GAP DOWN 📉"

        if prob_up >= 0.65 or prob_up <= 0.35:
            confidence = "High Confidence"
        elif prob_up >= 0.56 or prob_up <= 0.44:
            confidence = "Moderate Confidence"
        else:
            confidence = "Low / Neutral Confidence"

        feature_importances = dict(zip(feature_cols, rf_model.feature_importances_))
        sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

        current_price = round(latest_row['Close'], 2)
        options_call = generate_options_trading_call(symbol, current_price, prob_up * 100.0, confidence)

        return {
            "status": "success",
            "symbol": symbol,
            "direction": direction,
            "probability_up_pct": round(prob_up * 100, 1),
            "probability_down_pct": round((1 - prob_up) * 100, 1),
            "confidence": confidence,
            "test_accuracy_pct": round(test_acc * 100, 1),
            "precision_pct": round(test_prec * 100, 1),
            "recall_pct": round(test_rec * 100, 1),
            "feature_importances": sorted_importances,
            "sample_count": len(data),
            "test_sample_count": len(X_test),
            "current_price": current_price,
            "news_info": news_info,
            "gap_offset_pct": round(gap_offset, 1),
            "gap_reason": gap_reason,
            "options_call": options_call
        }
    except Exception as e:
        logging.error(f"Error predicting opening gap for {symbol}: {e}")
        return {"status": "error", "message": str(e)}
