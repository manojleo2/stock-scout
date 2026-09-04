import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging

from utils.data_loader import get_stock_data
from utils.indicators import calculate_technical_indicators
from config import BENCHMARK_TICKER, ML_TEST_SIZE, ML_MAX_DEPTH, ML_MIN_SAMPLES_LEAF, ML_N_ESTIMATORS

def prepare_feature_dataset(symbol: str, period: str = "2y") -> tuple:
    """
    Construct stationary ML feature set and target variable without lookahead bias.
    """
    stock_df = get_stock_data(symbol, period=period)
    if stock_df.empty or len(stock_df) < 100:
        return None, None, None, "Insufficient stock historical data (need >= 100 trading days)."

    # Add technical indicators
    df = calculate_technical_indicators(stock_df)

    # Fetch Nifty data as external factor
    nifty_df = get_stock_data(BENCHMARK_TICKER, period=period)
    if not nifty_df.empty:
        nifty_ret = nifty_df['Close'].pct_change()
        df['Nifty_Ret1'] = nifty_ret
    else:
        df['Nifty_Ret1'] = 0.0

    # Feature Engineering (strictly stationary ratios & percentages)
    df['Ret_1'] = df['Close'].pct_change(1)
    df['Ret_5'] = df['Close'].pct_change(5)
    df['Ret_20'] = df['Close'].pct_change(20)

    df['Dist_SMA50'] = (df['Close'] / df['SMA_50']) - 1.0
    df['Dist_SMA200'] = (df['Close'] / df['SMA_200']) - 1.0
    df['EMA_Cross_Ratio'] = (df['EMA_12'] / df['EMA_26']) - 1.0

    df['RSI_Norm'] = df['RSI_14'] / 100.0
    df['MACD_Hist_Norm'] = df['MACD_Hist'] / df['Close']
    df['BB_PctB'] = df['BB_PctB']
    df['BB_Width'] = df['BB_Width']
    df['Vol_Surge'] = df['Vol_Surge']

    # Target: 1 if next day's close > today's close, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    feature_cols = [
        'Ret_1', 'Ret_5', 'Ret_20',
        'Dist_SMA50', 'Dist_SMA200', 'EMA_Cross_Ratio',
        'RSI_Norm', 'MACD_Hist_Norm', 'BB_PctB', 'BB_Width',
        'Vol_Surge', 'Nifty_Ret1'
    ]

    # Clean missing values
    clean_df = df.dropna(subset=feature_cols).copy()
    
    # Exclude the very last row for training/testing (since its target shift(-1) is unknown)
    train_test_df = clean_df.iloc[:-1].dropna(subset=['Target'])

    return train_test_df, clean_df.iloc[-1], feature_cols, None

def train_and_predict(symbol: str, period: str = "2y") -> dict:
    """
    Train RandomForestClassifier on historical data and predict tomorrow's direction.
    """
    try:
        data, latest_row, feature_cols, err = prepare_feature_dataset(symbol, period=period)
        if err or data is None or len(data) < 60:
            return {
                "status": "error",
                "message": err or "Not enough clean rows after indicator processing."
            }

        X = data[feature_cols]
        y = data['Target']

        # Chronological train/test split (80% past data train, 20% recent data out-of-sample test)
        split_idx = int(len(X) * (1 - ML_TEST_SIZE))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=ML_N_ESTIMATORS,
            max_depth=ML_MAX_DEPTH,
            min_samples_leaf=ML_MIN_SAMPLES_LEAF,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Out-of-sample evaluation
        y_preds = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_preds)

        # Predict tomorrow using latest available features (today's closing features)
        latest_features = latest_row[feature_cols].values.reshape(1, -1)
        prob_up = model.predict_proba(latest_features)[0][1]
        direction = "UP 📈" if prob_up >= 0.50 else "DOWN 📉"

        # Signal confidence label
        if prob_up >= 0.65 or prob_up <= 0.35:
            confidence = "High Confidence"
        elif prob_up >= 0.58 or prob_up <= 0.42:
            confidence = "Moderate Confidence"
        else:
            confidence = "Low / Neutral Confidence"

        # Feature Importance Breakdown
        feature_importances = dict(zip(feature_cols, model.feature_importances_))
        sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

        return {
            "status": "success",
            "symbol": symbol,
            "direction": direction,
            "probability_up_pct": round(prob_up * 100, 1),
            "probability_down_pct": round((1 - prob_up) * 100, 1),
            "confidence": confidence,
            "test_accuracy_pct": round(test_acc * 100, 1),
            "feature_importances": sorted_importances,
            "sample_count": len(data),
            "test_sample_count": len(X_test),
            "latest_close": round(latest_row['Close'], 2)
        }
    except Exception as e:
        logging.error(f"Error training ML model for {symbol}: {e}")
        return {"status": "error", "message": str(e)}
