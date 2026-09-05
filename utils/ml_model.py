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

def prepare_feature_dataset(symbol: str, period: str = "2y") -> tuple:
    """
    Construct stationary ML feature set with technical, Nifty, India VIX, global cues, and news sentiment.
    """
    stock_df = get_stock_data(symbol, period=period)
    if stock_df.empty or len(stock_df) < 100:
        return None, None, None, "Insufficient stock historical data (need >= 100 trading days)."

    # Add technical indicators
    df = calculate_technical_indicators(stock_df)

    # Add Nifty benchmark data
    nifty_df = get_stock_data(BENCHMARK_TICKER, period=period)
    if not nifty_df.empty:
        nifty_df.index = nifty_df.index.tz_localize(None) if nifty_df.index.tz is not None else nifty_df.index
        df['Nifty_Ret1'] = nifty_df['Close'].pct_change(1)
        df['Nifty_Dist_SMA50'] = (nifty_df['Close'] / nifty_df['Close'].rolling(50).mean()) - 1.0
    else:
        df['Nifty_Ret1'] = 0.0
        df['Nifty_Dist_SMA50'] = 0.0

    # Add Macro Global Cues & India VIX
    macro_df = get_macro_market_cues(period=period)
    if not macro_df.empty:
        df = df.join(macro_df, how='left')

    # Ensure missing macro columns are safely filled
    for col in ['SP500_Ret1', 'Nasdaq_Ret1', 'VIX_Norm', 'VIX_Ret1', 'BankNifty_Ret1']:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].ffill().fillna(0.0)

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

    # Interaction feature: Stock Return x VIX Change
    df['Ret_VIX_Interact'] = df['Ret_1'] * df['VIX_Ret1']

    # Real-time News Sentiment Feature Column
    df['News_Sentiment'] = 0.0

    # Target: 1 if next day's close > today's close, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    feature_cols = [
        'Ret_1', 'Ret_5', 'Ret_20',
        'Dist_SMA50', 'Dist_SMA200', 'EMA_Cross_Ratio',
        'RSI_Norm', 'MACD_Hist_Norm', 'BB_PctB', 'BB_Width',
        'Vol_Surge', 'Nifty_Ret1', 'Nifty_Dist_SMA50',
        'SP500_Ret1', 'Nasdaq_Ret1', 'VIX_Norm', 'VIX_Ret1',
        'BankNifty_Ret1', 'Ret_VIX_Interact', 'News_Sentiment'
    ]

    clean_df = df.dropna(subset=feature_cols).copy()
    train_test_df = clean_df.iloc[:-1].dropna(subset=['Target'])

    return train_test_df, clean_df.iloc[-1], feature_cols, None

def train_and_predict(symbol: str, period: str = "2y") -> dict:
    """
    Train Ensemble Classifier (RandomForest + HistGradientBoosting) with live news sentiment.
    """
    try:
        data, latest_row, feature_cols, err = prepare_feature_dataset(symbol, period=period)
        if err or data is None or len(data) < 60:
            return {
                "status": "error",
                "message": err or "Not enough clean rows after indicator processing."
            }

        # Fetch real-time news sentiment score for this hour
        clean_stock_name = STOCK_NAME_MAP.get(symbol, symbol.replace(".NS", "").replace(".BO", ""))
        news_info = get_stock_news_sentiment_score(clean_stock_name)
        
        # Inject live news sentiment score into latest_row for prediction
        latest_row_dict = latest_row[feature_cols].to_dict()
        latest_row_dict['News_Sentiment'] = news_info.get("score", 0.0)
        
        latest_features = np.array([latest_row_dict[col] for col in feature_cols]).reshape(1, -1)

        X = data[feature_cols]
        y = data['Target']

        # Chronological train/test split
        split_idx = int(len(X) * (1 - ML_TEST_SIZE))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Base Models
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

        # Out-of-sample evaluation
        y_preds = ensemble.predict(X_test)
        test_acc = accuracy_score(y_test, y_preds)
        test_prec = precision_score(y_test, y_preds, zero_division=0)
        test_rec = recall_score(y_test, y_preds, zero_division=0)

        # Predict probability incorporating live news score
        raw_prob_up = ensemble.predict_proba(latest_features)[0][1]
        
        # Apply news sentiment bias adjustment (+/- 5% max adjustment based on real-time news in this hour)
        news_bias = news_info.get("score", 0.0) * 0.05
        prob_up = float(np.clip(raw_prob_up + news_bias, 0.05, 0.95))
        
        direction = "UP 📈" if prob_up >= 0.50 else "DOWN 📉"

        if prob_up >= 0.65 or prob_up <= 0.35:
            confidence = "High Confidence"
        elif prob_up >= 0.56 or prob_up <= 0.44:
            confidence = "Moderate Confidence"
        else:
            confidence = "Low / Neutral Confidence"

        feature_importances = dict(zip(feature_cols, rf_model.feature_importances_))
        sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

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
            "latest_close": round(latest_row['Close'], 2),
            "news_info": news_info
        }
    except Exception as e:
        logging.error(f"Error training ML model for {symbol}: {e}")
        return {"status": "error", "message": str(e)}
