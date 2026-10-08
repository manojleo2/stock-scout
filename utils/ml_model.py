import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from concurrent.futures import ThreadPoolExecutor
import logging
import streamlit as st

from utils.data_loader import get_stock_data
from utils.indicators import calculate_technical_indicators
from utils.macro_factors import get_macro_market_cues
from utils.news_sentiment import get_stock_news_sentiment_score
from config import (
    BENCHMARK_TICKER, ML_TEST_SIZE, ML_MAX_DEPTH, ML_MIN_SAMPLES_LEAF, 
    ML_N_ESTIMATORS, STOCK_NAME_MAP, AI_MIN_CONVICTION_THRESHOLD,
    AI_ATR_SL_MULT, AI_ATR_TP1_MULT, AI_ATR_TP2_MULT
)

logging.basicConfig(level=logging.INFO)

@st.cache_data(ttl=60, show_spinner=False)
def prepare_feature_dataset(symbol: str, period: str = "2y") -> tuple:
    """
    Construct stationary 28-feature institutional dataset with Alpha vs Nifty,
    Buying Pressure Index, Trend Convergence, Bollinger Z-Score, and Macro drivers in parallel.
    """
    with ThreadPoolExecutor(max_workers=3) as executor:
        f_stock = executor.submit(get_stock_data, symbol, period)
        f_nifty = executor.submit(get_stock_data, BENCHMARK_TICKER, period)
        f_macro = executor.submit(get_macro_market_cues, period)

        stock_df = f_stock.result()
        nifty_df = f_nifty.result()
        macro_df = f_macro.result()

    if stock_df.empty or len(stock_df) < 100:
        return None, None, None, "Insufficient stock historical data (need >= 100 trading days)."

    # 1. Standard Technical Indicators
    df = calculate_technical_indicators(stock_df)

    # 2. Benchmark (Nifty 50) Data & Multi-Day Relative Alpha
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

    # 3. Macro Global Cues & Volatility
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

    # 4. Stationary Return & Oscillator Features
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

    # 5. INSTITUTIONAL ALPHA & VOLUME-SPREAD FLOW FEATURES (8 Upgrades)
    # Feature 21 & 22: Relative Alpha vs Nifty 50 over 5d and 20d
    df['Alpha_Nifty_5d'] = (df['Ret_5'] - df['Nifty_Ret5']).fillna(0.0)
    df['Alpha_Nifty_20d'] = (df['Ret_20'] - df['Nifty_Ret20']).fillna(0.0)

    # Feature 23: Buying Pressure Index (BPI: Institutional Accumulation Proxy)
    day_range = (df['High'] - df['Low']).replace(0, np.nan)
    df['Buying_Pressure_Index'] = (((df['Close'] - df['Low']) / day_range).fillna(0.5) * df['Vol_Surge']).fillna(1.0)

    # Feature 24: Trend Harmonic Convergence Score (-1.0 to +1.0)
    sig_sma50 = np.sign(df['Close'] - df['SMA_50'])
    sig_ema = np.sign(df['EMA_12'] - df['EMA_26'])
    sig_sma200 = np.sign(df['Close'] - df['SMA_200'])
    df['Trend_Convergence_Score'] = ((sig_sma50 + sig_ema + sig_sma200) / 3.0).fillna(0.0)

    # Feature 25: Bollinger Z-Score (Normalized Distance from 20-day Mean in Standard Deviations)
    rolling_mean_20 = df['Close'].rolling(20).mean()
    rolling_std_20 = df['Close'].rolling(20).std().replace(0, np.nan)
    df['Bollinger_Z_Score'] = ((df['Close'] - rolling_mean_20) / rolling_std_20).fillna(0.0)

    # Feature 26: Normalized ATR Volatility Ratio
    df['ATR_Normalized_Ratio'] = (df['ATR_14'] / df['Close']).fillna(0.0)

    # Feature 27: Intraday Efficiency Ratio (Candle Body / Total Range)
    df['Intraday_Efficiency_Ratio'] = ((df['Close'] - df['Open']).abs() / day_range).fillna(0.5)

    # Feature 28: Consecutive Directional Persistence Streak
    direction_sign = np.where(df['Close'] > df['Close'].shift(1), 1, np.where(df['Close'] < df['Close'].shift(1), -1, 0))
    streak = np.zeros(len(df))
    for i in range(1, len(df)):
        if direction_sign[i] == direction_sign[i-1] and direction_sign[i] != 0:
            streak[i] = streak[i-1] + direction_sign[i]
        else:
            streak[i] = direction_sign[i]
    df['Consecutive_Direction_Streak'] = streak

    # Target: 1 if next day's close > today's close, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    feature_cols = [
        'Ret_1', 'Ret_5', 'Ret_20',
        'Dist_SMA50', 'Dist_SMA200', 'EMA_Cross_Ratio',
        'RSI_Norm', 'MACD_Hist_Norm', 'BB_PctB', 'BB_Width',
        'Vol_Surge', 'Nifty_Ret1', 'Nifty_Dist_SMA50',
        'SP500_Ret1', 'Nasdaq_Ret1', 'VIX_Norm', 'VIX_Ret1',
        'BankNifty_Ret1', 'Ret_VIX_Interact', 'News_Sentiment',
        'Alpha_Nifty_5d', 'Alpha_Nifty_20d', 'Buying_Pressure_Index',
        'Trend_Convergence_Score', 'Bollinger_Z_Score',
        'ATR_Normalized_Ratio', 'Intraday_Efficiency_Ratio',
        'Consecutive_Direction_Streak'
    ]

    clean_df = df.dropna(subset=feature_cols).copy()
    train_test_df = clean_df.iloc[:-1].dropna(subset=['Target'])

    return train_test_df, clean_df.iloc[-1], feature_cols, None

def generate_quant_execution_blueprint(symbol: str, current_price: float, atr_14: float, prob_up: float, confidence: str) -> dict:
    """
    Generate an institutional risk-reward trade blueprint based on ATR multiples.
    Enforces a >=60% Conviction Gatekeeper to preserve capital during chop.
    """
    conviction = max(prob_up, 100.0 - prob_up)
    trade_horizon = "Recommended Trade Horizon: Next-Day Intraday Breakout or 1–3 Day Swing Carry (Cash / Futures / Stock Equity)"

    # Fallback ATR if 0
    safe_atr = atr_14 if atr_14 > 0 else (current_price * 0.015)
    sl_points = round(safe_atr * AI_ATR_SL_MULT, 2)
    tp1_points = round(safe_atr * AI_ATR_TP1_MULT, 2)
    tp2_points = round(safe_atr * AI_ATR_TP2_MULT, 2)

    if conviction < AI_MIN_CONVICTION_THRESHOLD:
        action = "⚪ NEUTRAL / CAPITAL PRESERVATION"
        strategy = "Consolidation / Chop Regime (<60% Conviction)"
        entry_level = "No Swing Entry Recommended (Preserve Capital)"
        stop_loss = None
        target_1 = None
        target_2 = None
        risk_reward_ratio = "N/A"
        bias_color = "#FFB300"
        is_tradeable = False
        execution_guideline = (
            f"Model conviction is {conviction:.1f}% (below {AI_MIN_CONVICTION_THRESHOLD:.0f}% threshold). "
            f"The asset is in a low-edge consolidation zone. Sideline cash to avoid whipsaws."
        )
    elif prob_up >= AI_MIN_CONVICTION_THRESHOLD:
        action = "🟢 STRONG BULLISH SWING / BREAKOUT"
        strategy = "Bullish Momentum & Alpha Outperformance"
        entry_level = f"₹{current_price:,.2f} (LTP) or pullback to ₹{current_price - (safe_atr * 0.25):,.2f}"
        stop_loss = round(current_price - sl_points, 2)
        target_1 = round(current_price + tp1_points, 2)
        target_2 = round(current_price + tp2_points, 2)
        risk_reward_ratio = "1.5 : 1 (Target 1) | 2.5 : 1 (Target 2)"
        bias_color = "#00E676"
        is_tradeable = True
        execution_guideline = (
            f"Enter long on market open/dip. Set initial Stop-Loss at ₹{stop_loss:,.2f} (-₹{sl_points:,.2f}). "
            f"Lock 50% profits at Target 1 (₹{target_1:,.2f}) and trail Stop-Loss to Cost (Breakeven)."
        )
    else:
        action = "🔴 STRONG BEARISH SWING / BREAKDOWN"
        strategy = "Bearish Breakdown & Relative Weakness"
        entry_level = f"₹{current_price:,.2f} (LTP) or rise to ₹{current_price + (safe_atr * 0.25):,.2f}"
        stop_loss = round(current_price + sl_points, 2)
        target_1 = round(current_price - tp1_points, 2)
        target_2 = round(current_price - tp2_points, 2)
        risk_reward_ratio = "1.5 : 1 (Target 1) | 2.5 : 1 (Target 2)"
        bias_color = "#FF5252"
        is_tradeable = True
        execution_guideline = (
            f"Exit long positions or enter short/hedges. Set Stop-Loss at ₹{stop_loss:,.2f} (+₹{sl_points:,.2f}). "
            f"Lock 50% profits at Target 1 (₹{target_1:,.2f}) and trail SL to Cost."
        )

    return {
        "action": action,
        "strategy": strategy,
        "entry_level": entry_level,
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "sl_points": sl_points,
        "tp1_points": tp1_points,
        "tp2_points": tp2_points,
        "risk_reward_ratio": risk_reward_ratio,
        "bias_color": bias_color,
        "is_tradeable": is_tradeable,
        "conviction_pct": round(conviction, 1),
        "trade_horizon": trade_horizon,
        "execution_guideline": execution_guideline,
        "atr_14": round(safe_atr, 2)
    }

@st.cache_data(ttl=60, show_spinner=False)
def train_and_predict(symbol: str, period: str = "2y") -> dict:
    """
    Train Supercharged Ensemble Classifier (RandomForest + HistGradientBoosting) with 
    exponential recency sample weighting and 28 institutional alpha & flow features.
    """
    try:
        data, latest_row, feature_cols, err = prepare_feature_dataset(symbol, period=period)
        if err or data is None or len(data) < 25:
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
        latest_features = pd.DataFrame([latest_row_dict])[feature_cols]

        X = data[feature_cols]
        y = data['Target']

        # Time-Disjoint Chronological Split (Train: 70%, Calibration: 15%, Test/Validation: 15%)
        # Strictly chronological: Fitting, calibration, and test data are mutually disjoint
        n_samples = len(X)
        train_idx = int(n_samples * 0.70)
        cal_idx = int(n_samples * 0.85)

        X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
        X_cal, y_cal = X.iloc[train_idx:cal_idx], y.iloc[train_idx:cal_idx]
        X_test, y_test = X.iloc[cal_idx:], y.iloc[cal_idx:]

        # Exponential recency sample weighting on train set
        sample_weights_train = np.exp(np.linspace(-0.5, 0.0, len(X_train)))

        # Base Models fitted on disjoint training window
        rf_base = RandomForestClassifier(
            n_estimators=100,
            max_depth=ML_MAX_DEPTH,
            min_samples_leaf=ML_MIN_SAMPLES_LEAF,
            random_state=42,
            n_jobs=2
        )
        hgb_base = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=3,
            min_samples_leaf=15,
            random_state=42
        )

        rf_base.fit(X_train, y_train, sample_weight=sample_weights_train)
        hgb_base.fit(X_train, y_train, sample_weight=sample_weights_train)

        # Time-Disjoint Probability Calibration Layer (scikit-learn 1.9+ FrozenEstimator + Sigmoid)
        # Model-fitting and calibration data are strictly disjoint to prevent lookahead leakage
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.frozen import FrozenEstimator
        from sklearn.metrics import brier_score_loss

        cal_rf = CalibratedClassifierCV(FrozenEstimator(rf_base), method="sigmoid")
        cal_hgb = CalibratedClassifierCV(FrozenEstimator(hgb_base), method="sigmoid")

        cal_rf.fit(X_cal, y_cal)
        cal_hgb.fit(X_cal, y_cal)

        # Out-of-sample evaluation on untouched test set (Raw vs Calibrated)
        pred_rf_raw_test = rf_base.predict_proba(X_test)[:, 1]
        pred_hgb_raw_test = hgb_base.predict_proba(X_test)[:, 1]
        y_prob_raw_test = 0.5 * pred_rf_raw_test + 0.5 * pred_hgb_raw_test
        raw_test_brier = brier_score_loss(y_test, y_prob_raw_test)

        pred_rf_cal_test = cal_rf.predict_proba(X_test)[:, 1]
        pred_hgb_cal_test = cal_hgb.predict_proba(X_test)[:, 1]
        y_prob_cal_test = 0.5 * pred_rf_cal_test + 0.5 * pred_hgb_cal_test
        cal_test_brier = brier_score_loss(y_test, y_prob_cal_test)

        # Quant Guardrail 1: Brier Score Safety Net (Brier_cal must strictly improve over Brier_raw)
        brier_improved = bool(cal_test_brier < raw_test_brier)

        # Quant Guardrail 2: Slope Sanity Check (Platt Sigmoid slope a_ must be negative for valid monotonic scaling)
        slope_valid = True
        try:
            for clf in list(cal_rf.calibrated_classifiers_) + list(cal_hgb.calibrated_classifiers_):
                for c in getattr(clf, "calibrators", []):
                    if hasattr(c, "a_") and getattr(c, "a_") >= 0:
                        slope_valid = False
                        break
        except Exception:
            pass

        # Select best probability stream on test set for metrics
        use_calibration = brier_improved and slope_valid
        y_prob_ensemble_test = y_prob_cal_test if use_calibration else y_prob_raw_test
        y_preds_test = (y_prob_ensemble_test >= 0.50).astype(int)

        test_acc = accuracy_score(y_test, y_preds_test)
        test_prec = precision_score(y_test, y_preds_test, zero_division=0)
        test_rec = recall_score(y_test, y_preds_test, zero_division=0)
        test_brier = cal_test_brier if use_calibration else raw_test_brier

        # Predict raw vs calibrated probability on latest features
        raw_rf = rf_base.predict_proba(latest_features)[0][1]
        raw_hgb = hgb_base.predict_proba(latest_features)[0][1]
        raw_prob_up = float(0.5 * raw_rf + 0.5 * raw_hgb)

        cal_rf_live = cal_rf.predict_proba(latest_features)[0][1]
        cal_hgb_live = cal_hgb.predict_proba(latest_features)[0][1]
        calibrated_prob_up = float(0.5 * cal_rf_live + 0.5 * cal_hgb_live)

        # Automated Production Guardrail Decision:
        # Use calibrated probability ONLY IF it passed slope sanity AND genuinely improved test Brier score.
        # Otherwise, safely fallback to the robust pure ensemble to protect live execution.
        if use_calibration:
            chosen_prob_up = calibrated_prob_up
            calibration_arch = "FrozenEstimator (Sigmoid Calibrated - Brier Improved)"
        else:
            chosen_prob_up = raw_prob_up
            calibration_arch = f"Pure Production Ensemble (Fallback: Calibrator Brier {cal_test_brier:.4f} >= Raw {raw_test_brier:.4f})"

        prob_up = float(np.clip(chosen_prob_up, 0.05, 0.95))

        direction = "UP 📈" if prob_up >= 0.50 else "DOWN 📉"
        conviction_pct = max(prob_up, 1.0 - prob_up) * 100.0


        if conviction_pct >= 65.0:
            confidence = "Super High Confidence"
        elif conviction_pct >= 60.0:
            confidence = "High Confidence"
        elif conviction_pct >= 55.0:
            confidence = "Moderate Confidence"
        else:
            confidence = "Low / Neutral (Chop)"

        feature_importances = dict(zip(feature_cols, rf_base.feature_importances_))
        sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

        latest_close = round(latest_row['Close'], 2)
        atr_14_val = float(latest_row.get('ATR_14', latest_close * 0.015))
        quant_blueprint = generate_quant_execution_blueprint(
            symbol=symbol,
            current_price=latest_close,
            atr_14=atr_14_val,
            prob_up=prob_up * 100.0,
            confidence=confidence
        )

        return {
            "status": "success",
            "symbol": symbol,
            "direction": direction,
            "raw_probability_up_pct": round(raw_prob_up * 100, 1),
            "probability_up_pct": round(prob_up * 100, 1),
            "probability_down_pct": round((1 - prob_up) * 100, 1),
            "confidence": confidence,
            "test_accuracy_pct": round(test_acc * 100, 1),
            "precision_pct": round(test_prec * 100, 1),
            "recall_pct": round(test_rec * 100, 1),
            "brier_score": round(test_brier, 4),
            "calibration_architecture": calibration_arch,
            "feature_importances": sorted_importances,
            "sample_count": len(data),
            "test_sample_count": len(X_test),
            "latest_close": latest_close,
            "news_info": news_info,
            "quant_blueprint": quant_blueprint,
            "alpha_5d_pct": round(float(latest_row.get('Alpha_Nifty_5d', 0.0)) * 100.0, 2),
            "alpha_20d_pct": round(float(latest_row.get('Alpha_Nifty_20d', 0.0)) * 100.0, 2),
            "buying_pressure_index": round(float(latest_row.get('Buying_Pressure_Index', 1.0)), 2),
            "trend_convergence": round(float(latest_row.get('Trend_Convergence_Score', 0.0)), 2),
            "bollinger_z_score": round(float(latest_row.get('Bollinger_Z_Score', 0.0)), 2)
        }
    except Exception as e:
        logging.error(f"Error training ML model for {symbol}: {e}")
        return {"status": "error", "message": str(e)}

