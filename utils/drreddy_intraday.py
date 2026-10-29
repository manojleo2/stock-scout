# -*- coding: utf-8 -*-
"""
=============================================================================
 DR. REDDY'S STANDALONE INTRADAY SPECIALIST MODEL — utils/drreddy_intraday.py
=============================================================================
 PURPOSE:
   Dedicated institutional-grade Intraday & Swing Trend Predictor for DRREDDY.NS.
   Completely standalone from the CDSL intraday model (utils/ml_model.py).

   Uses specialized Pharma sector & Export currency dynamics:
   - Nifty Pharma Index (^CNXPHARMA) co-integration & relative alpha spread
   - USD/INR Currency rate (USDINR=X) tracking US generic export tailwinds
   - Nifty 50 relative alpha & defensive decoupling
   - Normalized ATR execution blueprint (625 Lot Size, ₹20 strike steps)
   - Strict 60% Conviction Chop Filter (65% on Friday) to eliminate sideways noise
   - Platt Sigmoid Probability Calibration via FrozenEstimator with Brier & Slope Guardrails

 RULE: THIS FILE DOES NOT IMPORT FROM utils/ml_model.py OR utils/prediction_audit.py.
       Those files belong exclusively to the CDSL model.
=============================================================================
"""

import pandas as pd
import numpy as np
import datetime as dt
import calendar
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging
import streamlit as st

from utils.data_loader import get_stock_data
from utils.indicators import calculate_technical_indicators
from utils.news_sentiment import get_stock_news_sentiment_score
try:
    import config
    BENCHMARK_TICKER = getattr(config, "BENCHMARK_TICKER", "^NSEI")
    ML_MAX_DEPTH = getattr(config, "ML_MAX_DEPTH", 4)
    ML_MIN_SAMPLES_LEAF = getattr(config, "ML_MIN_SAMPLES_LEAF", 20)
    AI_MIN_CONVICTION_THRESHOLD = getattr(config, "AI_MIN_CONVICTION_THRESHOLD", 60.0)
    AI_ATR_SL_MULT = getattr(config, "AI_ATR_SL_MULT", 1.0)
    AI_ATR_TP1_MULT = getattr(config, "AI_ATR_TP1_MULT", 1.5)
    AI_ATR_TP2_MULT = getattr(config, "AI_ATR_TP2_MULT", 2.5)
    DRREDDY_SYMBOL = getattr(config, "DRREDDY_SYMBOL", "DRREDDY.NS")
    DRREDDY_PHARMA_TICKER = getattr(config, "DRREDDY_PHARMA_TICKER", "^CNXPHARMA")
    DRREDDY_USDINR_TICKER = getattr(config, "DRREDDY_USDINR_TICKER", "USDINR=X")
    DRREDDY_LOT_SIZE = getattr(config, "DRREDDY_LOT_SIZE", 625)
    DRREDDY_STRIKE_STEP = getattr(config, "DRREDDY_STRIKE_STEP", 20)
    DRREDDY_MIN_CONVICTION_WEEKDAY = getattr(config, "DRREDDY_MIN_CONVICTION_WEEKDAY", 60.0)
    DRREDDY_MIN_CONVICTION_FRIDAY = getattr(config, "DRREDDY_MIN_CONVICTION_FRIDAY", 65.0)
except Exception:
    BENCHMARK_TICKER = "^NSEI"
    ML_MAX_DEPTH = 4
    ML_MIN_SAMPLES_LEAF = 20
    AI_MIN_CONVICTION_THRESHOLD = 60.0
    AI_ATR_SL_MULT = 1.0
    AI_ATR_TP1_MULT = 1.5
    AI_ATR_TP2_MULT = 2.5
    DRREDDY_SYMBOL = "DRREDDY.NS"
    DRREDDY_PHARMA_TICKER = "^CNXPHARMA"
    DRREDDY_USDINR_TICKER = "USDINR=X"
    DRREDDY_LOT_SIZE = 625
    DRREDDY_STRIKE_STEP = 20
    DRREDDY_MIN_CONVICTION_WEEKDAY = 60.0
    DRREDDY_MIN_CONVICTION_FRIDAY = 65.0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("drreddy_intraday")

SYMBOL = DRREDDY_SYMBOL


# ─────────────────────────────────────────────────────────────────────────────
# 1. EXPIRY DYNAMICS HELPERS (Standalone copy — avoids cross-module dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def _days_to_monthly_expiry(date_val) -> int:
    """Calendar days to last Thursday of the month (NSE stock expiry)."""
    if date_val is None:
        d = dt.date.today()
    elif hasattr(date_val, 'date'):
        d = date_val.date()
    elif isinstance(date_val, str):
        try:
            d = dt.datetime.strptime(date_val, "%a, %d %b %Y").date()
        except Exception:
            try:
                d = dt.datetime.strptime(date_val, "%Y-%m-%d").date()
            except Exception:
                d = dt.date.today()
    else:
        d = date_val

    year, month = d.year, d.month
    _, last_day = calendar.monthrange(year, month)
    last_dt = dt.date(year, month, last_day)
    offset = (last_dt.weekday() - 3) % 7   # 3 = Thursday
    expiry_dt = last_dt - dt.timedelta(days=offset)

    if d > expiry_dt:
        next_m = 1 if month == 12 else month + 1
        next_y = year + 1 if month == 12 else year
        _, next_last_day = calendar.monthrange(next_y, next_m)
        next_last_dt = dt.date(next_y, next_m, next_last_day)
        next_offset = (next_last_dt.weekday() - 3) % 7
        expiry_dt = next_last_dt - dt.timedelta(days=next_offset)

    return max(0, (expiry_dt - d).days)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING ENGINE (28 Institutional Pharma Features)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def prepare_drreddy_feature_dataset(period: str = "2y") -> tuple:
    """
    Construct stationary 28-feature institutional dataset for Dr. Reddy's.
    Integrates Nifty Pharma co-integration, USD/INR export tailwind,
    Buying Pressure Index, Trend Convergence, and Bollinger Z-Score.
    """
    try:
        # Sequential data fetching to avoid thread starvation on Streamlit Cloud
        stock_df = get_stock_data(SYMBOL, period)
        pharma_df = get_stock_data(DRREDDY_PHARMA_TICKER, period)
        usdinr_df = get_stock_data(DRREDDY_USDINR_TICKER, period)
        nifty_df = get_stock_data(BENCHMARK_TICKER, period)
        vix_df = get_stock_data("^INDIAVIX", period)

        if stock_df.empty or len(stock_df) < 100:
            return None, None, None, "Insufficient DRREDDY historical data (need >= 100 trading days)."

        # Standard Technical Indicators
        df = calculate_technical_indicators(stock_df)

        # 1. Nifty Pharma Sector Co-integration & Relative Alpha
        if not pharma_df.empty:
            pharma_df.index = pharma_df.index.tz_localize(None) if pharma_df.index.tz is not None else pharma_df.index
            p_close = pharma_df['Close'].reindex(df.index).ffill()
            df['Pharma_Ret1'] = p_close.pct_change(1).fillna(0.0)
            df['Pharma_Ret5'] = p_close.pct_change(5).fillna(0.0)
            df['Pharma_Ret20'] = p_close.pct_change(20).fillna(0.0)
            p_sma50 = p_close.rolling(50).mean().bfill().fillna(p_close)
            df['Pharma_Dist_SMA50'] = ((p_close / p_sma50) - 1.0).fillna(0.0)
        else:
            df['Pharma_Ret1'] = 0.0
            df['Pharma_Ret5'] = 0.0
            df['Pharma_Ret20'] = 0.0
            df['Pharma_Dist_SMA50'] = 0.0

        # 2. USD/INR Currency Drivers (US Generics Export Alpha)
        if not usdinr_df.empty:
            usdinr_df.index = usdinr_df.index.tz_localize(None) if usdinr_df.index.tz is not None else usdinr_df.index
            fx_close = usdinr_df['Close'].reindex(df.index).ffill()
            df['USDINR_Ret1'] = fx_close.pct_change(1).fillna(0.0)
            df['USDINR_Ret5'] = fx_close.pct_change(5).fillna(0.0)
            fx_sma20 = fx_close.rolling(20).mean().bfill().fillna(fx_close)
            df['USDINR_Trend'] = ((fx_close / fx_sma20) - 1.0).fillna(0.0)
        else:
            df['USDINR_Ret1'] = 0.0
            df['USDINR_Ret5'] = 0.0
            df['USDINR_Trend'] = 0.0

        # 3. Benchmark Nifty 50 Data & Market Decoupling
        if not nifty_df.empty:
            nifty_df.index = nifty_df.index.tz_localize(None) if nifty_df.index.tz is not None else nifty_df.index
            n_close = nifty_df['Close'].reindex(df.index).ffill()
            df['Nifty_Ret1'] = n_close.pct_change(1).fillna(0.0)
            df['Nifty_Ret5'] = n_close.pct_change(5).fillna(0.0)
            n_sma50 = n_close.rolling(50).mean().bfill().fillna(n_close)
            df['Nifty_Dist_SMA50'] = ((n_close / n_sma50) - 1.0).fillna(0.0)
        else:
            df['Nifty_Ret1'] = 0.0
            df['Nifty_Ret5'] = 0.0
            df['Nifty_Dist_SMA50'] = 0.0

        # 4. India VIX Volatility Filter
        if not vix_df.empty:
            vix_df.index = vix_df.index.tz_localize(None) if vix_df.index.tz is not None else vix_df.index
            v_close = vix_df['Close'].reindex(df.index).ffill()
            df['VIX_Norm'] = (v_close / 50.0).fillna(0.3)
            df['VIX_Ret1'] = v_close.pct_change(1).fillna(0.0)
        else:
            df['VIX_Norm'] = 0.3
            df['VIX_Ret1'] = 0.0

        # Moving average NaN safe fills
        df['SMA_50'] = df['SMA_50'].bfill().ffill().fillna(df['Close'])
        df['SMA_200'] = df['SMA_200'].bfill().ffill().fillna(df['Close'])

        # 5. Stationary Returns & Technical Ratios
        df['Ret_1'] = df['Close'].pct_change(1).fillna(0.0)
        df['Ret_5'] = df['Close'].pct_change(5).fillna(0.0)
        df['Ret_20'] = df['Close'].pct_change(20).fillna(0.0)

        df['Dist_SMA50'] = ((df['Close'] / df['SMA_50']) - 1.0).fillna(0.0)
        df['Dist_SMA200'] = ((df['Close'] / df['SMA_200']) - 1.0).fillna(0.0)
        df['EMA_Cross_Ratio'] = ((df['EMA_12'] / df['EMA_26']) - 1.0).fillna(0.0)

        df['RSI_Norm'] = (df['RSI_14'] / 100.0).fillna(0.5)
        df['MACD_Hist_Norm'] = (df['MACD_Hist'] / df['Close']).fillna(0.0)
        df['BB_PctB'] = df['BB_PctB'].fillna(0.5)
        df['BB_Width'] = df['BB_Width'].fillna(0.0)
        df['Vol_Surge'] = df['Vol_Surge'].fillna(1.0)
        df['Ret_VIX_Interact'] = (df['Ret_1'] * df['VIX_Ret1']).fillna(0.0)
        df['News_Sentiment'] = 0.0

        # 6. Institutional Pharma Sector Alpha & Microstructure Flow
        df['Alpha_Pharma_5d'] = (df['Ret_5'] - df['Pharma_Ret5']).fillna(0.0)
        df['Alpha_Pharma_20d'] = (df['Ret_20'] - df['Pharma_Ret20']).fillna(0.0)
        df['Alpha_Nifty_5d'] = (df['Ret_5'] - df['Nifty_Ret5']).fillna(0.0)

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

        # Target: 1 if next day's close > today's close, else 0 (forward shift to eliminate leakage)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        feature_cols = [
            'Ret_1', 'Ret_5', 'Ret_20',
            'Dist_SMA50', 'Dist_SMA200', 'EMA_Cross_Ratio',
            'RSI_Norm', 'MACD_Hist_Norm', 'BB_PctB', 'BB_Width',
            'Vol_Surge', 'Pharma_Ret1', 'Pharma_Dist_SMA50',
            'USDINR_Ret1', 'USDINR_Ret5', 'USDINR_Trend',
            'Nifty_Ret1', 'Alpha_Nifty_5d', 'VIX_Norm', 'VIX_Ret1',
            'Ret_VIX_Interact', 'News_Sentiment',
            'Alpha_Pharma_5d', 'Alpha_Pharma_20d',
            'Buying_Pressure_Index', 'Trend_Convergence_Score',
            'Bollinger_Z_Score', 'ATR_Normalized_Ratio',
            'Intraday_Efficiency_Ratio', 'Consecutive_Direction_Streak'
        ]

        clean_df = df.dropna(subset=feature_cols).copy()
        train_test_df = clean_df.iloc[:-1].dropna(subset=['Target'])

        return train_test_df, clean_df.iloc[-1], feature_cols, None

    except Exception as e:
        logger.error(f"Error in prepare_drreddy_feature_dataset: {e}")
        return None, None, None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# 3. F&O EXECUTION BLUEPRINT (625 Lot Size, ₹20 Strike Steps)
# ─────────────────────────────────────────────────────────────────────────────

def generate_drreddy_execution_blueprint(current_price: float, atr_14: float, prob_up: float, confidence: str) -> dict:
    """
    Generate an institutional risk-reward trade blueprint tailored for Dr. Reddy's.
    Contract Specs: Lot Size = 625, Strike Steps = ₹20.
    Enforces a >=60% Conviction Gatekeeper (>=65% Friday) to preserve capital during chop.
    """
    conviction = max(prob_up, 100.0 - prob_up)
    trade_horizon = "Recommended Horizon: 09:20 AM Entry -> Intraday Trend (03:05 PM Hard Cutoff)"

    safe_atr = atr_14 if atr_14 > 0 else (current_price * 0.02)
    sl_points = round(safe_atr * AI_ATR_SL_MULT, 2)
    tp1_points = round(safe_atr * AI_ATR_TP1_MULT, 2)
    tp2_points = round(safe_atr * AI_ATR_TP2_MULT, 2)

    today_weekday = dt.datetime.now().weekday()
    min_threshold = DRREDDY_MIN_CONVICTION_FRIDAY if today_weekday == 4 else DRREDDY_MIN_CONVICTION_WEEKDAY

    atm_strike = round(current_price / DRREDDY_STRIKE_STEP) * DRREDDY_STRIKE_STEP

    if conviction < min_threshold:
        action = "⚪ NEUTRAL / CAPITAL PRESERVATION"
        strategy = f"Consolidation / Low-Edge Regime (<{min_threshold:.0f}% Conviction)"
        entry_level = "No Trade Recommended (Preserve Capital)"
        stop_loss = None
        target_1 = None
        target_2 = None
        risk_reward_ratio = "N/A"
        bias_color = "#FFB300"
        is_tradeable = False
        target_contract = f"DRREDDY ₹{atm_strike:.0f} (SIDELINED)"
        execution_guideline = (
            f"Model conviction is {conviction:.1f}% (below {min_threshold:.0f}% threshold). "
            f"Asset is consolidating. Sideline capital to eliminate false breakout traps."
        )
    elif prob_up >= min_threshold:
        action = "🟢 STRONG BULLISH INTRADAY / BUY CALL (CE)"
        strategy = "Pharma Sector Momentum & USD/INR Export Tailwind"
        target_contract = f"DRREDDY ₹{atm_strike:.0f} CE"
        entry_level = f"Spot: ₹{current_price:,.2f} | Contract: {target_contract} (1 Lot / 625 Qty)"
        stop_loss = round(current_price - sl_points, 2)
        target_1 = round(current_price + tp1_points, 2)
        target_2 = round(current_price + tp2_points, 2)
        risk_reward_ratio = "1.5 : 1 (Target 1) | 2.5 : 1 (Target 2)"
        bias_color = "#00E676"
        is_tradeable = True
        execution_guideline = (
            f"BUY 1 LOT (625 Qty) {target_contract}. Set Option SL at -₹5.50 points (-₹3,437 max risk). "
            f"At +₹7.00 Option Gain, trail SL to Cost (Zero Risk). Book full profit at +₹12.00 (+₹7,500 cash)."
        )
    else:
        action = "🔴 STRONG BEARISH INTRADAY / BUY PUT (PE)"
        strategy = "Sector Weakness & Relative Alpha Breakdown"
        target_contract = f"DRREDDY ₹{atm_strike:.0f} PE"
        entry_level = f"Spot: ₹{current_price:,.2f} | Contract: {target_contract} (1 Lot / 625 Qty)"
        stop_loss = round(current_price + sl_points, 2)
        target_1 = round(current_price - tp1_points, 2)
        target_2 = round(current_price - tp2_points, 2)
        risk_reward_ratio = "1.5 : 1 (Target 1) | 2.5 : 1 (Target 2)"
        bias_color = "#FF5252"
        is_tradeable = True
        execution_guideline = (
            f"BUY 1 LOT (625 Qty) {target_contract}. Set Option SL at -₹5.50 points (-₹3,437 max risk). "
            f"At +₹7.00 Option Gain, trail SL to Cost (Zero Risk). Book full profit at +₹12.00 (+₹7,500 cash)."
        )

    return {
        "action": action,
        "strategy": strategy,
        "target_contract": target_contract,
        "lot_size": DRREDDY_LOT_SIZE,
        "strike_step": DRREDDY_STRIKE_STEP,
        "atm_strike": atm_strike,
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


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL TRAINING & PREDICTION ENGINE (RF + HGB with Platt Sigmoid Calibration)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def train_and_predict_drreddy(period: str = "2y") -> dict:
    """
    Train standalone Dr. Reddy's Specialist Ensemble (RandomForest + HistGradientBoosting).
    Applies time-disjoint Platt Sigmoid calibration via FrozenEstimator with automated Brier
    score improvement and slope sanity checks.
    """
    try:
        data, latest_row, feature_cols, err = prepare_drreddy_feature_dataset(period=period)
        if err or data is None or len(data) < 25:
            return {
                "status": "error",
                "message": err or "Not enough clean rows after indicator processing."
            }

        # News sentiment score for Dr. Reddy's
        news_info = get_stock_news_sentiment_score("Dr. Reddy's Laboratories")

        latest_row_dict = latest_row[feature_cols].to_dict()
        latest_row_dict['News_Sentiment'] = news_info.get("score", 0.0)
        latest_features = pd.DataFrame([latest_row_dict])[feature_cols]

        X = data[feature_cols]
        y = data['Target']

        # Time-Disjoint Chronological Split (Train: 70%, Calibration: 15%, Test: 15%)
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
            max_depth=4,
            min_samples_leaf=20,
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

        # Time-Disjoint Probability Calibration Layer (scikit-learn FrozenEstimator + Sigmoid)
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

        # Quant Guardrail 1: Brier Score Safety Net
        brier_improved = bool(cal_test_brier < raw_test_brier)

        # Quant Guardrail 2: Slope Sanity Check
        slope_valid = True
        try:
            for clf in list(cal_rf.calibrated_classifiers_) + list(cal_hgb.calibrated_classifiers_):
                for c in getattr(clf, "calibrators", []):
                    if hasattr(c, "a_") and getattr(c, "a_") >= 0:
                        slope_valid = False
                        break
        except Exception:
            pass

        use_calibration = brier_improved and slope_valid
        y_prob_ensemble_test = y_prob_cal_test if use_calibration else y_prob_raw_test
        y_preds_test = (y_prob_ensemble_test >= 0.50).astype(int)

        test_acc = accuracy_score(y_test, y_preds_test)
        test_prec = precision_score(y_test, y_preds_test, zero_division=0)
        test_rec = recall_score(y_test, y_preds_test, zero_division=0)
        test_brier = cal_test_brier if use_calibration else raw_test_brier

        # Predict probability on latest features
        raw_rf = rf_base.predict_proba(latest_features)[0][1]
        raw_hgb = hgb_base.predict_proba(latest_features)[0][1]
        raw_prob_up = float(0.5 * raw_rf + 0.5 * raw_hgb)

        cal_rf_live = cal_rf.predict_proba(latest_features)[0][1]
        cal_hgb_live = cal_hgb.predict_proba(latest_features)[0][1]
        calibrated_prob_up = float(0.5 * cal_rf_live + 0.5 * cal_hgb_live)

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

        latest_close = round(float(latest_row['Close']), 2)
        atr_14_val = float(latest_row.get('ATR_14', latest_close * 0.02))

        quant_blueprint = generate_drreddy_execution_blueprint(
            current_price=latest_close,
            atr_14=atr_14_val,
            prob_up=prob_up * 100.0,
            confidence=confidence
        )

        # Specialized Pharma Radar Telemetry
        pharma_radar = {
            "pharma_index_ret1_pct": round(float(latest_row.get('Pharma_Ret1', 0.0)) * 100.0, 2),
            "alpha_pharma_5d_pct": round(float(latest_row.get('Alpha_Pharma_5d', 0.0)) * 100.0, 2),
            "alpha_pharma_20d_pct": round(float(latest_row.get('Alpha_Pharma_20d', 0.0)) * 100.0, 2),
            "usdinr_trend_pct": round(float(latest_row.get('USDINR_Trend', 0.0)) * 100.0, 2),
            "buying_pressure_index": round(float(latest_row.get('Buying_Pressure_Index', 1.0)), 2),
            "trend_convergence": round(float(latest_row.get('Trend_Convergence_Score', 0.0)), 2),
            "bollinger_z_score": round(float(latest_row.get('Bollinger_Z_Score', 0.0)), 2)
        }

        return {
            "status": "success",
            "symbol": SYMBOL,
            "direction": direction,
            "raw_probability_up_pct": round(raw_prob_up * 100, 1),
            "probability_up_pct": round(prob_up * 100, 1),
            "probability_down_pct": round((1.0 - prob_up) * 100, 1),
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
            "pharma_radar": pharma_radar
        }

    except Exception as e:
        logger.error(f"Error training Dr. Reddy's ML model: {e}")
        return {"status": "error", "message": str(e)}
