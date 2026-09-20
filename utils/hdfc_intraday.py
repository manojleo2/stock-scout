# -*- coding: utf-8 -*-
"""
=============================================================================
 HDFCBANK STANDALONE INTRADAY SPECIALIST MODEL  —  utils/hdfc_intraday.py
=============================================================================
 PURPOSE:
   Dedicated institutional-grade Intraday & Swing Trend Predictor for HDFCBANK.NS.
   Completely standalone from the CDSL intraday model (utils/ml_model.py).

   Uses banking-specific intraday dynamics:
   - Real-time Bank Nifty co-integration & relative strength (alpha/beta spread)
   - US 10-Year Treasury Yields & global financials (FII flow proxy)
   - Wednesday Bank Nifty weekly expiry volatility tracking
   - ATR-based execution blueprint (1.0x ATR Stop Loss, 1.5:1 & 2.5:1 Targets)
   - Standard NSE HDFC Bank Lot Size: 550 shares (₹10 strike steps)
   - Strict 60% Conviction Chop Filter to eliminate sideways noise

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
from concurrent.futures import ThreadPoolExecutor
import logging
import streamlit as st

from utils.data_loader import get_stock_data
from utils.indicators import calculate_technical_indicators
from utils.macro_factors import get_macro_market_cues
from utils.news_sentiment import get_stock_news_sentiment_score
from config import (
    ML_TEST_SIZE, ML_MAX_DEPTH, ML_MIN_SAMPLES_LEAF, ML_N_ESTIMATORS,
    BENCHMARK_TICKER, HDFCBANK_BANKNIFTY_TICKER, HDFCBANK_US_YIELD_TICKER,
    HDFCBANK_LOT_SIZE, HDFCBANK_STRIKE_STEP,
    AI_MIN_CONVICTION_THRESHOLD, AI_ATR_SL_MULT, AI_ATR_TP1_MULT, AI_ATR_TP2_MULT
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hdfc_intraday")

SYMBOL = "HDFCBANK.NS"


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


def _days_to_banknifty_expiry(date_val) -> int:
    """Calendar days to next Wednesday (Bank Nifty weekly index expiry)."""
    if date_val is None:
        d = dt.date.today()
    elif hasattr(date_val, 'date'):
        d = date_val.date()
    elif isinstance(date_val, str):
        try:
            d = dt.datetime.strptime(date_val, "%a, %d %b %Y").date()
        except Exception:
            d = dt.date.today()
    else:
        d = date_val

    days_ahead = (2 - d.weekday()) % 7
    return 7 if days_ahead == 0 else days_ahead


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING: BANKING-SPECIFIC INTRADAY DATASET
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def prepare_hdfc_intraday_dataset(period: str = "2y") -> tuple:
    """
    Construct stationary feature dataset targeting daytime session direction:
    Target_Dir = 1 if Close[t] > Open[t] (Green Intraday Session), else 0 (Red/Flat)

    Banking Features:
      - Bank Nifty 1-day return & SMA50 distance (primary sector driver)
      - HDFC vs Bank Nifty spread (alpha/beta relative strength)
      - US 10-Yr Treasury Yield delta (FII banking flow cue)
      - Wednesday Bank Nifty expiry & Monthly Thursday expiry cycles
      - Close vs VWAP ratio & Wick ratios (intraday price exhaustion)
      - ATR expansion ratio & volume surge
    """
    try:
        # Fetch datasets safely — avoids nested thread pools on Streamlit Cloud 1-vCPU containers
        hdfc_df      = get_stock_data(SYMBOL, period)
        nifty_df     = get_stock_data(BENCHMARK_TICKER, period)
        banknifty_df = get_stock_data(HDFCBANK_BANKNIFTY_TICKER, period)
        macro_df     = get_macro_market_cues(period)

        try:
            us10y_df = get_stock_data(HDFCBANK_US_YIELD_TICKER, period)
        except Exception:
            us10y_df = pd.DataFrame()

        if hdfc_df.empty or len(hdfc_df) < 100:
            return None, None, None, "Insufficient HDFCBANK historical data (need >= 100 trading days)."


        # Technical indicators on HDFC Bank
        df = calculate_technical_indicators(hdfc_df)

        # ── Nifty 50 Benchmark ──────────────────────────────────────────────
        if not nifty_df.empty:
            nifty_df.index = nifty_df.index.tz_localize(None) if nifty_df.index.tz is not None else nifty_df.index
            df['Nifty_Ret1'] = nifty_df['Close'].pct_change(1).reindex(df.index).ffill().fillna(0.0)
            df['Nifty_Dist_SMA50'] = ((nifty_df['Close'] / nifty_df['Close'].rolling(50).mean()) - 1.0).reindex(df.index).ffill().fillna(0.0)
        else:
            df['Nifty_Ret1'] = 0.0
            df['Nifty_Dist_SMA50'] = 0.0

        # ── Bank Nifty Co-Integration (Core Banking Driver) ─────────────────
        if not banknifty_df.empty:
            banknifty_df.index = banknifty_df.index.tz_localize(None) if banknifty_df.index.tz is not None else banknifty_df.index
            bn_ret = banknifty_df['Close'].pct_change(1).reindex(df.index).ffill().fillna(0.0)
            df['BankNifty_Ret1'] = bn_ret
            df['BankNifty_Dist_SMA50'] = ((banknifty_df['Close'] / banknifty_df['Close'].rolling(50).mean()) - 1.0).reindex(df.index).ffill().fillna(0.0)
        else:
            df['BankNifty_Ret1'] = 0.0
            df['BankNifty_Dist_SMA50'] = 0.0

        # ── HDFC vs Bank Nifty Spread (Alpha / Outperformance Spread) ───────
        df['Ret_1'] = df['Close'].pct_change(1)
        df['HDFC_vs_BankNifty'] = (df['Ret_1'] - df['BankNifty_Ret1']).fillna(0.0)

        # ── US 10-Yr Treasury Yield Delta (FII Inflow/Outflow Leading Cue) ──
        if not us10y_df.empty:
            us10y_df.index = us10y_df.index.tz_localize(None) if us10y_df.index.tz is not None else us10y_df.index
            df['US_10Y_Ret1'] = us10y_df['Close'].pct_change(1).reindex(df.index).ffill().fillna(0.0)
        else:
            df['US_10Y_Ret1'] = 0.0

        # ── Global Macro Cues (Drop duplicate columns before join) ──────────
        if not macro_df.empty:
            already_computed = [c for c in macro_df.columns if c in df.columns]
            macro_to_join = macro_df.drop(columns=already_computed, errors='ignore')
            if not macro_to_join.empty:
                df = df.join(macro_to_join, how='left')

        for col in ['SP500_Ret1', 'Nasdaq_Ret1', 'VIX_Norm', 'VIX_Ret1']:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].ffill().fillna(0.0)

        # ── Moving Averages & Oscillators ───────────────────────────────────
        df['SMA_50']  = df['SMA_50'].bfill().ffill().fillna(df['Close'])
        df['SMA_200'] = df['SMA_200'].bfill().ffill().fillna(df['Close'])
        df['Ret_5']   = df['Close'].pct_change(5)
        df['Ret_20']  = df['Close'].pct_change(20)
        df['Dist_SMA50']      = (df['Close'] / df['SMA_50']) - 1.0
        df['Dist_SMA200']     = (df['Close'] / df['SMA_200']) - 1.0
        df['EMA_Cross_Ratio'] = (df['EMA_12'] / df['EMA_26']) - 1.0
        df['RSI_Norm']        = df['RSI_14'] / 100.0
        df['MACD_Hist_Norm']  = df['MACD_Hist'] / df['Close']
        df['BB_PctB']         = df['BB_PctB'].fillna(0.5)
        df['BB_Width']        = df['BB_Width'].fillna(0.0)
        df['Vol_Surge']       = df['Vol_Surge'].fillna(1.0)
        df['Ret_VIX_Interact']= (df['Ret_1'] * df['VIX_Ret1']).fillna(0.0)
        df['News_Sentiment']  = 0.0

        # ── Intraday Session Mechanics (Candle Structure) ───────────────────
        day_range = (df['High'] - df['Low']).replace(0, np.nan)
        df['Close_High_Ratio']   = ((df['Close'] - df['Low']) / day_range).fillna(0.5)
        df['Intraday_Range_Pct'] = ((df['High'] - df['Low']) / df['Close']).fillna(0.0)
        df['Open_To_Close_Ret']  = ((df['Close'] - df['Open']) / df['Open']).fillna(0.0)

        upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Upper_Wick_Ratio'] = (upper_wick / day_range).fillna(0.0)
        lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['Lower_Wick_Ratio'] = (lower_wick / day_range).fillna(0.0)

        # ── Expiry & Calendar Features ──────────────────────────────────────
        df['Days_To_Monthly_Exp']   = [_days_to_monthly_expiry(idx) for idx in df.index]
        df['Days_To_BankNifty_Exp'] = [_days_to_banknifty_expiry(idx) for idx in df.index]
        df['Is_Monday']    = (df.index.dayofweek == 0).astype(float)
        df['Is_Wednesday'] = (df.index.dayofweek == 2).astype(float)
        df['Is_Friday']    = (df.index.dayofweek == 4).astype(float)
        df['Vol_Price_Trend'] = (df['Vol_Surge'] * df['Ret_1']).fillna(0.0)

        df['ATR_Expansion_Ratio'] = (df['ATR_14'] / df['Close']).fillna(0.0)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
        df['Close_VWAP_Ratio'] = ((df['Close'] - typical_price) / typical_price).fillna(0.0)

        # ── TARGET: Next Trading Session Direction (Close[t+1] > Close[t]) ──────────
        df['Target_Dir'] = (df['Close'].shift(-1) > df['Close']).astype(float)

        feature_cols = [
            'Ret_1', 'Ret_5', 'Ret_20',
            'Dist_SMA50', 'Dist_SMA200', 'EMA_Cross_Ratio',
            'RSI_Norm', 'MACD_Hist_Norm', 'BB_PctB', 'BB_Width',
            'Vol_Surge', 'Ret_VIX_Interact', 'News_Sentiment',
            'Nifty_Ret1', 'Nifty_Dist_SMA50',
            'BankNifty_Ret1', 'BankNifty_Dist_SMA50', 'HDFC_vs_BankNifty',
            'US_10Y_Ret1',
            'SP500_Ret1', 'Nasdaq_Ret1', 'VIX_Norm', 'VIX_Ret1',
            'Close_High_Ratio', 'Intraday_Range_Pct', 'Open_To_Close_Ret',
            'Upper_Wick_Ratio', 'Lower_Wick_Ratio',
            'Days_To_Monthly_Exp', 'Days_To_BankNifty_Exp',
            'Is_Monday', 'Is_Wednesday', 'Is_Friday',
            'Vol_Price_Trend', 'ATR_Expansion_Ratio', 'Close_VWAP_Ratio'
        ]

        clean_df = df.dropna(subset=feature_cols).copy()
        train_test_df = clean_df.iloc[:-1].dropna(subset=['Target_Dir']).copy()
        train_test_df['Target_Dir'] = train_test_df['Target_Dir'].astype(int)

        return train_test_df, clean_df.iloc[-1], feature_cols, None


    except Exception as e:
        logger.error(f"Error preparing HDFC intraday dataset: {e}")
        return None, None, None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# 3. QUANT EXECUTION BLUEPRINT (ATR-BASED RISK & LOT SIZING FOR HDFC)
# ─────────────────────────────────────────────────────────────────────────────

def generate_hdfc_intraday_blueprint(current_price: float, atr_14: float,
                                     prob_up: float, conviction: float,
                                     direction: str) -> dict:
    """
    Generate an actionable intraday trade execution plan calibrated specifically
    to HDFC Bank's daily ATR (~₹15 - ₹25) and standard 550 lot size.
    """
    safe_atr = max(atr_14, current_price * 0.008)  # minimum 0.8% floor

    is_tradeable = (conviction >= AI_MIN_CONVICTION_THRESHOLD)

    strike_step = HDFCBANK_STRIKE_STEP
    atm_strike = round(current_price / strike_step) * strike_step

    if not is_tradeable:
        action = "⚪ NEUTRAL / CHOP FILTER ACTIVE"
        strategy = f"Capital Preservation (<{AI_MIN_CONVICTION_THRESHOLD:.0f}% Conviction)"
        bias_color = "#FFB300"
        entry_price = round(current_price, 2)
        stop_loss = round(current_price, 2)
        target_1 = round(current_price, 2)
        target_2 = round(current_price, 2)
        suggested_strike = "None (Avoid Intraday Trade)"
        trade_recommendation = (
            f"Conviction is {conviction:.1f}% (below the {AI_MIN_CONVICTION_THRESHOLD:.0f}% threshold). "
            "High probability of intraday sideways chop. Preserving 100% capital in cash."
        )
    elif "UP" in direction:
        action = "🟢 BUY / LONG (INTRADAY)"
        strategy = "Bullish Banking Trend Momentum"
        bias_color = "#00E676"
        entry_price = round(current_price, 2)
        stop_loss = round(current_price - (safe_atr * AI_ATR_SL_MULT), 2)
        target_1 = round(current_price + (safe_atr * AI_ATR_TP1_MULT), 2)
        target_2 = round(current_price + (safe_atr * AI_ATR_TP2_MULT), 2)
        suggested_strike = f"₹{atm_strike} CE (ATM Call) or Cash Equity"
        trade_recommendation = (
            f"Enter Long above ₹{entry_price:,.2f}. Place Stop Loss at ₹{stop_loss:,.2f} (-₹{entry_price - stop_loss:,.2f}). "
            f"Book 50% profits at Target 1 (₹{target_1:,.2f}) and trail balance to Target 2 (₹{target_2:,.2f})."
        )
    else:
        action = "🔴 SELL / SHORT (INTRADAY)"
        strategy = "Bearish Banking Breakdown / Short"
        bias_color = "#FF5252"
        entry_price = round(current_price, 2)
        stop_loss = round(current_price + (safe_atr * AI_ATR_SL_MULT), 2)
        target_1 = round(current_price - (safe_atr * AI_ATR_TP1_MULT), 2)
        target_2 = round(current_price - (safe_atr * AI_ATR_TP2_MULT), 2)
        suggested_strike = f"₹{atm_strike} PE (ATM Put) or Cash Short"
        trade_recommendation = (
            f"Enter Short below ₹{entry_price:,.2f}. Place Stop Loss at ₹{stop_loss:,.2f} (+₹{stop_loss - entry_price:,.2f}). "
            f"Book 50% profits at Target 1 (₹{target_1:,.2f}) and trail balance to Target 2 (₹{target_2:,.2f})."
        )

    risk_per_share = abs(entry_price - stop_loss) if is_tradeable else 0.0
    reward_per_share = abs(target_1 - entry_price) if is_tradeable else 0.0
    rr_ratio = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 1.5

    return {
        "action": action,
        "strategy": strategy,
        "bias_color": bias_color,
        "entry_price": entry_price,
        "entry_level": f"Above ₹{entry_price:,.2f}" if "BUY" in action else (f"Below ₹{entry_price:,.2f}" if "SELL" in action else f"₹{entry_price:,.2f}"),
        "stop_loss": stop_loss,
        "sl_points": round(abs(entry_price - stop_loss), 2),
        "target_1": target_1,
        "tp1_points": round(abs(target_1 - entry_price), 2),
        "target_2": target_2,
        "tp2_points": round(abs(target_2 - entry_price), 2),
        "suggested_strike": suggested_strike,
        "trade_recommendation": trade_recommendation,
        "execution_guideline": trade_recommendation,
        "entry_window": "9:25 AM – 9:45 AM IST (After 10-min open settles)",
        "exit_window": "Before 3:15 PM IST Same Day (Strict Intraday Exit)",
        "risk_per_share": round(risk_per_share, 2),
        "reward_per_share": round(reward_per_share, 2),
        "risk_reward_ratio": f"{rr_ratio}:1",
        "lot_size": HDFCBANK_LOT_SIZE,
        "strike_step": HDFCBANK_STRIKE_STEP,
        "atm_strike": atm_strike,
        "is_tradeable": is_tradeable,
        "conviction_pct": round(conviction, 1),
        "atr_14": round(safe_atr, 2)
    }



# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN INTRADAY PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def train_and_predict_hdfc_intraday(period: str = "2y") -> dict:
    """
    Train HDFCBANK Intraday Specialist Ensemble (RandomForest + HistGradientBoosting)
    with banking-specific features and recency weighting.
    Returns complete execution dictionary for the AI Up/Down Forecast page.
    """
    try:
        data, latest_row, feature_cols, err = prepare_hdfc_intraday_dataset(period=period)
        if err or data is None or len(data) < 25:
            return {"status": "error", "message": err or "Insufficient clean data for HDFC intraday model."}

        # Live news sentiment for HDFC Bank
        news_info = get_stock_news_sentiment_score("HDFC Bank")
        latest_row_dict = latest_row[feature_cols].to_dict()
        latest_row_dict['News_Sentiment'] = news_info.get("score", 0.0)
        latest_features = pd.DataFrame([latest_row_dict])[feature_cols]

        X = data[feature_cols]
        y = data['Target_Dir']

        split_idx = int(len(X) * (1 - ML_TEST_SIZE))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Exponential recency weighting: recent 2026 market regime has higher weight
        sample_weights_train = np.exp(np.linspace(-0.5, 0.0, len(X_train)))

        rf_model = RandomForestClassifier(
            n_estimators=ML_N_ESTIMATORS,
            max_depth=ML_MAX_DEPTH,
            min_samples_leaf=ML_MIN_SAMPLES_LEAF,
            random_state=42,
            n_jobs=2          # 2 worker threads for Streamlit Cloud stability
        )
        hgb_model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=3,
            min_samples_leaf=15,
            random_state=42
        )

        rf_model.fit(X_train, y_train, sample_weight=sample_weights_train)
        hgb_model.fit(X_train, y_train, sample_weight=sample_weights_train)

        # Ensemble prediction on test split
        pred_rf  = rf_model.predict_proba(X_test)[:, 1]
        pred_hgb = hgb_model.predict_proba(X_test)[:, 1]
        y_prob_ensemble = 0.5 * pred_rf + 0.5 * pred_hgb
        y_preds = (y_prob_ensemble >= 0.50).astype(int)

        test_acc  = accuracy_score(y_test, y_preds)
        test_prec = precision_score(y_test, y_preds, zero_division=0)
        test_rec  = recall_score(y_test, y_preds, zero_division=0)

        # Raw inference on current bar
        raw_rf_prob  = rf_model.predict_proba(latest_features)[0][1]
        raw_hgb_prob = hgb_model.predict_proba(latest_features)[0][1]
        raw_prob_up  = 0.5 * raw_rf_prob + 0.5 * raw_hgb_prob

        # News sentiment nudge (±5% max)
        news_bias   = news_info.get("score", 0.0) * 0.05
        prob_up_raw = float(np.clip(raw_prob_up + news_bias, 0.05, 0.95))

        # Dynamic Calibration Offset from HDFC Intraday Audit
        from utils.hdfc_intraday_audit import calculate_hdfc_intraday_offset
        intraday_offset, offset_reason, recent_lessons = calculate_hdfc_intraday_offset()
        prob_up = float(np.clip(prob_up_raw + (intraday_offset / 100.0), 0.05, 0.95))

        direction = "UP 📈" if prob_up >= 0.50 else "DOWN 📉"
        conviction_pct = max(prob_up, 1.0 - prob_up) * 100.0

        if conviction_pct >= 70.0:
            confidence = "Super High Confidence (HDFC Specialist)"
        elif conviction_pct >= 65.0:
            confidence = "High Confidence"
        elif conviction_pct >= 56.0:
            confidence = "Moderate Confidence"
        else:
            confidence = "Low / Neutral Confidence"

        feature_importances = dict(zip(feature_cols, rf_model.feature_importances_))
        sorted_importances  = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))

        current_price = round(float(latest_row['Close']), 2)
        atr_14        = round(float(latest_row.get('ATR_14', current_price * 0.015)), 2)

        blueprint = generate_hdfc_intraday_blueprint(
            current_price=current_price,
            atr_14=atr_14,
            prob_up=prob_up * 100.0,
            conviction=conviction_pct,
            direction=direction
        )

        banking_radar = {
            "banknifty_ret1":    round(float(latest_row.get('BankNifty_Ret1', 0.0)) * 100, 3),
            "hdfc_vs_bn":        round(float(latest_row.get('HDFC_vs_BankNifty', 0.0)) * 100, 3),
            "us_10y_ret1":       round(float(latest_row.get('US_10Y_Ret1', 0.0)) * 100, 3),
            "days_to_bn_expiry": int(latest_row.get('Days_To_BankNifty_Exp', 0)),
            "days_to_monthly":   int(latest_row.get('Days_To_Monthly_Exp', 0)),
            "is_wednesday":      bool(latest_row.get('Is_Wednesday', 0) == 1.0),
        }

        return {
            "status": "success",
            "symbol": SYMBOL,
            "direction": direction,
            "probability_up_pct":   round(prob_up * 100, 1),
            "probability_down_pct": round((1 - prob_up) * 100, 1),
            "confidence": confidence,
            "test_accuracy_pct": round(test_acc  * 100, 1),
            "precision_pct":     round(test_prec * 100, 1),
            "recall_pct":        round(test_rec  * 100, 1),
            "feature_importances": sorted_importances,
            "sample_count":      len(data),
            "test_sample_count": len(X_test),
            "current_price":     current_price,
            "atr_14":            atr_14,
            "news_info":         news_info,
            "intraday_offset_pct": round(intraday_offset, 1),
            "offset_reason":     offset_reason,
            "recent_lessons":    recent_lessons,
            "blueprint":         blueprint,
            "quant_blueprint":   blueprint,
            "banking_radar":     banking_radar,
            "latest_close":      current_price,
        }


    except Exception as e:
        logger.error(f"Error in HDFC intraday prediction: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
