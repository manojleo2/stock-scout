# -*- coding: utf-8 -*-
"""
=============================================================================
 HDFCBANK STANDALONE SPECIALIST MODEL  —  utils/hdfc_specialist.py
=============================================================================
 PURPOSE:
   Dedicated institutional-grade overnight gap predictor for HDFCBANK.NS.
   Completely standalone from the CDSL predictor (opening_predictor.py).
   Uses banking-specific microstructure: Bank Nifty co-integration,
   US 10-Year Treasury Yield (FII lead indicator), Wednesday Bank Nifty
   expiry dynamics, and HDFC-specific pre-close delivery flow features.

 RULE: THIS FILE DOES NOT IMPORT FROM opening_predictor.py OR opening_audit.py
       Those files belong exclusively to the CDSL specialist.
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
    STOCK_NAME_MAP,
    HDFCBANK_MIN_CONVICTION_WEEKDAY, HDFCBANK_MIN_CONVICTION_FRIDAY,
    HDFCBANK_EXPIRY_WEEK_DAYS, HDFCBANK_LOT_SIZE, HDFCBANK_STRIKE_STEP,
    HDFCBANK_BANKNIFTY_TICKER, HDFCBANK_US_YIELD_TICKER,
    BENCHMARK_TICKER
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hdfc_specialist")

SYMBOL = "HDFCBANK.NS"


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: Monthly expiry calculator (same logic, fully standalone copy)
# ─────────────────────────────────────────────────────────────────────────────

def _days_to_monthly_expiry(date_val) -> int:
    """Calendar days to last Thursday of the month (NSE stock option expiry)."""
    if date_val is None:
        d = dt.date.today()
    elif isinstance(date_val, str):
        try:
            d = dt.datetime.strptime(date_val, "%a, %d %b %Y").date()
        except Exception:
            try:
                d = dt.datetime.strptime(date_val, "%Y-%m-%d").date()
            except Exception:
                d = dt.date.today()
    elif hasattr(date_val, 'date'):
        d = date_val.date()
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
    """
    Calendar days to next Bank Nifty weekly expiry (every Wednesday).
    Bank Nifty index options expire every Wednesday — creates additional
    volatility and gap pressure on HDFCBANK as the index's largest constituent.
    """
    if date_val is None:
        d = dt.date.today()
    elif hasattr(date_val, 'date'):
        d = date_val.date()
    elif isinstance(date_val, str):
        try:
            d = dt.datetime.strptime(date_val, "%Y-%m-%d").date()
        except Exception:
            d = dt.date.today()
    else:
        d = date_val

    # Find next Wednesday (weekday 2)
    days_ahead = (2 - d.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7  # already Wednesday → go to next one
    return days_ahead


# ─────────────────────────────────────────────────────────────────────────────
# CORE: HDFC-SPECIFIC FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def prepare_hdfc_gap_dataset(period: str = "2y") -> tuple:
    """
    Build the HDFCBANK-specific feature dataset for overnight gap prediction.
    Features are purpose-built for the Indian private banking sector:

    Base Features (shared with CDSL architecture):
      Ret_1/5/20, RSI, MACD, BB, SMA distance, Vol Surge, Nifty cues,
      SP500/Nasdaq/VIX macro, Wick ratios, Close/High ratio, ATR expansion

    HDFC-Specific Banking Features (NEW — not in CDSL model):
      BankNifty_Ret1        : Bank Nifty 1-day return (primary sector driver)
      BankNifty_Dist_SMA50  : Bank Nifty vs its 50-day SMA (institutional trend)
      HDFC_vs_BankNifty     : Divergence of HDFC return vs Bank Nifty (alpha/beta spread)
      US_10Y_Ret1           : US 10-Year Treasury yield daily change (FII flow proxy)
      Days_To_BankNifty_Exp : Sessions to next Bank Nifty weekly Wednesday expiry
      Is_Wednesday          : Wednesday = Bank Nifty expiry day (gamma spike risk)
      Is_Monday             : Monday risk-on/risk-off carry from US weekend
      Is_Friday             : Friday = 3-day theta risk for banking options
      Days_To_Monthly_Exp   : Sessions to HDFCBANK monthly last-Thursday expiry
      Vol_Price_Trend       : Vol × Return (institutional delivery accumulation proxy)
      Consecutive_Gap_Streak: Gap directional exhaustion (mean-reversion signal)
      ATR_Expansion_Ratio   : Volatility expansion ahead of close
      Close_VWAP_Ratio      : 3PM close position vs intraday VWAP (institutional bias)
    """
    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            f_hdfc     = executor.submit(get_stock_data, SYMBOL, period)
            f_nifty    = executor.submit(get_stock_data, BENCHMARK_TICKER, period)
            f_banknifty= executor.submit(get_stock_data, HDFCBANK_BANKNIFTY_TICKER, period)
            f_us10y    = executor.submit(get_stock_data, HDFCBANK_US_YIELD_TICKER, period)
            f_macro    = executor.submit(get_macro_market_cues, period)

            hdfc_df     = f_hdfc.result()
            nifty_df    = f_nifty.result()
            banknifty_df= f_banknifty.result()
            us10y_df    = f_us10y.result()
            macro_df    = f_macro.result()

        if hdfc_df.empty or len(hdfc_df) < 100:
            return None, None, None, "Insufficient HDFCBANK historical data (need >= 100 trading days)."

        df = calculate_technical_indicators(hdfc_df)

        # ── Nifty 50 Benchmark ──────────────────────────────────────────────
        if not nifty_df.empty:
            nifty_df.index = nifty_df.index.tz_localize(None) if nifty_df.index.tz is not None else nifty_df.index
            df['Nifty_Ret1'] = nifty_df['Close'].pct_change(1).reindex(df.index).ffill().fillna(0.0)
            df['Nifty_Dist_SMA50'] = ((nifty_df['Close'] / nifty_df['Close'].rolling(50).mean()) - 1.0).reindex(df.index).ffill().fillna(0.0)
        else:
            df['Nifty_Ret1'] = 0.0
            df['Nifty_Dist_SMA50'] = 0.0

        # ── Bank Nifty Co-Integration (HDFC-specific anchor) ────────────────
        if not banknifty_df.empty:
            banknifty_df.index = banknifty_df.index.tz_localize(None) if banknifty_df.index.tz is not None else banknifty_df.index
            bn_ret = banknifty_df['Close'].pct_change(1).reindex(df.index).ffill().fillna(0.0)
            df['BankNifty_Ret1'] = bn_ret
            df['BankNifty_Dist_SMA50'] = ((banknifty_df['Close'] / banknifty_df['Close'].rolling(50).mean()) - 1.0).reindex(df.index).ffill().fillna(0.0)
        else:
            df['BankNifty_Ret1'] = 0.0
            df['BankNifty_Dist_SMA50'] = 0.0

        # ── HDFC vs Bank Nifty Spread (alpha/beta divergence) ───────────────
        df['Ret_1'] = df['Close'].pct_change(1)
        df['HDFC_vs_BankNifty'] = (df['Ret_1'] - df['BankNifty_Ret1']).fillna(0.0)

        # ── US 10-Year Treasury Yield (FII banking flow leading indicator) ──
        if not us10y_df.empty:
            us10y_df.index = us10y_df.index.tz_localize(None) if us10y_df.index.tz is not None else us10y_df.index
            df['US_10Y_Ret1'] = us10y_df['Close'].pct_change(1).reindex(df.index).ffill().fillna(0.0)
        else:
            df['US_10Y_Ret1'] = 0.0

        # ── Global Macro Cues ────────────────────────────────────────────────
        if not macro_df.empty:
            # Drop any columns we already computed ourselves to prevent 'columns overlap' crash.
            # macro_df from get_macro_market_cues() can include BankNifty_Ret1 which we
            # already computed directly from raw Bank Nifty data with higher precision.
            already_computed = [c for c in macro_df.columns if c in df.columns]
            macro_to_join = macro_df.drop(columns=already_computed, errors='ignore')
            if not macro_to_join.empty:
                df = df.join(macro_to_join, how='left')
        for col in ['SP500_Ret1', 'Nasdaq_Ret1', 'VIX_Norm', 'VIX_Ret1']:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].ffill().fillna(0.0)

        # ── SMA fill ────────────────────────────────────────────────────────
        df['SMA_50']  = df['SMA_50'].bfill().ffill().fillna(df['Close'])
        df['SMA_200'] = df['SMA_200'].bfill().ffill().fillna(df['Close'])

        # ── Returns & Oscillators ────────────────────────────────────────────
        df['Ret_5']  = df['Close'].pct_change(5)
        df['Ret_20'] = df['Close'].pct_change(20)
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

        # ── Overnight Gap Microstructure ─────────────────────────────────────
        day_range = (df['High'] - df['Low']).replace(0, np.nan)
        df['Close_High_Ratio']   = ((df['Close'] - df['Low']) / day_range).fillna(0.5)
        df['Intraday_Range_Pct'] = ((df['High'] - df['Low']) / df['Close']).fillna(0.0)
        df['Open_To_Close_Ret']  = ((df['Close'] - df['Open']) / df['Open']).fillna(0.0)
        upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Upper_Wick_Ratio'] = (upper_wick / day_range).fillna(0.0)
        lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['Lower_Wick_Ratio'] = (lower_wick / day_range).fillna(0.0)

        # ── HDFC-Specific Banking Expiry & Regime Features ───────────────────
        df['Days_To_Monthly_Exp']   = [_days_to_monthly_expiry(idx) for idx in df.index]
        df['Days_To_BankNifty_Exp'] = [_days_to_banknifty_expiry(idx) for idx in df.index]
        df['Is_Monday']    = (df.index.dayofweek == 0).astype(float)
        df['Is_Wednesday'] = (df.index.dayofweek == 2).astype(float)
        df['Is_Friday']    = (df.index.dayofweek == 4).astype(float)
        df['Vol_Price_Trend'] = (df['Vol_Surge'] * df['Ret_1']).fillna(0.0)

        # Gap streak (directional exhaustion / mean-reversion signal)
        prev_gap = np.where(df['Open'] > df['Close'].shift(1), 1,
                   np.where(df['Open'] < df['Close'].shift(1), -1, 0))
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

        # ── TARGET: Next day 9:15 AM Open > Today 3:30 PM Close ─────────────
        next_open = df['Open'].shift(-1)
        df['Target_Gap'] = (next_open > df['Close']).astype(int)

        feature_cols = [
            # Base momentum & oscillators
            'Ret_1', 'Ret_5', 'Ret_20',
            'Dist_SMA50', 'Dist_SMA200', 'EMA_Cross_Ratio',
            'RSI_Norm', 'MACD_Hist_Norm', 'BB_PctB', 'BB_Width',
            'Vol_Surge', 'Ret_VIX_Interact', 'News_Sentiment',
            # Macro & global
            'Nifty_Ret1', 'Nifty_Dist_SMA50',
            'SP500_Ret1', 'Nasdaq_Ret1', 'VIX_Norm', 'VIX_Ret1',
            # HDFC Banking-Specific
            'BankNifty_Ret1', 'BankNifty_Dist_SMA50', 'HDFC_vs_BankNifty',
            'US_10Y_Ret1',
            # Microstructure & overnight gap drivers
            'Close_High_Ratio', 'Intraday_Range_Pct', 'Open_To_Close_Ret',
            'Upper_Wick_Ratio', 'Lower_Wick_Ratio',
            # HDFC Expiry & regime
            'Days_To_Monthly_Exp', 'Days_To_BankNifty_Exp',
            'Is_Monday', 'Is_Wednesday', 'Is_Friday',
            'Vol_Price_Trend', 'Consecutive_Gap_Streak',
            'ATR_Expansion_Ratio', 'Close_VWAP_Ratio',
        ]

        clean_df = df.dropna(subset=feature_cols).copy()
        train_test_df = clean_df.iloc[:-1].dropna(subset=['Target_Gap'])

        return train_test_df, clean_df.iloc[-1], feature_cols, None

    except Exception as e:
        logger.error(f"Error preparing HDFC gap dataset: {e}")
        return None, None, None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONS CALL GENERATOR (HDFC-specific parameters)
# ─────────────────────────────────────────────────────────────────────────────

def generate_hdfc_options_call(current_price: float, prob_up: float,
                                confidence: str, date_obj=None) -> dict:
    """
    Generate actionable HDFC Bank Put/Call options recommendation for 3:08 PM entry.
    Uses HDFCBANK-specific lot size (550), ₹10 strike steps, and Friday/Expiry shields.
    """
    strike_step = HDFCBANK_STRIKE_STEP
    atm_strike = round(current_price / strike_step) * strike_step
    conviction = max(prob_up, 100.0 - prob_up)

    days_to_exp = _days_to_monthly_expiry(date_obj)
    days_to_bn_exp = _days_to_banknifty_expiry(date_obj)

    if date_obj is None:
        today_date = dt.date.today()
    elif isinstance(date_obj, str):
        try:
            today_date = dt.datetime.strptime(date_obj, "%a, %d %b %Y").date()
        except Exception:
            try:
                today_date = dt.datetime.strptime(date_obj, "%Y-%m-%d").date()
            except Exception:
                today_date = dt.date.today()
    elif hasattr(date_obj, 'date'):
        today_date = date_obj.date()
    else:
        today_date = date_obj

    is_friday    = (today_date.weekday() == 4)
    is_wednesday = (today_date.weekday() == 2)
    is_expiry_week = (days_to_exp <= HDFCBANK_EXPIRY_WEEK_DAYS)

    active_threshold = HDFCBANK_MIN_CONVICTION_FRIDAY if is_friday else HDFCBANK_MIN_CONVICTION_WEEKDAY

    # ITM Expiry Armor — shift strikes by 1 step during monthly expiry week
    if is_expiry_week:
        call_strike = atm_strike - strike_step
        put_strike  = atm_strike + strike_step
        strike_desc_call = f"₹{call_strike} CE (ITM Call — Monthly Expiry Armor)"
        strike_desc_put  = f"₹{put_strike} PE (ITM Put — Monthly Expiry Armor)"
        strategy_suffix  = " [ITM Monthly Expiry Armor Active]"
    else:
        call_strike = atm_strike
        put_strike  = atm_strike
        strike_desc_call = f"₹{atm_strike} CE (ATM Call)"
        strike_desc_put  = f"₹{atm_strike} PE (ATM Put)"
        strategy_suffix  = ""

    # Wednesday Bank Nifty Expiry Warning
    bn_expiry_note = ""
    if is_wednesday:
        bn_expiry_note = " ⚠️ Bank Nifty Wednesday Expiry — Elevated Gamma Volatility"

    if conviction < active_threshold:
        action = "⚪ NEUTRAL / NO TRADE"
        if is_friday:
            strategy = f"Weekend Theta Shield Active (<{HDFCBANK_MIN_CONVICTION_FRIDAY:.0f}% Conviction)"
            risk_guideline = (f"Friday banking options carry 3-day weekend theta decay. "
                              f"Requires ≥{HDFCBANK_MIN_CONVICTION_FRIDAY:.0f}% conviction. "
                              f"Current: {conviction:.1f}% — 100% Cash Preserved.")
        else:
            strategy = f"Overnight Capital Preservation (<{active_threshold:.0f}% Conviction)"
            risk_guideline = (f"Conviction {conviction:.1f}% below {active_threshold:.0f}% threshold. "
                              "Preserving capital against overnight banking theta decay.")
        suggested_strike = "None (Avoid Options Overnight)"
        alt_strike = "Wait for 9:15 AM Cash Open"
        entry_window = "No Entry Recommended"
        exit_window = "N/A"
        bias_color = "#FFB300"
        is_tradeable = False

    elif prob_up >= active_threshold:
        action = "🟢 BUY CALL (CE)"
        strategy = f"Bullish HDFC Overnight Gap Carry{strategy_suffix}{bn_expiry_note}"
        suggested_strike = strike_desc_call
        alt_strike = f"₹{call_strike + strike_step} CE (Slightly OTM Alternative)"
        entry_window = "3:08 PM – 3:20 PM IST Today"
        exit_window = "9:15 AM – 9:20 AM IST Tomorrow (First 5 min)"
        risk_guideline = "Exit within 5 min of 9:15 AM open. Do not carry HDFC options into intraday chop."
        bias_color = "#00E676"
        is_tradeable = True

    else:
        action = "🔴 BUY PUT (PE)"
        strategy = f"Bearish HDFC Overnight Gap Down Carry{strategy_suffix}{bn_expiry_note}"
        suggested_strike = strike_desc_put
        alt_strike = f"₹{put_strike - strike_step} PE (Slightly OTM Alternative)"
        entry_window = "3:08 PM – 3:20 PM IST Today"
        exit_window = "9:15 AM – 9:20 AM IST Tomorrow (First 5 min)"
        risk_guideline = "Lock profits immediately at 9:15 AM open drop. Exit flat if HDFC opens green."
        bias_color = "#FF5252"
        is_tradeable = True

    return {
        "action": action,
        "strategy": strategy,
        "suggested_strike": suggested_strike,
        "alt_strike": alt_strike,
        "entry_window": entry_window,
        "exit_window": exit_window,
        "risk_guideline": risk_guideline,
        "bias_color": bias_color,
        "atm_strike": atm_strike,
        "selected_strike": call_strike if prob_up >= active_threshold else put_strike,
        "conviction_pct": round(conviction, 1),
        "is_tradeable": is_tradeable,
        "is_friday": is_friday,
        "is_wednesday": is_wednesday,
        "is_expiry_week": is_expiry_week,
        "days_to_monthly_expiry": days_to_exp,
        "days_to_banknifty_expiry": days_to_bn_exp,
        "active_threshold": active_threshold,
        "lot_size": HDFCBANK_LOT_SIZE,
        "strike_step": HDFCBANK_STRIKE_STEP,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def predict_hdfc_opening_gap(period: str = "2y") -> dict:
    """
    Train the HDFCBANK Standalone Specialist Ensemble (RandomForest + HistGradientBoosting)
    with exponential recency weighting and banking-specific microstructure features.
    Returns full prediction dict compatible with the opening_prediction view router.
    """
    try:
        data, latest_row, feature_cols, err = prepare_hdfc_gap_dataset(period=period)
        if err or data is None or len(data) < 25:
            return {"status": "error", "message": err or "Insufficient clean data for HDFC model training."}

        # Live news sentiment for HDFC Bank
        news_info = get_stock_news_sentiment_score("HDFC Bank")
        latest_row_dict = latest_row[feature_cols].to_dict()
        latest_row_dict['News_Sentiment'] = news_info.get("score", 0.0)
        latest_features = pd.DataFrame([latest_row_dict])[feature_cols]

        X = data[feature_cols]
        y = data['Target_Gap']

        split_idx = int(len(X) * (1 - ML_TEST_SIZE))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Exponential recency weighting: recent 2026 regime matters more than 2024
        sample_weights_train = np.exp(np.linspace(-0.5, 0.0, len(X_train)))

        rf_model = RandomForestClassifier(
            n_estimators=ML_N_ESTIMATORS,
            max_depth=ML_MAX_DEPTH,
            min_samples_leaf=ML_MIN_SAMPLES_LEAF,
            random_state=42,
            n_jobs=2          # capped at 2 for Streamlit Cloud 1-vCPU safety
        )
        hgb_model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=3,
            min_samples_leaf=15,
            random_state=42
        )

        rf_model.fit(X_train, y_train, sample_weight=sample_weights_train)
        hgb_model.fit(X_train, y_train, sample_weight=sample_weights_train)

        # Ensemble averaging
        pred_rf  = rf_model.predict_proba(X_test)[:, 1]
        pred_hgb = hgb_model.predict_proba(X_test)[:, 1]
        y_prob_ensemble = 0.5 * pred_rf + 0.5 * pred_hgb
        y_preds = (y_prob_ensemble >= 0.50).astype(int)

        test_acc  = accuracy_score(y_test, y_preds)
        test_prec = precision_score(y_test, y_preds, zero_division=0)
        test_rec  = recall_score(y_test, y_preds, zero_division=0)

        raw_rf_prob  = rf_model.predict_proba(latest_features)[0][1]
        raw_hgb_prob = hgb_model.predict_proba(latest_features)[0][1]
        raw_prob_up  = 0.5 * raw_rf_prob + 0.5 * raw_hgb_prob

        # News sentiment nudge (±5% max)
        news_bias   = news_info.get("score", 0.0) * 0.05
        prob_up_raw = float(np.clip(raw_prob_up + news_bias, 0.05, 0.95))

        # Self-Learning Calibration Offset (from hdfc_learning_audit)
        from utils.hdfc_learning_audit import calculate_hdfc_learning_offset
        learning_offset, learning_reason, recent_lessons = calculate_hdfc_learning_offset()
        prob_up = float(np.clip(prob_up_raw + (learning_offset / 100.0), 0.05, 0.95))

        direction = "GAP UP 📈" if prob_up >= 0.50 else "GAP DOWN 📉"
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
        today_date    = latest_row.name if hasattr(latest_row, 'name') else dt.date.today()

        options_call = generate_hdfc_options_call(
            current_price=current_price,
            prob_up=prob_up * 100.0,
            confidence=confidence,
            date_obj=today_date
        )

        # Supplemental banking radar telemetry for the UI
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
            "news_info":         news_info,
            "gap_offset_pct":    round(learning_offset, 1),
            "gap_reason":        learning_reason,
            "recent_lessons":    recent_lessons,
            "options_call":      options_call,
            "banking_radar":     banking_radar,
            "days_to_expiry":    options_call.get("days_to_monthly_expiry", 0),
            "is_expiry_week":    options_call.get("is_expiry_week", False),
            "is_friday":         options_call.get("is_friday", False),
            "is_wednesday":      options_call.get("is_wednesday", False),
        }

    except Exception as e:
        logger.error(f"Error in HDFC opening gap prediction: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
