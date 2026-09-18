import pandas as pd
import numpy as np
import datetime as dt
import calendar
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging

from utils.data_loader import get_stock_data
from utils.indicators import calculate_technical_indicators
from utils.macro_factors import get_macro_market_cues
from utils.news_sentiment import get_stock_news_sentiment_score
from config import (
    BENCHMARK_TICKER, ML_TEST_SIZE, ML_MAX_DEPTH, ML_MIN_SAMPLES_LEAF, 
    ML_N_ESTIMATORS, STOCK_NAME_MAP, MIN_GAP_CONVICTION_THRESHOLD,
    CDSL_MIN_CONVICTION_WEEKDAY, CDSL_MIN_CONVICTION_FRIDAY,
    CDSL_EXPIRY_WEEK_DAYS, CDSL_LOT_SIZE
)

logging.basicConfig(level=logging.INFO)

def get_days_to_monthly_expiry(date_val) -> int:
    """
    Calculate calendar days remaining to the monthly NSE contract expiry (last Thursday).
    """
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
    offset = (last_dt.weekday() - 3) % 7  # 3 = Thursday
    expiry_dt = last_dt - dt.timedelta(days=offset)
    
    # If today is past this month's expiry date, calculate to next month's expiry
    if d > expiry_dt:
        next_m = 1 if month == 12 else month + 1
        next_y = year + 1 if month == 12 else year
        _, next_last_day = calendar.monthrange(next_y, next_m)
        next_last_dt = dt.date(next_y, next_m, next_last_day)
        next_offset = (next_last_dt.weekday() - 3) % 7
        expiry_dt = next_last_dt - dt.timedelta(days=next_offset)
        
    return max(0, (expiry_dt - d).days)

def prepare_opening_gap_dataset(symbol: str, period: str = "2y") -> tuple:
    """
    Construct stationary feature dataset targeting the next-morning opening gap:
    Target_Gap = 1 if Open[t+1] > Close[t] (GAP UP), else 0 (GAP DOWN/FLAT)
    Enriched with 6 single-stock CDSL microstructure & expiry features.
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

    # 3. CDSL SPECIALIST MICROSTRUCTURE & EXPIRY FEATURES
    # Feature 26: Days to Monthly Contract Expiry (rollover & gamma pinning)
    df['Days_To_Expiry'] = [get_days_to_monthly_expiry(idx) for idx in df.index]

    # Feature 27 & 28: Day of week asymmetries (Weekend risk-off vs Monday continuation)
    df['Is_Monday'] = (df.index.dayofweek == 0).astype(float)
    df['Is_Friday'] = (df.index.dayofweek == 4).astype(float)

    # Feature 29: Volume-Price Trend (Institutional accumulation vs distribution)
    df['Vol_Price_Trend'] = (df['Vol_Surge'] * df['Ret_1']).fillna(0.0)

    # Feature 30: Consecutive Gap Streak (Overnight trend exhaustion / mean-reversion)
    prev_gap = np.where(df['Open'] > df['Close'].shift(1), 1, np.where(df['Open'] < df['Close'].shift(1), -1, 0))
    streak = np.zeros(len(df))
    for i in range(1, len(df)):
        if prev_gap[i] == prev_gap[i-1] and prev_gap[i] != 0:
            streak[i] = streak[i-1] + prev_gap[i]
        else:
            streak[i] = prev_gap[i]
    df['Consecutive_Gap_Streak'] = streak

    # Feature 31: Normalized ATR Volatility Expansion Ratio
    df['ATR_Expansion_Ratio'] = (df['ATR_14'] / df['Close']).fillna(0.0)

    # Feature 32: Close to Daily Typical Price / VWAP Proxy Ratio
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    df['Close_VWAP_Ratio'] = ((df['Close'] - typical_price) / typical_price).fillna(0.0)

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
        'Upper_Wick_Ratio', 'Lower_Wick_Ratio',
        'Days_To_Expiry', 'Is_Monday', 'Is_Friday',
        'Vol_Price_Trend', 'Consecutive_Gap_Streak',
        'ATR_Expansion_Ratio', 'Close_VWAP_Ratio'
    ]

    clean_df = df.dropna(subset=feature_cols).copy()
    train_test_df = clean_df.iloc[:-1].dropna(subset=['Target_Gap'])

    return train_test_df, clean_df.iloc[-1], feature_cols, None

def generate_options_trading_call(symbol: str, current_price: float, prob_up: float, confidence: str, date_obj=None) -> dict:
    """
    Generate an actionable Put/Call options trading recommendation for entry at 3:05-3:15 PM.
    Enforces:
    1. Day-of-week threshold: 65% Mon-Thu, 70% Friday (Weekend Theta Shield).
    2. Expiry-week strike shift: Shifts to ITM (Delta ~0.70) when <= 4 days to monthly expiry.
    """
    strike_round = 10 if current_price < 500 else (20 if current_price < 2000 else 50)
    atm_strike = round(current_price / strike_round) * strike_round
    conviction = max(prob_up, 100.0 - prob_up)

    # Determine day-of-week and days to expiry
    days_to_expiry = get_days_to_monthly_expiry(date_obj)
    
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

    is_friday = (today_date.weekday() == 4)
    is_expiry_week = (days_to_expiry <= CDSL_EXPIRY_WEEK_DAYS)

    # Dynamic Conviction Threshold: Strict 70% on Friday, 65% on Weekdays
    active_threshold = CDSL_MIN_CONVICTION_FRIDAY if is_friday else CDSL_MIN_CONVICTION_WEEKDAY

    # Strike Selection: Shift 1-step In-The-Money (ITM) during Expiry Week to protect Delta & mitigate Theta decay
    if is_expiry_week:
        call_strike = atm_strike - strike_round  # ITM Call (e.g. 1320 CE instead of 1340 CE)
        put_strike = atm_strike + strike_round   # ITM Put (e.g. 1360 PE instead of 1340 PE)
        strike_desc_call = f"₹{call_strike} CE (In-The-Money Call — Expiry Armor)"
        strike_desc_put = f"₹{put_strike} PE (In-The-Money Put — Expiry Armor)"
        strategy_suffix = " [ITM Expiry Armor Active]"
    else:
        call_strike = atm_strike
        put_strike = atm_strike
        strike_desc_call = f"₹{atm_strike} CE (At-The-Money Call)"
        strike_desc_put = f"₹{atm_strike} PE (At-The-Money Put)"
        strategy_suffix = ""

    if conviction < active_threshold:
        action = "⚪ NEUTRAL / NO TRADE"
        if is_friday:
            strategy = "Weekend Theta Shield Active (<70% Conviction)"
            risk_guideline = f"Friday carry requires ≥{CDSL_MIN_CONVICTION_FRIDAY:.0f}% conviction to overcome 3-day weekend theta decay. Current conviction is {conviction:.1f}% — 100% Cash Preserved."
        else:
            strategy = f"Overnight Capital Preservation (<{active_threshold:.0f}% Conviction)"
            risk_guideline = f"Conviction below {active_threshold:.0f}% threshold. Preserving capital against overnight theta decay."
        suggested_strike = "None (Avoid Options Overnight)"
        alt_strike = "Wait for 9:15 AM Cash Open"
        entry_window = "No Entry Recommended"
        exit_window = "N/A"
        bias_color = "#FFB300"
        is_tradeable = False
    elif prob_up >= active_threshold:
        action = "🟢 BUY CALL (CE)"
        strategy = f"Bullish Overnight Gap Carry{strategy_suffix}"
        suggested_strike = strike_desc_call
        alt_strike = f"₹{call_strike + strike_round} CE (ATM Alternative)" if is_expiry_week else f"₹{call_strike + strike_round} CE (Slightly OTM)"
        entry_window = "3:08 PM - 3:20 PM IST Today"
        exit_window = "9:15 AM - 9:25 AM IST Tomorrow (First 5-10 min)"
        risk_guideline = "Exit immediately if market fails to gap up by 9:20 AM. Do not carry into intraday chop."
        bias_color = "#00E676"
        is_tradeable = True
    else:
        action = "🔴 BUY PUT (PE)"
        strategy = f"Bearish Overnight Gap Down Carry{strategy_suffix}"
        suggested_strike = strike_desc_put
        alt_strike = f"₹{put_strike - strike_round} PE (ATM Alternative)" if is_expiry_week else f"₹{put_strike - strike_round} PE (Slightly OTM)"
        entry_window = "3:08 PM - 3:20 PM IST Today"
        exit_window = "9:15 AM - 9:25 AM IST Tomorrow (First 5-10 min)"
        risk_guideline = "Lock profits on opening drop. Exit if stock opens flat or green."
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
        "days_to_expiry": days_to_expiry,
        "is_expiry_week": is_expiry_week,
        "active_threshold": active_threshold,
        "lot_size": CDSL_LOT_SIZE
    }

def predict_opening_gap(symbol: str, period: str = "2y") -> dict:
    """
    Train Specialist Ensemble Classifier (RandomForest + HistGradientBoosting) with 
    exponential recency sample weighting and microstructure features.
    """
    try:
        data, latest_row, feature_cols, err = prepare_opening_gap_dataset(symbol, period=period)
        if err or data is None or len(data) < 25:
            return {
                "status": "error",
                "message": err or "Not enough clean rows after gap indicator processing."
            }

        # Fetch real-time news sentiment
        clean_stock_name = STOCK_NAME_MAP.get(symbol, symbol.replace(".NS", "").replace(".BO", ""))
        news_info = get_stock_news_sentiment_score(clean_stock_name)

        # Inject news score into latest feature row
        latest_row_dict = latest_row[feature_cols].to_dict()
        latest_row_dict['News_Sentiment'] = news_info.get("score", 0.0)
        latest_features = np.array([latest_row_dict[col] for col in feature_cols]).reshape(1, -1)

        X = data[feature_cols]
        y = data['Target_Gap']

        # Chronological train/test split
        split_idx = int(len(X) * (1 - ML_TEST_SIZE))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Exponential recency sample weighting: Gives 60% higher importance to recent 2026 regimes
        sample_weights_train = np.exp(np.linspace(-0.5, 0.0, len(X_train)))

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

        rf_model.fit(X_train, y_train, sample_weight=sample_weights_train)
        hgb_model.fit(X_train, y_train, sample_weight=sample_weights_train)

        # Out-of-sample evaluation on gap prediction
        pred_rf = rf_model.predict_proba(X_test)[:, 1]
        pred_hgb = hgb_model.predict_proba(X_test)[:, 1]
        y_prob_ensemble = 0.5 * pred_rf + 0.5 * pred_hgb
        y_preds = (y_prob_ensemble >= 0.50).astype(int)

        test_acc = accuracy_score(y_test, y_preds)
        test_prec = precision_score(y_test, y_preds, zero_division=0)
        test_rec = recall_score(y_test, y_preds, zero_division=0)

        # Predict probability incorporating live news bias
        raw_rf_prob = rf_model.predict_proba(latest_features)[0][1]
        raw_hgb_prob = hgb_model.predict_proba(latest_features)[0][1]
        raw_prob_up = 0.5 * raw_rf_prob + 0.5 * raw_hgb_prob
        news_bias = news_info.get("score", 0.0) * 0.05
        prob_up_raw = float(np.clip(raw_prob_up + news_bias, 0.05, 0.95))

        # Recalibration from past opening gap memory if available
        from utils.opening_audit import calculate_gap_recalibration_offset
        gap_offset, gap_reason = calculate_gap_recalibration_offset(symbol)
        prob_up = float(np.clip(prob_up_raw + (gap_offset / 100.0), 0.05, 0.95))

        direction = "GAP UP 📈" if prob_up >= 0.50 else "GAP DOWN 📉"
        conviction_pct = max(prob_up, 1.0 - prob_up) * 100.0

        if conviction_pct >= 70.0:
            confidence = "Super High Confidence (Specialist)"
        elif conviction_pct >= 65.0:
            confidence = "High Confidence"
        elif conviction_pct >= 56.0:
            confidence = "Moderate Confidence"
        else:
            confidence = "Low / Neutral Confidence"

        feature_importances = dict(zip(feature_cols, rf_model.feature_importances_))
        sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

        current_price = round(latest_row['Close'], 2)
        today_date = latest_row.name if hasattr(latest_row, 'name') else dt.date.today()
        options_call = generate_options_trading_call(symbol, current_price, prob_up * 100.0, confidence, date_obj=today_date)

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
            "options_call": options_call,
            "days_to_expiry": options_call.get("days_to_expiry", 0),
            "is_expiry_week": options_call.get("is_expiry_week", False),
            "is_friday": options_call.get("is_friday", False)
        }
    except Exception as e:
        logging.error(f"Error predicting opening gap for {symbol}: {e}")
        return {"status": "error", "message": str(e)}
