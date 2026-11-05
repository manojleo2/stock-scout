import pandas as pd
import numpy as np

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate full suite of technical indicators on an OHLCV DataFrame using pure pandas/numpy.
    """
    if df.empty or 'Close' not in df.columns:
        return df

    data = df.copy()
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']

    # 1. Simple Moving Averages
    data['SMA_20'] = close.rolling(window=20).mean()
    data['SMA_50'] = close.rolling(window=50).mean()
    data['SMA_200'] = close.rolling(window=200).mean()

    # 2. Exponential Moving Averages
    data['EMA_12'] = close.ewm(span=12, adjust=False).mean()
    data['EMA_26'] = close.ewm(span=26, adjust=False).mean()
    data['EMA_50'] = close.ewm(span=50, adjust=False).mean()

    # 3. RSI (14) using Wilder's Exponential Smoothing
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data['RSI_14'] = 100 - (100 / (1 + rs))

    # 4. MACD (12, 26, 9)
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    # 5. Bollinger Bands (20, 2 std)
    data['BB_Middle'] = close.rolling(window=20).mean()
    rolling_std = close.rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (rolling_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (rolling_std * 2)
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
    data['BB_PctB'] = (close - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

    # 6. Average True Range (ATR 14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['ATR_14'] = tr.ewm(span=14, adjust=False).mean()

    # 7. Volume Moving Average & Surge Ratio
    data['Vol_SMA20'] = volume.rolling(window=20).mean()
    data['Vol_Surge'] = volume / data['Vol_SMA20'].replace(0, np.nan)

    # 8. ADX (14) — Trend Strength / Regime Detector
    # ADX > 25 = trending (momentum signals reliable)
    # ADX < 20 = choppy (mean-reversion zone, suppress trend signals)
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # Zero out when the other side is bigger
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
    atr_smooth = tr.ewm(span=14, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    dx_sum   = (plus_di + minus_di).replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / dx_sum
    data['ADX_14']    = dx.ewm(span=14, adjust=False).mean()
    data['Plus_DI']   = plus_di
    data['Minus_DI']  = minus_di

    # 9. Opening Gap % vs Previous Close
    data['Gap_Pct'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)

    # 10. Prior Day High / Low distance (Support & Resistance proxy)
    data['PDH_Dist'] = (data['Close'] - data['High'].shift(1)) / data['Close']   # negative = below PDH
    data['PDL_Dist'] = (data['Close'] - data['Low'].shift(1))  / data['Close']   # positive = above PDL

    # 11. Upper & Lower Wick Ratios (Rejection signal — previously listed in feature list but never computed)
    body       = (data['Close'] - data['Open']).abs()
    day_range  = (high - low).replace(0, np.nan)
    data['Upper_Wick_Ratio'] = (high - data[['Open', 'Close']].max(axis=1)) / day_range
    data['Lower_Wick_Ratio'] = (data[['Open', 'Close']].min(axis=1) - low) / day_range
    data['Upper_Wick_Ratio'] = data['Upper_Wick_Ratio'].fillna(0.0)
    data['Lower_Wick_Ratio'] = data['Lower_Wick_Ratio'].fillna(0.0)

    # 12. On-Balance Volume (OBV) — 5-day slope as accumulation/distribution signal
    direction_sign = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    obv = (volume * direction_sign).cumsum()
    data['OBV'] = obv
    obv_sma5 = pd.Series(obv).rolling(5).mean()
    data['OBV_Slope'] = (obv - obv_sma5) / (obv_sma5.replace(0, np.nan).abs())   # normalised OBV momentum

    # 13. Supertrend (ATR-based trend direction: +1 bullish, -1 bearish)
    multiplier = 3.0
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * data['ATR_14']
    lower_band = hl2 - multiplier * data['ATR_14']
    # Iterative Supertrend construction
    supertrend = pd.Series(np.nan, index=close.index)
    direction  = pd.Series(np.nan, index=close.index)
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    for i in range(1, len(close)):
        # Upper band
        if upper_band.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]
        # Lower band
        if lower_band.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]
        # Direction: +1 = bullish (above lower band), -1 = bearish (below upper band)
        if pd.isna(direction.iloc[i - 1]):
            direction.iloc[i] = 1 if close.iloc[i] >= final_upper.iloc[i] else -1
        elif direction.iloc[i - 1] == -1 and close.iloc[i] > final_upper.iloc[i]:
            direction.iloc[i] = 1
        elif direction.iloc[i - 1] == 1 and close.iloc[i] < final_lower.iloc[i]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
    data['Supertrend_Dir'] = direction.fillna(0)   # +1 bullish, -1 bearish, 0 unknown


    # 14. Stochastic Oscillator (14, 3, 3)
    low_14 = low.rolling(window=14).min()
    high_14 = high.rolling(window=14).max()
    denom = (high_14 - low_14).replace(0, np.nan)
    stoch_k = 100.0 * ((close - low_14) / denom)
    data['Stoch_K'] = stoch_k
    data['Stoch_D'] = stoch_k.rolling(window=3).mean()
    data['Stoch_K_Norm'] = (data['Stoch_K'] / 100.0).fillna(0.5)
    data['Stoch_D_Norm'] = (data['Stoch_D'] / 100.0).fillna(0.5)

    # 15. Rolling 20-day Volume-Weighted Average Price (VWAP)
    typical_price = (high + low + close) / 3.0
    cum_pv = (typical_price * volume).rolling(window=20).sum()
    cum_vol = volume.rolling(window=20).sum().replace(0, np.nan)
    data['VWAP_20'] = cum_pv / cum_vol
    data['Dist_VWAP'] = ((close - data['VWAP_20']) / data['VWAP_20']).fillna(0.0)

    # 16. Candlestick Pattern Recognition (Normalized -1.0 to +1.0)
    body_size = (close - data['Open']).abs()
    full_range = (high - low).replace(0, np.nan)
    prev_open = data['Open'].shift(1)
    prev_close = close.shift(1)

    is_hammer = (data['Lower_Wick_Ratio'] >= 0.55) & (data['Upper_Wick_Ratio'] <= 0.15) & (body_size <= 0.35 * full_range)
    is_shooting_star = (data['Upper_Wick_Ratio'] >= 0.55) & (data['Lower_Wick_Ratio'] <= 0.15) & (body_size <= 0.35 * full_range)
    is_bullish_engulf = (prev_close < prev_open) & (close > data['Open']) & (close >= prev_open) & (data['Open'] <= prev_close)
    is_bearish_engulf = (prev_close > prev_open) & (close < data['Open']) & (close <= prev_open) & (data['Open'] >= prev_close)
    is_doji = (body_size <= 0.08 * full_range)

    candlestick_bias = np.zeros(len(data))
    candlestick_bias = np.where(is_hammer | is_bullish_engulf, 1.0, candlestick_bias)
    candlestick_bias = np.where(is_shooting_star | is_bearish_engulf, -1.0, candlestick_bias)
    data['Candlestick_Bias'] = candlestick_bias
    data['Is_Doji'] = np.where(is_doji, 1.0, 0.0)

    return data

def generate_composite_signals(df: pd.DataFrame) -> dict:
    """
    Evaluate latest row of technical indicators and produce a composite signal analysis.
    """
    if df.empty or len(df) < 50:
        return {
            "summary": "Insufficient Data",
            "score": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "details": []
        }

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    details = []
    bullish = 0
    bearish = 0
    neutral = 0

    # Indicator 1: RSI Level
    rsi = latest.get('RSI_14')
    if pd.notna(rsi):
        if rsi < 30:
            bullish += 1
            details.append({"indicator": "RSI (14)", "value": f"{rsi:.1f}", "signal": "Bullish", "reason": "Oversold condition (<30)"})
        elif rsi > 70:
            bearish += 1
            details.append({"indicator": "RSI (14)", "value": f"{rsi:.1f}", "signal": "Bearish", "reason": "Overbought condition (>70)"})
        else:
            neutral += 1
            details.append({"indicator": "RSI (14)", "value": f"{rsi:.1f}", "signal": "Neutral", "reason": "Normal momentum range (30-70)"})

    # Indicator 2: MACD Crossover / Histogram
    macd = latest.get('MACD')
    macd_sig = latest.get('MACD_Signal')
    macd_hist = latest.get('MACD_Hist')
    prev_hist = prev.get('MACD_Hist')
    if pd.notna(macd) and pd.notna(macd_sig):
        if macd > macd_sig and (prev_hist is not None and prev_hist <= 0):
            bullish += 1
            details.append({"indicator": "MACD Crossover", "value": f"{macd_hist:.2f}", "signal": "Bullish", "reason": "Fresh Bullish MACD Crossover"})
        elif macd < macd_sig and (prev_hist is not None and prev_hist >= 0):
            bearish += 1
            details.append({"indicator": "MACD Crossover", "value": f"{macd_hist:.2f}", "signal": "Bearish", "reason": "Fresh Bearish MACD Crossover"})
        elif macd > macd_sig:
            bullish += 1
            details.append({"indicator": "MACD Trend", "value": f"{macd_hist:.2f}", "signal": "Bullish", "reason": "MACD line above signal line"})
        else:
            bearish += 1
            details.append({"indicator": "MACD Trend", "value": f"{macd_hist:.2f}", "signal": "Bearish", "reason": "MACD line below signal line"})

    # Indicator 3: Price vs SMA 50 & SMA 200
    close = latest.get('Close')
    sma_50 = latest.get('SMA_50')
    sma_200 = latest.get('SMA_200')
    if pd.notna(close) and pd.notna(sma_50) and pd.notna(sma_200):
        if close > sma_50 and close > sma_200:
            bullish += 1
            details.append({"indicator": "Moving Averages", "value": f"₹{close:.2f}", "signal": "Bullish", "reason": "Trading above both 50 & 200 SMA"})
        elif close < sma_50 and close < sma_200:
            bearish += 1
            details.append({"indicator": "Moving Averages", "value": f"₹{close:.2f}", "signal": "Bearish", "reason": "Trading below both 50 & 200 SMA"})
        else:
            neutral += 1
            details.append({"indicator": "Moving Averages", "value": f"₹{close:.2f}", "signal": "Neutral", "reason": "Mixed trading vs 50/200 SMA"})

    # Indicator 4: Golden Cross / Death Cross
    if pd.notna(sma_50) and pd.notna(sma_200):
        if sma_50 > sma_200:
            bullish += 1
            details.append({"indicator": "SMA Crossover Trend", "value": f"50>200", "signal": "Bullish", "reason": "Golden Cross trend active"})
        else:
            bearish += 1
            details.append({"indicator": "SMA Crossover Trend", "value": f"50<200", "signal": "Bearish", "reason": "Death Cross trend active"})

    # Indicator 5: Bollinger Bands %B
    pct_b = latest.get('BB_PctB')
    if pd.notna(pct_b):
        if pct_b < 0.0:
            bullish += 1
            details.append({"indicator": "Bollinger Bands", "value": f"{pct_b:.2f}", "signal": "Bullish", "reason": "Price broke below Lower Band (potential bounce)"})
        elif pct_b > 1.0:
            bearish += 1
            details.append({"indicator": "Bollinger Bands", "value": f"{pct_b:.2f}", "signal": "Bearish", "reason": "Price broke above Upper Band (extended)"})
        else:
            neutral += 1
            details.append({"indicator": "Bollinger Bands", "value": f"{pct_b:.2f}", "signal": "Neutral", "reason": "Price within bands"})

    # Indicator 6: Volume Surge
    vol_surge = latest.get('Vol_Surge')
    if pd.notna(vol_surge):
        if vol_surge > 1.5:
            if close > prev.get('Close', close):
                bullish += 1
                details.append({"indicator": "Volume Analysis", "value": f"{vol_surge:.1f}x avg", "signal": "Bullish", "reason": "High volume buying surge"})
            else:
                bearish += 1
                details.append({"indicator": "Volume Analysis", "value": f"{vol_surge:.1f}x avg", "signal": "Bearish", "reason": "High volume selling pressure"})
        else:
            neutral += 1
            details.append({"indicator": "Volume Analysis", "value": f"{vol_surge:.1f}x avg", "signal": "Neutral", "reason": "Volume around normal 20-day average"})

    # Total recommendation
    total_signals = bullish + bearish + neutral
    score = (bullish - bearish) / total_signals if total_signals > 0 else 0

    if score >= 0.4:
        summary = "Strong Buy Signal"
    elif score >= 0.15:
        summary = "Mild Buy Signal"
    elif score <= -0.4:
        summary = "Strong Sell Signal"
    elif score <= -0.15:
        summary = "Mild Sell Signal"
    else:
        summary = "Neutral Signal"

    return {
        "summary": summary,
        "score": round(score, 2),
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "details": details
    }
