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
