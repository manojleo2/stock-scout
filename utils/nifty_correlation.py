import pandas as pd
import numpy as np
import logging
from utils.data_loader import get_stock_data
from config import BENCHMARK_TICKER

def analyze_nifty_impact(symbol: str, period: str = "1y") -> dict:
    """
    Analyze correlation, beta, and downside impact of Nifty 50 on the target stock.
    """
    try:
        stock_df = get_stock_data(symbol, period=period)
        nifty_df = get_stock_data(BENCHMARK_TICKER, period=period)

        if stock_df.empty or nifty_df.empty:
            return {
                "status": "error",
                "message": "Data unavailable for stock or Nifty index."
            }

        # Align on date index
        combined = pd.DataFrame({
            "Stock_Close": stock_df['Close'],
            "Stock_Ret": stock_df['Close'].pct_change(),
            "Nifty_Close": nifty_df['Close'],
            "Nifty_Ret": nifty_df['Close'].pct_change()
        }).dropna()

        if len(combined) < 30:
            return {"status": "error", "message": "Insufficient overlapping trading days."}

        # 1. Overall Pearson Correlation
        corr = combined['Stock_Ret'].corr(combined['Nifty_Ret'])

        # 2. Rolling 30-day Correlation (latest value)
        rolling_corr = combined['Stock_Ret'].rolling(30).corr(combined['Nifty_Ret']).iloc[-1]

        # 3. Beta calculation (Covariance(Stock, Nifty) / Variance(Nifty))
        cov = combined['Stock_Ret'].cov(combined['Nifty_Ret'])
        var_nifty = combined['Nifty_Ret'].var()
        beta = cov / var_nifty if var_nifty != 0 else 1.0

        # 4. Downside reaction: Avg stock drop when Nifty falls > 1%
        nifty_down_days = combined[combined['Nifty_Ret'] < -0.01]
        avg_stock_drop = nifty_down_days['Stock_Ret'].mean() if not nifty_down_days.empty else 0.0

        # 5. Upside reaction: Avg stock gain when Nifty rises > 1%
        nifty_up_days = combined[combined['Nifty_Ret'] > 0.01]
        avg_stock_gain = nifty_up_days['Stock_Ret'].mean() if not nifty_up_days.empty else 0.0

        # 6. Current Nifty Momentum
        nifty_latest = nifty_df.iloc[-1]['Close']
        nifty_sma50 = nifty_df['Close'].rolling(50).mean().iloc[-1] if len(nifty_df) >= 50 else nifty_latest
        nifty_sma200 = nifty_df['Close'].rolling(200).mean().iloc[-1] if len(nifty_df) >= 200 else nifty_latest

        nifty_1d_change = nifty_df['Close'].pct_change().iloc[-1] * 100

        if nifty_latest > nifty_sma50 and nifty_sma50 > nifty_sma200:
            nifty_trend = "Bullish"
            nifty_badge = "🟢"
        elif nifty_latest < nifty_sma50 and nifty_sma50 < nifty_sma200:
            nifty_trend = "Bearish"
            nifty_badge = "🔴"
        else:
            nifty_trend = "Sideways / Mixed"
            nifty_badge = "🟡"

        # Impact level synthesis
        if abs(corr) > 0.6:
            impact_level = "High Correlation — Stock closely tracks Nifty movements."
        elif abs(corr) > 0.3:
            impact_level = "Moderate Correlation — Stock shows partial Nifty sensitivity."
        else:
            impact_level = "Low Correlation — Stock trades independently of broader Nifty sentiment."

        return {
            "status": "success",
            "correlation": round(corr, 2),
            "rolling_correlation_30d": round(rolling_corr, 2),
            "beta": round(beta, 2),
            "nifty_1d_change_pct": round(nifty_1d_change, 2),
            "nifty_trend": nifty_trend,
            "nifty_badge": nifty_badge,
            "avg_stock_drop_on_nifty_fall": round(avg_stock_drop * 100, 2),
            "avg_stock_gain_on_nifty_rise": round(avg_stock_gain * 100, 2),
            "impact_level": impact_level,
            "sample_days": len(combined)
        }
    except Exception as e:
        logging.error(f"Error in Nifty impact analysis: {e}")
        return {"status": "error", "message": str(e)}
