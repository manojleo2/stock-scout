"""
utils/nifty_regime.py
──────────────────────
Market Regime Classifier: Trending / Chop / Volatile.
Used by nifty_sensex_model.py to gate conviction thresholds.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def classify_regime(df: pd.DataFrame, lookback: int = 20) -> Tuple[str, float, str]:
    """
    Classify current market regime using last `lookback` rows.
    
    Algorithm:
      1. ADX-proxy: average directional movement using ATR-normalized range
      2. BB Width percentile: wide = volatile, narrow = chop
      3. Trend persistence: % of days in same direction over lookback

    Returns:
      regime: "TRENDING" | "CHOP" | "VOLATILE"
      score:  -1.0 (strong down) to +1.0 (strong up), 0 = chop/neutral
      detail: human-readable description
    """
    if df.empty or len(df) < lookback:
        return "CHOP", 0.0, "Insufficient data for regime classification"

    recent = df.tail(lookback).copy()
    close  = recent['Close']
    high   = recent['High']
    low    = recent['Low']

    # ── ADX Proxy (Wilder directional movement) ───────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False).mean().iloc[-1]

    dm_up   = (high.diff().clip(lower=0)).ewm(span=14, adjust=False).mean().iloc[-1]
    dm_down = (-low.diff().clip(upper=0)).ewm(span=14, adjust=False).mean().iloc[-1]
    di_plus  = (dm_up   / atr_14 * 100) if atr_14 > 0 else 0.0
    di_minus = (dm_down / atr_14 * 100) if atr_14 > 0 else 0.0
    dx = (abs(di_plus - di_minus) / (di_plus + di_minus) * 100) if (di_plus + di_minus) > 0 else 0.0
    adx_proxy = float(dx)

    # ── BB Width Percentile ───────────────────────────────────────────────────
    bb_width_col = recent.get('BB_Width', pd.Series(dtype=float))
    if bb_width_col is None or bb_width_col.empty:
        bb_mid  = close.rolling(20).mean()
        bb_std  = close.rolling(20).std()
        bb_width_col = ((bb_mid + 2*bb_std - (bb_mid - 2*bb_std)) / bb_mid).fillna(0.0)
    bb_now  = float(bb_width_col.iloc[-1]) if not bb_width_col.empty else 0.0
    # Historical 90th percentile of BB width over lookback
    bb_90   = float(np.percentile(bb_width_col.fillna(0.0), 90))

    # ── Trend Persistence ─────────────────────────────────────────────────────
    daily_dirs = np.sign(close.pct_change().fillna(0).values)
    up_pct   = float((daily_dirs > 0).mean())
    down_pct = float((daily_dirs < 0).mean())
    trend_persistence = abs(up_pct - down_pct)   # 0 = 50/50 chop, 1 = all one direction

    # ── Direction Bias ────────────────────────────────────────────────────────
    ret_20 = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
    score  = float(np.clip(ret_20 * 10, -1.0, 1.0))  # Scaled to -1..+1

    # ── Regime Classification ─────────────────────────────────────────────────
    # VOLATILE: BB width near 90th pct OR VIX-like ATR expansion
    atr_norm = atr_14 / float(close.iloc[-1]) if close.iloc[-1] > 0 else 0.0
    is_volatile  = (bb_now >= bb_90 * 0.85) or (atr_norm > 0.015)

    # CHOP: low ADX + low trend persistence
    is_chop = (adx_proxy < 20.0) and (trend_persistence < 0.20)

    # TRENDING: decent ADX + trend persistence
    is_trending = (adx_proxy >= 20.0) and (trend_persistence >= 0.25)

    if is_volatile and not is_trending:
        regime = "VOLATILE"
        detail = (f"Volatile — BB_Width={bb_now:.3f} (90th={bb_90:.3f}) | "
                  f"ATR_Norm={atr_norm:.4f} | ADX_proxy={adx_proxy:.1f}")
    elif is_chop:
        regime = "CHOP"
        detail = (f"Chop — ADX_proxy={adx_proxy:.1f} (<20) | "
                  f"Persistence={trend_persistence:.2f} (<20%)")
    elif is_trending:
        dir_str = "Bullish" if score > 0 else "Bearish"
        regime = "TRENDING"
        detail = (f"Trending {dir_str} — ADX_proxy={adx_proxy:.1f} | "
                  f"Persistence={trend_persistence:.2f} | 20d_Ret={ret_20*100:.2f}%")
    else:
        # Mixed — classify as mild chop
        regime = "CHOP"
        detail = (f"Mixed/Neutral — ADX_proxy={adx_proxy:.1f} | "
                  f"Persistence={trend_persistence:.2f}")

    return regime, round(score, 4), detail


def get_conviction_threshold(
    regime: str,
    is_friday: bool = False,
    is_expiry_day: bool = False,
    weekday_threshold: float = 62.0,
    friday_threshold: float = 68.0,
    expiry_threshold: float = 75.0,
    chop_threshold: float = 70.0,
    volatile_threshold: float = 72.0,
) -> float:
    """
    Return the appropriate conviction threshold for the given regime and day.
    Highest applicable threshold wins (most conservative).
    """
    thresholds = [weekday_threshold]
    if is_friday:
        thresholds.append(friday_threshold)
    if is_expiry_day:
        thresholds.append(expiry_threshold)
    if regime == "CHOP":
        thresholds.append(chop_threshold)
    elif regime == "VOLATILE":
        thresholds.append(volatile_threshold)
    return max(thresholds)
