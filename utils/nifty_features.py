"""
utils/nifty_features.py
───────────────────────
45-Feature Matrix Builder for Nifty/Sensex Intraday Options Predictor.

Fix #1: Expiry calendar handles Sep 2025 Thursday→Tuesday regime shift.
Fix #2: ALL Group A (Technical) and Group D (FII/DII) features are SHIFTED BY 1
        (T-1 EOD) to prevent any lookahead bias in the 09:18 AM prediction.
Fix #3: Group B (OI/PCR) is intentionally EXCLUDED from this matrix —
        used only as a live post-model conviction modifier in nifty_oi_pcr.py.

Feature Groups in this matrix (45 total):
  Group A (20): Technical indicators — all T-1 shifted
  Group C (11): Macro & global cues (overnight, VIX, Asia)
  Group D ( 5): FII/DII institutional flow — T-1 shifted
  Group E (10): Calendar, regime, microstructure
"""

import os
import sys
import logging
import datetime as dt
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    NIFTY_TICKER, SENSEX_TICKER, NIFTYBEES_TICKER,
    BANKNIFTY_TICKER, INDIA_VIX_TICKER,
    SP500_TICKER, NASDAQ_TICKER, DOW_TICKER, NIKKEI_TICKER, HANGSENG_TICKER,
    NIFTY_EXPIRY_WEEKDAY, NIFTY_OLD_EXPIRY_WEEKDAY, NIFTY_EXPIRY_REGIME_SHIFT,
    SENSEX_EXPIRY_WEEKDAY, NIFTY_INTRADAY_MIN_MOVE_PCT,
)

logger = logging.getLogger("nifty_features")

# ─────────────────────────────────────────────────────────────────────────────
# Fix #1: Expiry Calendar with Sep 2025 Regime Shift
# ─────────────────────────────────────────────────────────────────────────────

EXPIRY_SHIFT_DATE = pd.Timestamp(NIFTY_EXPIRY_REGIME_SHIFT)


def _nifty_expiry_weekday(ref: pd.Timestamp) -> int:
    """Return Nifty expiry weekday: Tuesday (1) post Sep-2025, Thursday (3) before."""
    return NIFTY_EXPIRY_WEEKDAY if ref >= EXPIRY_SHIFT_DATE else NIFTY_OLD_EXPIRY_WEEKDAY


def _next_expiry(ref: pd.Timestamp, wd: int) -> pd.Timestamp:
    """Next (or same-day) occurrence of weekday `wd` from `ref`."""
    days = (wd - ref.weekday()) % 7
    return ref + timedelta(days=days if days > 0 else 7)


def get_days_to_nifty_expiry(ref: pd.Timestamp) -> int:
    """Days to next Nifty weekly expiry. Handles pre/post Sep-2025 regime."""
    wd = _nifty_expiry_weekday(ref)
    expiry = _next_expiry(ref, wd)
    return (expiry - ref).days


def get_days_to_sensex_expiry(ref: pd.Timestamp) -> int:
    """Days to next Sensex weekly expiry (Thursday)."""
    return (_next_expiry(ref, SENSEX_EXPIRY_WEEKDAY) - ref).days


def is_nifty_expiry_day(ref: pd.Timestamp) -> bool:
    return ref.weekday() == _nifty_expiry_weekday(ref)


def is_sensex_expiry_day(ref: pd.Timestamp) -> bool:
    return ref.weekday() == SENSEX_EXPIRY_WEEKDAY


# ─────────────────────────────────────────────────────────────────────────────
# Technical Indicator Calculations (internal, applied to EOD data)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Standard OHLCV technical indicators. Returns df with indicator columns."""
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    vol   = df['Volume']

    # Use min_periods to prevent wiping out historical warmup sessions
    df['SMA_20']  = close.rolling(20, min_periods=5).mean().bfill().ffill().fillna(close)
    df['SMA_50']  = close.rolling(50, min_periods=10).mean().bfill().ffill().fillna(close)
    df['SMA_200'] = close.rolling(200, min_periods=20).mean().bfill().ffill().fillna(close)
    df['EMA_12']  = close.ewm(span=12, adjust=False).mean()
    df['EMA_26']  = close.ewm(span=26, adjust=False).mean()

    delta    = close.diff()
    gain     = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss     = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    df['MACD']       = df['EMA_12'] - df['EMA_26']
    df['MACD_Sig']   = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']  = df['MACD'] - df['MACD_Sig']

    bb_mid        = close.rolling(20).mean()
    bb_std        = close.rolling(20).std()
    df['BB_Upper']= bb_mid + 2 * bb_std
    df['BB_Lower']= bb_mid - 2 * bb_std
    df['BB_Width']= (df['BB_Upper'] - df['BB_Lower']) / bb_mid
    df['BB_PctB'] = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['ATR_14'] = tr.ewm(span=14, adjust=False).mean()

    vol_sma         = vol.rolling(20).mean()
    df['Vol_Surge'] = vol / vol_sma.replace(0, np.nan)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FII/DII Historical Data (T-1 EOD, optional — zero fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_fii_dii() -> pd.DataFrame:
    """
    Try to fetch FII/DII data from NSE website.
    Returns DataFrame [FII_Net_Cash, FII_Net_FO, DII_Net_Cash] indexed by date.
    Returns empty DataFrame on any failure — model trains with zeros (no lookahead).
    """
    try:
        import requests
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        })
        s.get("https://www.nseindia.com", timeout=3)
        r = s.get("https://www.nseindia.com/api/fiidiiTradeReact", timeout=5)
        if r.status_code != 200:
            return pd.DataFrame()
        rows = []
        for entry in r.json():
            try:
                d = pd.to_datetime(entry.get("date", ""), dayfirst=True)
                def _parse(v):
                    return float(str(v).replace(",", "").replace("−","-") or 0)
                fii_c = _parse(entry.get("fiibuy", 0)) - _parse(entry.get("fiisell", 0))
                dii_c = _parse(entry.get("diibuy", 0)) - _parse(entry.get("diisell", 0))
                rows.append({"date": d, "FII_Net_Cash": fii_c,
                             "FII_Net_FO": 0.0, "DII_Net_Cash": dii_c})
            except Exception:
                continue
        if not rows:
            return pd.DataFrame()
        raw_df = pd.DataFrame(rows)
        out = raw_df.groupby("date").sum().sort_index()
        out.index = out.index.tz_localize(None)
        out = out[~out.index.duplicated(keep='last')]
        return out
    except Exception as e:
        logger.debug(f"FII/DII fetch failed (using zeros): {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Ticker Fetch Helper
# ─────────────────────────────────────────────────────────────────────────────

def _fetch(sym: str, period: str) -> tuple:
    try:
        df = yf.Ticker(sym).history(period=period, auto_adjust=True)
        if not df.empty and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if not df.empty:
            df = df[~df.index.duplicated(keep='last')]
        return sym, df
    except Exception as e:
        logger.warning(f"Failed fetching {sym}: {e}")
        return sym, pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Main Feature Matrix Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(ticker: str = NIFTY_TICKER, period: str = "5y") -> pd.DataFrame:
    """
    Build the 46-column ML feature matrix for `ticker` (Nifty or Sensex).

    Fix #2 & Issue 4 compliance:
      - All Group A features are SHIFTED BY 1 (using T-1 EOD data).
      - All Group D features are SHIFTED BY 1.
      - US markets (SP500, Nasdaq, Dow) use the most recent overnight close (date T-1).
      - Domestic macro cues (VIX, BankNifty) are strictly shifted by 1 (T-1 EOD).
      - Target uses CURRENT row's Open and Close (T intraday — only known after market).
    """
    sidecar = {
        "sp500":     SP500_TICKER,
        "nasdaq":    NASDAQ_TICKER,
        "dow":       DOW_TICKER,
        "vix":       INDIA_VIX_TICKER,
        "banknifty": BANKNIFTY_TICKER,
        "nikkei":    NIKKEI_TICKER,
        "hangseng":  HANGSENG_TICKER,
        "niftybees": NIFTYBEES_TICKER,
    }

    logger.info(f"Downloading {ticker} + {len(sidecar)} sidecars ({period})...")
    with ThreadPoolExecutor(max_workers=9) as ex:
        main_fut = ex.submit(_fetch, ticker, period)
        side_futs = {name: ex.submit(_fetch, sym, period)
                     for name, sym in sidecar.items()}
    _, main_df = main_fut.result()
    side = {name: fut.result()[1] for name, fut in side_futs.items()}

    if main_df.empty or len(main_df) < 130:
        raise ValueError(f"Insufficient data for {ticker}: {len(main_df)} rows (need 130+)")

    df = _compute_indicators(main_df.copy())
    day_range = (df['High'] - df['Low']).replace(0, np.nan)

    # ── Group A: Compute raw indicators, then shift(1) for T-1 EOD ──────────
    def _s1(series): return series.shift(1)  # T-1 EOD shift

    df['A_Ret_1']       = _s1(df['Close'].pct_change(1))
    df['A_Ret_3']       = _s1(df['Close'].pct_change(3))
    df['A_Ret_5']       = _s1(df['Close'].pct_change(5))
    df['A_Ret_10']      = _s1(df['Close'].pct_change(10))
    df['A_Ret_20']      = _s1(df['Close'].pct_change(20))
    df['A_Dist_SMA20']  = _s1((df['Close'] / df['SMA_20']) - 1.0)
    df['A_Dist_SMA50']  = _s1((df['Close'] / df['SMA_50']) - 1.0)
    df['A_Dist_SMA200'] = _s1((df['Close'] / df['SMA_200']) - 1.0)
    df['A_EMA_Cross']   = _s1((df['EMA_12'] / df['EMA_26']) - 1.0)
    df['A_RSI_Norm']    = _s1(df['RSI_14'] / 100.0)
    df['A_RSI_Slope']   = _s1(df['RSI_14'].diff(3) / 100.0)
    df['A_MACD_Norm']   = _s1(df['MACD_Hist'] / df['Close'])
    df['A_BB_PctB']     = _s1(df['BB_PctB'].fillna(0.5))
    df['A_BB_Width']    = _s1(df['BB_Width'].fillna(0.0))
    df['A_ATR_Norm']    = _s1((df['ATR_14'] / df['Close']).fillna(0.0))
    df['A_Vol_Surge']   = _s1(df['Vol_Surge'].fillna(1.0))
    df['A_Close_High']  = _s1(((df['Close'] - df['Low']) / day_range).fillna(0.5))
    df['A_Intra_Range'] = _s1((day_range / df['Close']).fillna(0.0))
    _body = df[['Open', 'Close']]
    df['A_Upper_Wick']  = _s1(((df['High'] - _body.max(axis=1)) / day_range).fillna(0.0))
    df['A_Lower_Wick']  = _s1(((      _body.min(axis=1) - df['Low']) / day_range).fillna(0.0))

    # ── Group C: Macro & Global Cues ─────────────────────────────────────────
    def _align_us_ret(side_df: pd.DataFrame) -> pd.Series:
        if side_df.empty or 'Close' not in side_df.columns:
            return pd.Series(0.0, index=df.index)
        clean = side_df['Close'].dropna()
        if clean.empty:
            return pd.Series(0.0, index=df.index)
        clean = clean[~clean.index.duplicated(keep='last')]
        s = pd.Series(clean.values, index=pd.to_datetime(clean.index.date))
        # Reindexing with ffill maps date T-1 US close to Indian date T (overnight return)
        aligned = s.reindex(pd.to_datetime(df.index.date), method='ffill').ffill().bfill()
        ret = aligned.pct_change(1, fill_method=None).fillna(0.0)
        ret.index = df.index
        return ret

    def _align_domestic_ret(side_df: pd.DataFrame) -> pd.Series:
        if side_df.empty or 'Close' not in side_df.columns:
            return pd.Series(0.0, index=df.index)
        clean = side_df['Close'].dropna()
        if clean.empty:
            return pd.Series(0.0, index=df.index)
        clean = clean[~clean.index.duplicated(keep='last')]
        s = pd.Series(clean.values, index=pd.to_datetime(clean.index.date))
        aligned = s.reindex(pd.to_datetime(df.index.date), method='ffill').ffill().bfill()
        # Domestic close is 15:30 IST — MUST shift(1) to be strictly T-1 EOD at 09:18 AM
        ret = aligned.pct_change(1, fill_method=None).shift(1).fillna(0.0)
        ret.index = df.index
        return ret

    def _align_domestic_val(side_df: pd.DataFrame, transform=None) -> pd.Series:
        if side_df.empty or 'Close' not in side_df.columns:
            return pd.Series(0.0, index=df.index)
        clean = side_df['Close'].dropna()
        if clean.empty:
            return pd.Series(0.0, index=df.index)
        clean = clean[~clean.index.duplicated(keep='last')]
        s = pd.Series(clean.values, index=pd.to_datetime(clean.index.date))
        aligned = s.reindex(pd.to_datetime(df.index.date), method='ffill').ffill().bfill()
        val = (transform(aligned) if transform else aligned).shift(1).fillna(0.0)
        val.index = df.index
        return val

    # Overnight US Cues: Date T-1 close finished at 02:30 AM on Indian Date T
    df['C_SP500_Ret1']     = _align_us_ret(side['sp500'])
    df['C_Nasdaq_Ret1']    = _align_us_ret(side['nasdaq'])
    df['C_DowJones_Ret1']  = _align_us_ret(side['dow'])

    # Domestic Cues: Shifted by 1 (Strict T-1 EOD)
    df['C_VIX_Norm']       = _align_domestic_val(side['vix'], lambda s: s / 100.0)
    df['C_VIX_Ret1']       = _align_domestic_ret(side['vix'])

    vix_shifted = _align_domestic_val(side['vix']).replace(0, np.nan).fillna(15.0)
    df['C_VIX_Regime']     = pd.cut(
        vix_shifted,
        bins=[0, 13, 18, 200], labels=[0, 1, 2]
    ).astype(float).fillna(1.0)

    # 5-day correlation between Nifty and BankNifty using strictly T-1 EOD data
    df['C_BankNifty_Ret1'] = _align_domestic_ret(side['banknifty'])
    df['C_BankNifty_Corr'] = (
        df['Close'].pct_change(1, fill_method=None).shift(1)
        .rolling(5, min_periods=2)
        .corr(_align_domestic_ret(side['banknifty']))
    ).fillna(0.0)

    # Asian Cues: Shifted by 1 (Strict T-1 EOD)
    df['C_Nikkei_Ret1']    = _align_domestic_ret(side['nikkei'])
    df['C_HangSeng_Ret1']  = _align_domestic_ret(side['hangseng'])
    df['C_Asia_Sentiment'] = (df['C_Nikkei_Ret1'] + df['C_HangSeng_Ret1']) / 2.0

    # SGX Gap Proxy: NIFTYBEES today's open vs yesterday's close (available at 09:18 AM)
    nb = side['niftybees']
    if not nb.empty and 'Open' in nb.columns:
        nb_clean = nb[~nb.index.duplicated(keep='last')]
        nb_open  = nb_clean['Open'].reindex(df.index, method='ffill').ffill().bfill()
        nb_pclose= nb_clean['Close'].reindex(df.index, method='ffill').shift(1).ffill().bfill()
        df['C_SGX_Gap_Pct'] = ((nb_open - nb_pclose) / nb_pclose.replace(0, np.nan)).fillna(0.0)
    else:
        df['C_SGX_Gap_Pct'] = 0.0

    # ── Group D: FII/DII — T-1 EOD shift with cross-calendar ffill ───────────
    fii_df = _fetch_fii_dii()
    if not fii_df.empty:
        fii_df = fii_df[~fii_df.index.duplicated(keep='last')]
    for col in ['FII_Net_Cash', 'FII_Net_FO', 'DII_Net_Cash']:
        if not fii_df.empty and col in fii_df.columns:
            raw = fii_df[col].reindex(df.index, method='ffill').ffill().bfill().fillna(0.0) / 1000.0
        else:
            raw = pd.Series(0.0, index=df.index)
        df[f'D_{col}'] = raw.shift(1).ffill().fillna(0.0)  # T-1 shift

    if not fii_df.empty and {'FII_Net_Cash', 'DII_Net_Cash'}.issubset(fii_df.columns):
        fii_r = fii_df['FII_Net_Cash'].reindex(df.index, method='ffill').fillna(0.0)
        dii_r = fii_df['DII_Net_Cash'].reindex(df.index, method='ffill').fillna(0.0)
        ratio = (fii_r / dii_r.replace(0, np.nan)).clip(-5, 5).fillna(0.0)
        df['D_FII_DII_Ratio'] = ratio.shift(1).fillna(0.0)
        # FII consecutive buy/sell streak
        dirs = np.sign(fii_r.values)
        streak = np.zeros(len(dirs))
        for i in range(1, len(dirs)):
            if dirs[i] == dirs[i-1] and dirs[i] != 0:
                streak[i] = streak[i-1] + dirs[i]
            else:
                streak[i] = dirs[i]
        df['D_FII_Streak'] = pd.Series(streak, index=df.index).shift(1).fillna(0.0)
    else:
        df['D_FII_DII_Ratio'] = 0.0
        df['D_FII_Streak']    = 0.0

    # ── Group E: Calendar & Regime (no price lookahead) ──────────────────────
    timestamps = [pd.Timestamp(d) for d in df.index]
    df['E_Days_Nifty_Expiry']  = [get_days_to_nifty_expiry(t) for t in timestamps]
    df['E_Days_Sensex_Expiry'] = [get_days_to_sensex_expiry(t) for t in timestamps]
    df['E_Is_Expiry_Day']      = [1.0 if is_nifty_expiry_day(t) else 0.0 for t in timestamps]
    df['E_Is_Monday']          = (df.index.dayofweek == 0).astype(float)
    df['E_Is_Friday']          = (df.index.dayofweek == 4).astype(float)

    # Regime proxy: 20-day directional slope (T-1 shifted)
    df['E_Regime_Trend']      = df['Close'].pct_change(20, fill_method=None).shift(1).fillna(0.0)
    df['E_Regime_Volatility'] = df['Close'].pct_change(5, fill_method=None).rolling(5).std().shift(1).fillna(0.0)

    # Consecutive gap direction streak prior to today (shift by 1 so only past streak is used)
    prev_gap_dir = np.where(
        df['Open'] > df['Close'].shift(1), 1,
        np.where(df['Open'] < df['Close'].shift(1), -1, 0)
    )
    streak_g = np.zeros(len(df))
    for i in range(1, len(df)):
        if prev_gap_dir[i] == prev_gap_dir[i-1] and prev_gap_dir[i] != 0:
            streak_g[i] = streak_g[i-1] + prev_gap_dir[i]
        else:
            streak_g[i] = prev_gap_dir[i]
    df['E_Consec_Gap_Streak'] = pd.Series(streak_g, index=df.index).shift(1).fillna(0.0)

    # Opening drive strength: today's gap / yesterday's ATR — valid at 09:18 AM
    atr_prev = df['ATR_14'].shift(1).replace(0, np.nan)
    df['E_Open_Continuation'] = ((df['Open'] - df['Close'].shift(1)) / atr_prev).fillna(0.0)

    # Vol-Price trend: yesterday's surge × direction (already T-1 via A_Vol_Surge, A_Ret_1)
    df['E_Vol_Price_Trend'] = (df['A_Vol_Surge'] * df['A_Ret_1']).fillna(0.0)

    # ── Target: Intraday Direction (T row — unknown at 09:18 AM, only used for training) ──
    oc_pct = (df['Close'] - df['Open']) / df['Open'].replace(0, np.nan) * 100.0
    min_mv = NIFTY_INTRADAY_MIN_MOVE_PCT
    df['Target_Intraday'] = np.where(
        oc_pct >=  min_mv, 1,
        np.where(oc_pct <= -min_mv, 0, np.nan)
    )

    return df


# ── Feature Column Lists (exported for use in model + backtest) ───────────────

GROUP_A = [
    'A_Ret_1', 'A_Ret_3', 'A_Ret_5', 'A_Ret_10', 'A_Ret_20',
    'A_Dist_SMA20', 'A_Dist_SMA50', 'A_Dist_SMA200',
    'A_EMA_Cross', 'A_RSI_Norm', 'A_RSI_Slope',
    'A_MACD_Norm', 'A_BB_PctB', 'A_BB_Width',
    'A_ATR_Norm', 'A_Vol_Surge', 'A_Close_High',
    'A_Intra_Range', 'A_Upper_Wick', 'A_Lower_Wick',
]
GROUP_C = [
    'C_SP500_Ret1', 'C_Nasdaq_Ret1', 'C_DowJones_Ret1',
    'C_VIX_Norm', 'C_VIX_Ret1', 'C_VIX_Regime',
    'C_BankNifty_Ret1', 'C_BankNifty_Corr',
    'C_Nikkei_Ret1', 'C_HangSeng_Ret1',
    'C_Asia_Sentiment', 'C_SGX_Gap_Pct',
]
GROUP_D = [
    'D_FII_Net_Cash', 'D_FII_Net_FO', 'D_DII_Net_Cash',
    'D_FII_DII_Ratio', 'D_FII_Streak',
]
GROUP_E = [
    'E_Days_Nifty_Expiry', 'E_Days_Sensex_Expiry',
    'E_Is_Expiry_Day', 'E_Is_Monday', 'E_Is_Friday',
    'E_Regime_Trend', 'E_Regime_Volatility',
    'E_Consec_Gap_Streak', 'E_Open_Continuation', 'E_Vol_Price_Trend',
]

ALL_ML_FEATURES = GROUP_A + GROUP_C + GROUP_D + GROUP_E  # 46 total (no Group B)
