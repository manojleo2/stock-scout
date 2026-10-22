"""
utils/nifty_overnight_cues.py
──────────────────────────────
Live pre-open market summary for the Streamlit dashboard.
Fetches overnight global cues, India VIX, and SGX/GIFT Nifty gap proxy.
Standalone module — does NOT depend on the ML model.
Used to show a "Pre-Market Briefing" panel at 09:00–09:18 AM.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

import yfinance as yf
import pandas as pd
import numpy as np

from config import (
    SP500_TICKER, NASDAQ_TICKER, NIKKEI_TICKER, HANGSENG_TICKER,
    INDIA_VIX_TICKER, BANKNIFTY_TICKER, NIFTY_TICKER, NIFTYBEES_TICKER,
)

logger = logging.getLogger("nifty_overnight_cues")


def _safe_fast_info(sym: str) -> Dict[str, Any]:
    try:
        t = yf.Ticker(sym)
        fi = t.fast_info
        return {
            "last":  getattr(fi, "last_price", None),
            "prev":  getattr(fi, "previous_close", None),
            "open":  getattr(fi, "open", None),
        }
    except Exception:
        return {}


def _pct_change(now, prev) -> float:
    if now and prev and prev != 0:
        return round((now - prev) / prev * 100, 2)
    return 0.0


def _vix_badge(vix: float) -> tuple:
    if vix < 13:
        return "🟢", "Low Volatility — Bullish Climate"
    elif vix < 18:
        return "🟡", "Moderate Volatility"
    else:
        return "🔴", "High Volatility — Fear Spike"


def get_overnight_cues() -> Dict[str, Any]:
    """
    Fetch live pre-open cues in parallel.
    Returns a structured dict for the dashboard pre-market panel.
    Suitable to call at 09:00–09:18 AM before the trade signal is generated.
    """
    symbols = {
        "nifty":     NIFTY_TICKER,
        "sp500":     SP500_TICKER,
        "nasdaq":    NASDAQ_TICKER,
        "nikkei":    NIKKEI_TICKER,
        "hangseng":  HANGSENG_TICKER,
        "vix":       INDIA_VIX_TICKER,
        "banknifty": BANKNIFTY_TICKER,
        "niftybees": NIFTYBEES_TICKER,
    }

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {name: ex.submit(_safe_fast_info, sym)
                   for name, sym in symbols.items()}
    data = {name: fut.result() for name, fut in futures.items()}

    # ── Nifty ──────────────────────────────────────────────────────────────────
    nifty_last = data["nifty"].get("last") or data["nifty"].get("open")
    nifty_prev = data["nifty"].get("prev")
    nifty_chg  = _pct_change(nifty_last, nifty_prev)

    # ── SGX/GIFT Nifty Gap Proxy (NIFTYBEES open vs prev close) ──────────────
    # NIFTYBEES ≈ Nifty / 100. Uses open price to detect pre-market gap.
    try:
        nb_hist = yf.Ticker(NIFTYBEES_TICKER).history(period="2d", interval="1d", auto_adjust=True)
        if len(nb_hist) >= 2:
            nb_close_prev = float(nb_hist['Close'].iloc[-2])
            nb_open_today = float(nb_hist['Open'].iloc[-1])
            sgx_gap_pct   = round((nb_open_today - nb_close_prev) / nb_close_prev * 100, 3)
            sgx_nifty_est = round(nb_open_today * 100, 0)
        else:
            sgx_gap_pct   = 0.0
            sgx_nifty_est = nifty_last or 0.0
    except Exception:
        sgx_gap_pct   = 0.0
        sgx_nifty_est = nifty_last or 0.0

    # ── VIX ────────────────────────────────────────────────────────────────────
    vix_last = data["vix"].get("last")
    vix_prev = data["vix"].get("prev")
    vix_chg  = _pct_change(vix_last, vix_prev)
    vix_badge, vix_status = _vix_badge(float(vix_last) if vix_last else 15.0)

    # ── Overnight Global Markets ───────────────────────────────────────────────
    def _mkt(name, label):
        d = data[name]
        last = d.get("last")
        prev = d.get("prev")
        chg  = _pct_change(last, prev)
        emoji = "🟢" if chg > 0.2 else ("🔴" if chg < -0.2 else "⚪")
        return {
            "label":   label,
            "last":    round(last, 2) if last else "N/A",
            "change":  chg,
            "emoji":   emoji,
            "bias":    "Bullish" if chg > 0.3 else ("Bearish" if chg < -0.3 else "Neutral"),
        }

    global_mkts = [
        _mkt("sp500",    "S&P 500"),
        _mkt("nasdaq",   "Nasdaq"),
        _mkt("nikkei",   "Nikkei 225"),
        _mkt("hangseng", "Hang Seng"),
    ]

    # ── Composite Overnight Bias Score ────────────────────────────────────────
    bias_scores = [m["change"] for m in global_mkts]
    avg_bias    = float(np.mean(bias_scores)) if bias_scores else 0.0
    if avg_bias > 0.25:
        overnight_sentiment = "POSITIVE 🟢"
    elif avg_bias < -0.25:
        overnight_sentiment = "NEGATIVE 🔴"
    else:
        overnight_sentiment = "NEUTRAL ⚪"

    return {
        "nifty_last":           round(float(nifty_last), 2) if nifty_last else "N/A",
        "nifty_prev_close":     round(float(nifty_prev), 2) if nifty_prev else "N/A",
        "nifty_chg_pct":        nifty_chg,
        "sgx_gap_pct":          sgx_gap_pct,
        "sgx_nifty_est":        sgx_nifty_est,
        "vix":                  round(float(vix_last), 2) if vix_last else "N/A",
        "vix_chg_pct":          vix_chg,
        "vix_badge":            vix_badge,
        "vix_status":           vix_status,
        "banknifty_chg_pct":    _pct_change(data["banknifty"].get("last"), data["banknifty"].get("prev")),
        "global_markets":       global_mkts,
        "overnight_sentiment":  overnight_sentiment,
        "avg_global_bias_pct":  round(avg_bias, 3),
    }
