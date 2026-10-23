"""
utils/nifty_oi_pcr.py
──────────────────────
Fix #3 & Live Data Feed:
Option Chain scraper and parser for Nifty options with dual-source architecture:
  1. Priority 1: Groww Public Option Chain Service (Live, high availability, no cookie blockage)
  2. Priority 2: Direct NSE Option Chain endpoint
  3. Live India VIX fetcher via NSE allIndices + yfinance fallback

This module is EXCLUSIVELY for live inference at 09:18 AM.
It is NOT used in historical ML training (prevents training-inference mismatch).
"""

import os
import sys
import logging
import time
import requests
from typing import Dict, Any, Optional
import yfinance as yf

logger = logging.getLogger("nifty_oi_pcr")

_NSE_SESSION: Optional[requests.Session] = None
_SESSION_INIT_TIME: float = 0.0
_SESSION_TTL: float = 180.0   # Refresh NSE session every 3 minutes


def _get_nse_session() -> requests.Session:
    """Warm reusable NSE session with browser headers. Refreshes every 3 min."""
    global _NSE_SESSION, _SESSION_INIT_TIME
    now = time.time()
    if _NSE_SESSION is not None and (now - _SESSION_INIT_TIME) < _SESSION_TTL:
        return _NSE_SESSION
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nseindia.com/option-chain",
    })
    try:
        s.get("https://www.nseindia.com", timeout=3)
        s.get("https://www.nseindia.com/option-chain", timeout=3)
    except Exception:
        pass
    _NSE_SESSION = s
    _SESSION_INIT_TIME = now
    return s


def get_live_india_vix() -> Dict[str, Any]:
    """
    Fetch live India VIX directly from NSE allIndices API with yfinance fallback.
    Returns: {level, change_pct, status, source}
    """
    # Source 1: NSE Live All Indices
    try:
        s = _get_nse_session()
        r = s.get("https://www.nseindia.com/api/allIndices", timeout=4)
        if r.status_code == 200:
            for item in r.json().get("data", []):
                if item.get("index") == "INDIA VIX":
                    level = float(item.get("last", 0.0))
                    chg   = float(item.get("percentChange", 0.0))
                    if level > 0:
                        return {
                            "level": round(level, 2),
                            "change_pct": round(chg, 2),
                            "status": "LIVE",
                            "source": "NSE_ALL_INDICES",
                        }
    except Exception as e:
        logger.debug(f"NSE allIndices VIX fetch error: {e}")

    # Source 2: Yahoo Finance fast_info
    try:
        t = yf.Ticker("^INDIAVIX")
        fi = t.fast_info
        last = getattr(fi, "last_price", None)
        prev = getattr(fi, "previous_close", None)
        if last is not None and float(last) > 0:
            level = float(last)
            chg = round((level - float(prev)) / float(prev) * 100, 2) if prev else 0.0
            return {
                "level": round(level, 2),
                "change_pct": chg,
                "status": "LIVE",
                "source": "YFINANCE_FAST_INFO",
            }
    except Exception as e:
        logger.debug(f"YF VIX fetch error: {e}")

    return {"level": 15.0, "change_pct": 0.0, "status": "DEFAULT", "source": "FALLBACK"}


def fetch_from_groww_option_chain(symbol: str = "nifty") -> Optional[Dict[str, Any]]:
    """
    Fetch complete live option chain from Groww's option chain API (zero delay).
    Supports Nifty ('nifty') and Sensex ('sp-bse-sensex').
    """
    clean_sym = symbol.lower().replace("^", "").replace(".ns", "").replace(".bo", "").strip()
    if clean_sym in ["bsesn", "sensex", "sp-bse-sensex"]:
        slug = "sp-bse-sensex"
    elif clean_sym in ["nsei", "nifty", "nifty50"]:
        slug = "nifty"
    elif clean_sym in ["banknifty", "nsebank"]:
        slug = "banknifty"
    else:
        slug = clean_sym

    url = f"https://groww.in/v1/api/option_chain_service/v1/option_chain/{slug}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
    }
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.debug(f"Groww option chain fetch error for {slug}: {e}")
    return None


def fetch_from_nse_option_chain(symbol: str = "NIFTY") -> Optional[Dict[str, Any]]:
    """Fetch live option chain from NSE API."""
    try:
        s = _get_nse_session()
        # Try both endpoints in case of API shifts
        for endpoint in [
            f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}",
            f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}",
        ]:
            r = s.get(endpoint, timeout=4)
            if r.status_code == 200 and r.text.strip() not in ["{}", ""]:
                return r.json()
    except Exception as e:
        logger.debug(f"NSE option chain fetch error: {e}")
    return None


def _parse_groww_chain(data: dict, spot: float, strike_step: int = 50) -> Dict[str, Any]:
    """Parse Groww option chain format."""
    opts = data.get("optionChain", {}).get("optionChains", [])
    if not opts:
        return {}

    atm_strike = round(spot / strike_step) * strike_step

    total_ce_oi = total_pe_oi = 0
    total_ce_vol = total_pe_vol = 0
    atm_ce_oi = atm_pe_oi = 0
    atm_ce_ltp = atm_pe_ltp = 0.0
    atm_ce_iv = atm_pe_iv = 0.0
    atm_ce_bid = atm_pe_bid = 0.0
    atm_ce_ask = atm_pe_ask = 0.0
    pain_dict: Dict[int, float] = {}

    for o in opts:
        strike_raw = o.get("strikePrice", 0)
        # Groww strike prices are scaled by 100
        strike = int(strike_raw / 100.0) if strike_raw > 100000 else int(strike_raw)
        ce = o.get("callOption") or {}
        pe = o.get("putOption") or {}

        ce_oi  = float(ce.get("openInterest", 0) or 0)
        pe_oi  = float(pe.get("openInterest", 0) or 0)
        ce_vol = float(ce.get("volume", 0) or 0)
        pe_vol = float(pe.get("volume", 0) or 0)

        total_ce_oi  += ce_oi
        total_pe_oi  += pe_oi
        total_ce_vol += ce_vol
        total_pe_vol += pe_vol

        if strike == atm_strike:
            atm_ce_oi  = ce_oi
            atm_pe_oi  = pe_oi
            atm_ce_ltp = float(ce.get("ltp", 0.0) or 0.0)
            atm_pe_ltp = float(pe.get("ltp", 0.0) or 0.0)
            atm_ce_iv  = float(ce.get("impliedVolatility", 0) or 0) / 100.0
            atm_pe_iv  = float(pe.get("impliedVolatility", 0) or 0) / 100.0
            atm_ce_bid = float(ce.get("bestBidPrice") or ce.get("bidPrice") or ce.get("bid") or 0.0)
            atm_ce_ask = float(ce.get("bestAskPrice") or ce.get("askPrice") or ce.get("ask") or 0.0)
            atm_pe_bid = float(pe.get("bestBidPrice") or pe.get("bidPrice") or pe.get("bid") or 0.0)
            atm_pe_ask = float(pe.get("bestAskPrice") or pe.get("askPrice") or pe.get("ask") or 0.0)
            raw_ts = float(ce.get("lastTradeTime") or pe.get("lastTradeTime") or 0.0)
            atm_quote_ts = (raw_ts / 1000.0 if raw_ts > 1e11 else raw_ts) if raw_ts > 0 else None

        pain_dict[strike] = pain_dict.get(strike, 0.0) + ce_oi + pe_oi

    pcr_oi  = (total_pe_oi / total_ce_oi) if total_ce_oi > 0 else 1.0
    pcr_vol = (total_pe_vol / total_ce_vol) if total_ce_vol > 0 else 1.0
    max_pain_strike = max(pain_dict, key=pain_dict.get) if pain_dict else atm_strike
    iv_skew = atm_pe_iv - atm_ce_iv

    return {
        "atm_strike":      atm_strike,
        "spot":            spot,
        "pcr_oi":          round(pcr_oi, 3),
        "pcr_vol":         round(pcr_vol, 3),
        "max_pain_strike": max_pain_strike,
        "max_pain_dist":   round((spot - max_pain_strike) / spot * 100, 3) if spot else 0.0,
        "atm_ce_oi":       int(atm_ce_oi),
        "atm_pe_oi":       int(atm_pe_oi),
        "atm_ce_ltp":      round(atm_ce_ltp, 2),
        "atm_pe_ltp":      round(atm_pe_ltp, 2),
        "atm_ce_bid":      round(atm_ce_bid, 2),
        "atm_ce_ask":      round(atm_ce_ask, 2),
        "atm_pe_bid":      round(atm_pe_bid, 2),
        "atm_pe_ask":      round(atm_pe_ask, 2),
        "atm_ce_iv":       round(atm_ce_iv, 4),
        "atm_pe_iv":       round(atm_pe_iv, 4),
        "iv_skew":         round(iv_skew, 4),
        "total_ce_oi":     int(total_ce_oi),
        "total_pe_oi":     int(total_pe_oi),
        "oi_spread_ratio": round(total_pe_oi / total_ce_oi, 3) if total_ce_oi > 0 else 1.0,
        "quote_timestamp": atm_quote_ts,
        "source":          "GROWW_OPTION_CHAIN (Live Feed)",
    }


def _parse_nse_chain(data: dict, spot: float, strike_step: int = 50) -> Dict[str, Any]:
    """Parse standard NSE option chain format."""
    records = data.get("records", {}).get("data", [])
    if not records:
        return {}

    atm_strike = round(spot / strike_step) * strike_step
    total_ce_oi = total_pe_oi = 0
    total_ce_vol = total_pe_vol = 0
    atm_ce_oi = atm_pe_oi = 0
    atm_ce_ltp = atm_pe_ltp = 0.0
    atm_ce_iv = atm_pe_iv = 0.0
    atm_ce_bid = atm_pe_bid = 0.0
    atm_ce_ask = atm_pe_ask = 0.0
    pain_dict: Dict[int, float] = {}

    for rec in records:
        strike = rec.get("strikePrice", 0)
        ce = rec.get("CE", {})
        pe = rec.get("PE", {})

        ce_oi  = float(ce.get("openInterest", 0) or 0)
        pe_oi  = float(pe.get("openInterest", 0) or 0)
        ce_vol = float(ce.get("totalTradedVolume", 0) or 0)
        pe_vol = float(pe.get("totalTradedVolume", 0) or 0)

        total_ce_oi  += ce_oi
        total_pe_oi  += pe_oi
        total_ce_vol += ce_vol
        total_pe_vol += pe_vol

        if strike == atm_strike:
            atm_ce_oi  = ce_oi
            atm_pe_oi  = pe_oi
            atm_ce_ltp = float(ce.get("lastPrice", 0.0) or 0.0)
            atm_pe_ltp = float(pe.get("lastPrice", 0.0) or 0.0)
            atm_ce_iv  = float(ce.get("impliedVolatility", 0) or 0) / 100.0
            atm_pe_iv  = float(pe.get("impliedVolatility", 0) or 0) / 100.0
            atm_ce_bid = float(ce.get("bidprice", 0.0) or 0.0)
            atm_ce_ask = float(ce.get("askPrice", 0.0) or 0.0)
            atm_pe_bid = float(pe.get("bidprice", 0.0) or 0.0)
            atm_pe_ask = float(pe.get("askPrice", 0.0) or 0.0)

        pain_dict[strike] = pain_dict.get(strike, 0.0) + ce_oi + pe_oi

    pcr_oi  = (total_pe_oi / total_ce_oi) if total_ce_oi > 0 else 1.0
    pcr_vol = (total_pe_vol / total_ce_vol) if total_ce_vol > 0 else 1.0
    max_pain_strike = max(pain_dict, key=pain_dict.get) if pain_dict else atm_strike
    iv_skew = atm_pe_iv - atm_ce_iv

    return {
        "atm_strike":      atm_strike,
        "spot":            spot,
        "pcr_oi":          round(pcr_oi, 3),
        "pcr_vol":         round(pcr_vol, 3),
        "max_pain_strike": max_pain_strike,
        "max_pain_dist":   round((spot - max_pain_strike) / spot * 100, 3) if spot else 0.0,
        "atm_ce_oi":       int(atm_ce_oi),
        "atm_pe_oi":       int(atm_pe_oi),
        "atm_ce_ltp":      round(atm_ce_ltp, 2),
        "atm_pe_ltp":      round(atm_pe_ltp, 2),
        "atm_ce_bid":      round(atm_ce_bid, 2),
        "atm_ce_ask":      round(atm_ce_ask, 2),
        "atm_pe_bid":      round(atm_pe_bid, 2),
        "atm_pe_ask":      round(atm_pe_ask, 2),
        "atm_ce_iv":       round(atm_ce_iv, 4),
        "atm_pe_iv":       round(atm_pe_iv, 4),
        "iv_skew":         round(iv_skew, 4),
        "total_ce_oi":     int(total_ce_oi),
        "total_pe_oi":     int(total_pe_oi),
        "oi_spread_ratio": round(total_pe_oi / total_ce_oi, 3) if total_ce_oi > 0 else 1.0,
        "quote_timestamp": None,
        "source":          "NSE_LIVE_API",
    }


def get_live_nifty_oi_summary(spot: float, strike_step: int = 50, ticker: str = "^NSEI") -> Dict[str, Any]:
    """
    Master function to get live OI/PCR summary & ATM options:
      1. Tries Groww public API option chain first (zero delay, direct broker feed).
         - Nifty: 'nifty'
         - Sensex: 'sp-bse-sensex'
      2. Tries NSE direct API as secondary.
      3. Injects live India VIX.
    """
    # 1. Try Groww feed
    groww_data = fetch_from_groww_option_chain(ticker)
    if groww_data:
        parsed = _parse_groww_chain(groww_data, spot, strike_step)
        if parsed and (parsed.get("pcr_oi", 0) > 0 or parsed.get("atm_ce_ltp", 0) > 0 or parsed.get("atm_pe_ltp", 0) > 0):
            vix_info = get_live_india_vix()
            parsed["vix"] = vix_info["level"]
            return parsed

    # 2. Try NSE feed
    nse_sym = "NIFTY" if ticker in ["^NSEI", "nifty", "NIFTY"] else "SENSEX"
    nse_data = fetch_from_nse_option_chain(nse_sym)
    if nse_data:
        parsed = _parse_nse_chain(nse_data, spot, strike_step)
        if parsed and parsed.get("pcr_oi", 0) > 0:
            vix_info = get_live_india_vix()
            parsed["vix"] = vix_info["level"]
            return parsed

    logger.warning(f"All option chain feeds unavailable for {ticker} — returning neutral state.")
    vix_info = get_live_india_vix()
    return {
        "atm_strike":      round(spot / strike_step) * strike_step,
        "spot":            spot,
        "pcr_oi":          1.0,
        "pcr_vol":         1.0,
        "max_pain_strike": round(spot / strike_step) * strike_step,
        "max_pain_dist":   0.0,
        "atm_ce_oi":       0,
        "atm_pe_oi":       0,
        "atm_ce_ltp":      0.0,
        "atm_pe_ltp":      0.0,
        "atm_ce_iv":       0.15,
        "atm_pe_iv":       0.15,
        "iv_skew":         0.0,
        "total_ce_oi":     0,
        "total_pe_oi":     0,
        "oi_spread_ratio": 1.0,
        "source":          "UNAVAILABLE_DEFAULT",
        "vix":             vix_info["level"],
    }


def get_groww_account_status() -> Dict[str, Any]:
    """
    Connect to Groww API using user's credentials to get:
    - Available Option Buying Cash
    - UCC Account Number
    - Rule-based lot sizing (1 lot <= 30k, 2 lots > 30k, stop < 20k)
    """
    env_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        r"C:\Users\manoj\scout\stock_scout\.env",
    ]
    for ep in env_paths:
        if os.path.exists(ep):
            try:
                with open(ep, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip() not in os.environ:
                                os.environ[k.strip()] = v.strip().strip('"').strip("'")
            except Exception:
                pass

    token = os.environ.get("GROWW_AUTH_TOKEN", "")
    if not token:
        return {"connected": False, "balance": 24300.0, "ucc": "N/A", "lots": 1, "alert": None}

    try:
        from growwapi import GrowwAPI
        client = GrowwAPI(token=token)
        profile = client.get_user_profile()
        margin = client.get_available_margin_details()
        fno = margin.get("fno_margin_details", {})
        avail = fno.get("option_buy_balance_available") or margin.get("clear_cash", 0.0)
        balance = round(float(avail), 2)
        ucc = profile.get("ucc", "2390011137")

        if balance > 30000.0:
            lots = 2
            alert = None
        elif balance >= 20000.0:
            lots = 1
            alert = None
        else:
            lots = 0
            alert = "🚨 CAPITAL < ₹20,000 — STOP TRADING IMMEDIATELY!"

        return {
            "connected": True,
            "balance": balance,
            "ucc": ucc,
            "lots": lots,
            "alert": alert,
            "source": "GROWW_TRADE_API (Direct Zero-Delay Pipe)",
        }
    except Exception as e:
        logger.debug(f"Groww account status fetch error: {e}")
        return {"connected": False, "balance": 24300.0, "ucc": "N/A", "lots": 1, "alert": None}


def compute_pcr_conviction_modifier(
    oc_parsed: Optional[dict] = None,
    model_direction: str = "UP",   # "UP" or "DOWN"
    pcr_bearish_threshold: float = 1.30,
    pcr_bullish_threshold: float = 0.75,
    boost_per_signal: float = 3.0,
    max_total_adjustment: float = 6.0,
    oi_parsed: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Fix #3: Post-model OI/PCR conviction modifier.
    Accepts either `oc_parsed` or `oi_parsed`.
    Returns modifier dict: {adjustment_pct, signals, description}
    """
    if oc_parsed is None and oi_parsed is not None:
        oc_parsed = oi_parsed
    if not oc_parsed:
        return {"adjustment_pct": 0.0, "signals": [], "description": "OI data unavailable"}

    pcr_oi   = oc_parsed.get("pcr_oi", 1.0)
    iv_skew  = oc_parsed.get("iv_skew", 0.0)
    mp_dist  = oc_parsed.get("max_pain_dist", 0.0)

    signals = []
    raw_adj  = 0.0

    # Signal 1: PCR directional alignment
    if pcr_oi > pcr_bearish_threshold:
        if model_direction == "DOWN":
            raw_adj += boost_per_signal
            signals.append(f"PCR_OI={pcr_oi:.2f} confirms DOWN (+{boost_per_signal:.0f}%)")
        else:
            raw_adj -= boost_per_signal
            signals.append(f"PCR_OI={pcr_oi:.2f} contradicts UP (-{boost_per_signal:.0f}%)")
    elif pcr_oi < pcr_bullish_threshold:
        if model_direction == "UP":
            raw_adj += boost_per_signal
            signals.append(f"PCR_OI={pcr_oi:.2f} confirms UP (+{boost_per_signal:.0f}%)")
        else:
            raw_adj -= boost_per_signal
            signals.append(f"PCR_OI={pcr_oi:.2f} contradicts DOWN (-{boost_per_signal:.0f}%)")

    # Signal 2: IV Skew alignment
    if iv_skew > 0.015:
        if model_direction == "DOWN":
            raw_adj += boost_per_signal
            signals.append(f"IV_Skew={iv_skew:.3f} (PE bid) confirms DOWN (+{boost_per_signal:.0f}%)")
        else:
            raw_adj -= boost_per_signal / 2
            signals.append(f"IV_Skew={iv_skew:.3f} (PE bid) mild contradiction (-{boost_per_signal/2:.0f}%)")
    elif iv_skew < -0.015:
        if model_direction == "UP":
            raw_adj += boost_per_signal
            signals.append(f"IV_Skew={iv_skew:.3f} (CE bid) confirms UP (+{boost_per_signal:.0f}%)")

    # Signal 3: Max Pain gravity
    if mp_dist < -0.3 and model_direction == "DOWN":
        raw_adj += boost_per_signal / 2
        signals.append(f"Max Pain below spot ({mp_dist:.2f}%) gravity confirms DOWN")
    elif mp_dist > 0.3 and model_direction == "UP":
        raw_adj += boost_per_signal / 2
        signals.append(f"Max Pain above spot ({mp_dist:.2f}%) gravity confirms UP")

    adj = float(max(min(raw_adj, max_total_adjustment), -max_total_adjustment))

    return {
        "adjustment_pct": round(adj, 2),
        "signals": signals,
        "pcr_oi":  pcr_oi,
        "iv_skew": iv_skew,
        "description": " | ".join(signals) if signals else "PCR neutral — no adjustment",
    }
