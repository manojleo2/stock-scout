"""
utils/options_feed.py
─────────────────────
Unified Options Price Feed for CDSL Options.

3-Tier Architecture:
  1. Priority 1: Groww Trade API (`growwapi`) - Direct broker market data feed.
  2. Priority 2: Direct NSE Option Chain Web Scraper (Live exchange backup).
  3. Priority 3: Black-Scholes-Merton (BSM) Mathematical Engine (Offline fallback).

Usage:
  from utils.options_feed import get_live_option_quote
  quote = get_live_option_quote(symbol="CDSL.NS", strike=1340, is_call=False, spot=1335.0)
  print(quote["ltp"], quote["source"])
"""

import os
import logging
import datetime as dt
import requests
from typing import Dict, Any, Optional

from utils.paper_trading import calculate_bsm_option_price

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OptionsFeed")

# ── Load environment variables from .env if present ───────────────────────────
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")

def _load_env_vars():
    # 1. Load from Streamlit secrets if running in Streamlit Cloud
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for k in [
                "GROWW_AUTH_TOKEN", "GROWW_TOTP_TOKEN", "GROWW_TOTP_SECRET",
                "GROWW_API_KEY", "GROWW_API_SECRET", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"
            ]:
                if k in st.secrets and k not in os.environ:
                    os.environ[k] = str(st.secrets[k])
    except Exception:
        pass

    # 2. Load from local .env
    if os.path.exists(ENV_PATH):
        try:
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() not in os.environ:
                            os.environ[k.strip()] = v.strip().strip('"').strip("'")
        except Exception as e:
            logger.warning(f"Could not load .env file: {e}")

_load_env_vars()

GROWW_AUTH_TOKEN  = os.environ.get("GROWW_AUTH_TOKEN", "")
GROWW_TOTP_TOKEN  = os.environ.get("GROWW_TOTP_TOKEN", "")
GROWW_TOTP_SECRET = os.environ.get("GROWW_TOTP_SECRET", "")
GROWW_API_KEY     = os.environ.get("GROWW_API_KEY", "")
GROWW_API_SECRET  = os.environ.get("GROWW_API_SECRET", "")

_groww_client = None


def generate_totp_code(secret: str) -> str:
    """Generate standard RFC 6238 6-digit TOTP code using standard library."""
    import base64
    import hashlib
    import hmac
    import struct
    import time
    key = base64.b32decode(secret.strip().upper().replace(" ", ""))
    counter = int(time.time()) // 30
    msg = struct.pack(">Q", counter)
    h = hmac.new(key, msg, hashlib.sha1).digest()
    offset = h[19] & 15
    code = (struct.unpack(">I", h[offset:offset + 4])[0] & 0x7FFFFFFF) % 1000000
    return f"{code:06d}"


def _save_token_to_env(new_token: str):
    """Persist freshly minted Groww access token to .env."""
    try:
        lines = []
        token_written = False
        if os.path.exists(ENV_PATH):
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("GROWW_AUTH_TOKEN="):
                        lines.append(f'GROWW_AUTH_TOKEN="{new_token}"\n')
                        token_written = True
                    else:
                        lines.append(line)
        if not token_written:
            lines.insert(0, f'GROWW_AUTH_TOKEN="{new_token}"\n')
        with open(ENV_PATH, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception as e:
        logger.warning(f"Could not write fresh token to .env: {e}")


def refresh_groww_access_token() -> Optional[str]:
    """Auto-generate a fresh daily access token using TOTP token & API secret."""
    _load_env_vars()
    totp_token = os.environ.get("GROWW_TOTP_TOKEN", "")
    secret = os.environ.get("GROWW_TOTP_SECRET", "")
    if not totp_token or not secret:
        return None

    try:
        from growwapi import GrowwAPI
        code = generate_totp_code(secret)
        token_resp = GrowwAPI.get_access_token(api_key=totp_token, totp=code)
        if isinstance(token_resp, str) and token_resp:
            os.environ["GROWW_AUTH_TOKEN"] = token_resp
            _save_token_to_env(token_resp)
            logger.info("Successfully auto-generated fresh Groww access token via TOTP.")
            return token_resp
        elif isinstance(token_resp, dict) and "access_token" in token_resp:
            t = token_resp["access_token"]
            os.environ["GROWW_AUTH_TOKEN"] = t
            _save_token_to_env(t)
            logger.info("Successfully auto-generated fresh Groww access token via TOTP.")
            return t
    except Exception as e:
        logger.warning(f"Auto-refreshing Groww access token failed: {e}")
    return None


def _check_groww_market_data_role(token: str) -> bool:
    """Inspect JWT roles to check if market-data / quotes are permitted."""
    import base64
    import json
    try:
        parts = token.split(".")
        if len(parts) >= 2:
            payload_b64 = parts[1]
            payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            sub = json.loads(payload.get("sub", "{}"))
            return "market-data" in sub.get("role", "")
    except Exception:
        pass
    return False


def get_groww_client():
    """
    Initialize or reuse authenticated GrowwAPI client.
    Automatically refreshes access token using TOTP credentials if expired.
    """
    global _groww_client, _groww_market_data_available
    if _groww_client is not None:
        return _groww_client

    try:
        from growwapi import GrowwAPI
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(f"growwapi package not available: {e}")
        return None

    token = os.environ.get("GROWW_AUTH_TOKEN", "")
    if token:
        try:
            client = GrowwAPI(token=token)
            # Lightweight verification call
            client.get_user_profile()
            _groww_client = client
            _groww_market_data_available = _check_groww_market_data_role(token)
            logger.info("GrowwAPI client verified and initialized.")
            return _groww_client
        except Exception:
            logger.info("Existing Groww auth token expired or invalid; triggering auto-refresh...")

    # Automatic generation via TOTP
    new_token = refresh_groww_access_token()
    if new_token:
        try:
            _groww_client = GrowwAPI(token=new_token)
            _groww_market_data_available = _check_groww_market_data_role(new_token)
            logger.info("GrowwAPI client initialized with newly minted daily token.")
            return _groww_client
        except Exception as e:
            logger.warning(f"Could not initialize GrowwAPI with fresh token: {e}")

    return None


def get_groww_balance() -> Optional[float]:
    """Fetch live available cash margin from Groww broker API."""
    client = get_groww_client()
    if client is None:
        return None
    try:
        margin = client.get_available_margin_details()
        return float(margin.get("clear_cash") or margin.get("fno_margin_details", {}).get("option_buy_balance_available", 0.0))
    except Exception as e:
        logger.warning(f"Could not fetch Groww margin balance: {e}")
        return None


def get_groww_positions() -> list:
    """Fetch current intraday & carry-forward positions from Groww."""
    client = get_groww_client()
    if client is None:
        return []
    try:
        res = client.get_positions_for_user()
        return res.get("positions", []) if isinstance(res, dict) else (res if isinstance(res, list) else [])
    except Exception as e:
        logger.warning(f"Could not fetch Groww positions: {e}")
        return []


_groww_market_data_available: Optional[bool] = None
_nse_session: Optional[requests.Session] = None


def fetch_from_groww(symbol: str, strike: int, is_call: bool) -> Optional[Dict[str, Any]]:
    """Try to fetch live option quote via official Groww API."""
    global _groww_market_data_available
    if _groww_market_data_available is False:
        return None

    client = get_groww_client()
    if client is None:
        return None

    clean_sym = symbol.replace(".NS", "").replace(".BO", "")
    opt_type  = "CE" if is_call else "PE"

    try:
        # Search or get option quote for current month
        expiries_resp = client.get_expiries(exchange="NSE", underlying_symbol=clean_sym, timeout=2)
        _groww_market_data_available = True
        expiries = expiries_resp.get("expiry_dates", []) if isinstance(expiries_resp, dict) else (expiries_resp if isinstance(expiries_resp, list) else [])
        expiry_date = expiries[0] if expiries else None

        if expiry_date:
            chain = client.get_option_chain(exchange="NSE", underlying=clean_sym, expiry_date=expiry_date, timeout=2)
            strikes = chain.get("strikes", [])
            for item in strikes:
                if item.get("strike_price") == strike:
                    leg = item.get("call" if is_call else "put", {})
                    ltp = float(leg.get("ltp") or leg.get("last_price") or 0.0)
                    if ltp > 0:
                        return {
                            "source": "GROWW_API",
                            "contract": f"{clean_sym} ₹{strike} {opt_type}",
                            "ltp": round(ltp, 2),
                            "bid": round(float(leg.get("bid_price") or ltp), 2),
                            "ask": round(float(leg.get("ask_price") or ltp), 2),
                            "iv": float(leg.get("iv") or 0.32),
                            "oi": int(leg.get("open_interest") or 0),
                            "status": "LIVE",
                        }
    except Exception as e:
        err_str = str(e).lower()
        if "forbidden" in err_str or "403" in err_str:
            logger.info("Groww API token does not have market-data permissions; bypassing Groww quote tier.")
            _groww_market_data_available = False
        else:
            logger.debug(f"Groww live option fetch error: {e}")

    return None


def _get_nse_session() -> requests.Session:
    """Get or create reusable warm session for NSE requests."""
    global _nse_session
    if _nse_session is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.nseindia.com/option-chain",
        })
        try:
            s.get("https://www.nseindia.com", timeout=2)
        except Exception:
            pass
        _nse_session = s
    return _nse_session


def fetch_from_nse_scraper(symbol: str, strike: int, is_call: bool) -> Optional[Dict[str, Any]]:
    """Scrape live NSE option chain with browser session emulation."""
    clean_sym = symbol.replace(".NS", "").replace(".BO", "")
    opt_type  = "CE" if is_call else "PE"
    s = _get_nse_session()

    try:
        url = f"https://www.nseindia.com/api/option-chain-equities?symbol={clean_sym}"
        r = s.get(url, timeout=2.5)
        if r.status_code == 200:
            data = r.json()
            records = data.get("records", {}).get("data", [])
            for rec in records:
                if rec.get("strikePrice") == strike:
                    side = rec.get("CE" if is_call else "PE", {})
                    ltp = float(side.get("lastPrice") or 0.0)
                    if ltp > 0:
                        return {
                            "source": "NSE_SCRAPER",
                            "contract": f"{clean_sym} ₹{strike} {opt_type}",
                            "ltp": round(ltp, 2),
                            "bid": round(float(side.get("buyPrice") or ltp), 2),
                            "ask": round(float(side.get("sellPrice") or ltp), 2),
                            "iv": float(side.get("impliedVolatility") or 0.32),
                            "oi": int(side.get("openInterest") or 0),
                            "status": "LIVE",
                        }
    except Exception as e:
        logger.debug(f"NSE scraper fetch error: {e}")

    return None


def get_live_option_quote(symbol: str = "CDSL.NS",
                          strike: int = 1340,
                          is_call: bool = False,
                          spot: float = 1335.0,
                          days_to_expiry: float = 10.0) -> Dict[str, Any]:
    """
    Master function: queries Groww API -> NSE Scraper -> BSM Fallback.
    Guarantees a clean, robust quote dictionary with exact LTP.
    """
    clean_sym = symbol.replace(".NS", "").replace(".BO", "")
    opt_type  = "CE" if is_call else "PE"

    # Tier 1: Groww Trade API
    groww_res = fetch_from_groww(symbol, strike, is_call)
    if groww_res:
        return groww_res

    # Tier 2: NSE Option Chain Live Scraper
    nse_res = fetch_from_nse_scraper(symbol, strike, is_call)
    if nse_res:
        return nse_res

    # Tier 3: Mathematical BSM Engine Fallback
    bsm_price = calculate_bsm_option_price(spot, strike, days_to_expiry=days_to_expiry, is_call=is_call)
    return {
        "source": "BSM_ENGINE (Model Estimate)",
        "contract": f"{clean_sym} ₹{strike} {opt_type}",
        "ltp": round(bsm_price, 2),
        "bid": round(max(bsm_price - 0.20, 0.5), 2),
        "ask": round(bsm_price + 0.20, 2),
        "iv": 0.32,
        "oi": 0,
        "status": "ESTIMATED",
    }
