"""
utils/nifty_vwap_gate.py
─────────────────────────
Fix #5: 1-minute intraday VWAP gatekeeper for Index signals.

Problem: Yahoo Finance ^NSEI / ^BSESN often have Volume=0 on 1-min candles,
         breaking a standard sum(P×V)/sum(V) VWAP calculation.

Solution — 3-tier fallback:
  Tier 1: Use ^NSEI (or ^BSESN) 1-min candles IF >5% have non-zero volume.
  Tier 2: Use NIFTYBEES.NS 1-min candles (real volume, tracks Nifty closely).
           VWAP computed in NIFTYBEES price units (~Nifty/100), then scaled ×100.
  Tier 3: TWAP — simple EMA of 1-min typical prices (volume-free fallback).

Gate logic (fixed typo from blueprint — no "CDSL spot" reference):
  If model is DOWN (PUT): Index spot MUST trade BELOW 09:15–09:20 VWAP.
  If model is UP  (CALL): Index spot MUST trade ABOVE 09:15–09:20 VWAP.
  If NOT confirmed: hold 1 candle, re-check at 09:25 AM.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    NIFTY_TICKER, SENSEX_TICKER, NIFTYBEES_TICKER,
    VWAP_FETCH_PERIOD, VWAP_INTERVAL,
    VWAP_MIN_VOLUME_THRESHOLD, VWAP_NIFTYBEES_SCALE,
)

logger = logging.getLogger("nifty_vwap_gate")


def _compute_vwap(df_1m: pd.DataFrame) -> float:
    """Standard VWAP: sum(Typical_Price × Volume) / sum(Volume)."""
    tp  = (df_1m['High'] + df_1m['Low'] + df_1m['Close']) / 3.0
    vol = df_1m['Volume'].replace(0, np.nan)
    vwap = (tp * vol).sum() / vol.sum()
    return float(vwap)


def _compute_twap_ema(df_1m: pd.DataFrame, span: int = 15) -> float:
    """
    Fallback TWAP: EMA of typical prices over `span` candles.
    Used when volume is consistently 0.
    """
    tp  = (df_1m['High'] + df_1m['Low'] + df_1m['Close']) / 3.0
    ema = tp.ewm(span=span, adjust=False).mean()
    return float(ema.iloc[-1])


def _has_usable_volume(df_1m: pd.DataFrame) -> bool:
    """Return True if >VWAP_MIN_VOLUME_THRESHOLD fraction of candles have non-zero volume."""
    if df_1m.empty or 'Volume' not in df_1m.columns:
        return False
    non_zero = (df_1m['Volume'] > 0).mean()
    return float(non_zero) > VWAP_MIN_VOLUME_THRESHOLD


def _fetch_1min(sym: str) -> pd.DataFrame:
    """Fetch today's 1-min OHLCV data."""
    try:
        df = yf.download(sym, period=VWAP_FETCH_PERIOD,
                         interval=VWAP_INTERVAL, auto_adjust=True,
                         progress=False, show_errors=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("Asia/Kolkata")
        return df
    except Exception as e:
        logger.debug(f"1-min fetch failed for {sym}: {e}")
        return pd.DataFrame()


def _get_current_spot(sym: str) -> Optional[float]:
    """Fast live spot price for the index."""
    try:
        fi = yf.Ticker(sym).fast_info
        price = getattr(fi, "last_price", None) or getattr(fi, "open", None)
        return float(price) if price else None
    except Exception:
        return None


def compute_intraday_vwap(index_ticker: str = NIFTY_TICKER) -> Dict[str, Any]:
    """
    Compute VWAP for `index_ticker` with NIFTYBEES fallback and TWAP fallback.
    Returns dict: {vwap, method, spot, is_available}
    """
    # ── Tier 1: Try primary index ticker ──────────────────────────────────────
    df_spot = _fetch_1min(index_ticker)
    if not df_spot.empty and _has_usable_volume(df_spot):
        vwap   = _compute_vwap(df_spot)
        spot   = _get_current_spot(index_ticker) or float(df_spot['Close'].iloc[-1])
        return {
            "vwap":         round(vwap, 2),
            "spot":         round(spot, 2),
            "method":       f"VWAP via {index_ticker} (real volume)",
            "is_available": True,
        }

    # ── Tier 2: NIFTYBEES.NS proxy ────────────────────────────────────────────
    logger.info(f"{index_ticker} 1-min volume insufficient — using NIFTYBEES proxy.")
    df_bees = _fetch_1min(NIFTYBEES_TICKER)
    if not df_bees.empty and _has_usable_volume(df_bees):
        vwap_bees = _compute_vwap(df_bees)
        vwap      = vwap_bees * VWAP_NIFTYBEES_SCALE   # Scale to Nifty price units
        spot_raw  = _get_current_spot(index_ticker)
        if spot_raw is None:
            bees_spot = float(df_bees['Close'].iloc[-1])
            spot_raw  = bees_spot * VWAP_NIFTYBEES_SCALE
        return {
            "vwap":         round(vwap, 2),
            "spot":         round(spot_raw, 2),
            "method":       "VWAP via NIFTYBEES.NS proxy (×100 scaled)",
            "is_available": True,
        }

    # ── Tier 3: TWAP (volume-free EMA of typical prices) ─────────────────────
    logger.warning("Both index and NIFTYBEES volume unavailable — falling back to TWAP.")
    # Try to use Niftybees for price data at least
    df_tp = df_bees if not df_bees.empty else df_spot
    if not df_tp.empty:
        twap_raw = _compute_twap_ema(df_tp)
        if df_tp is df_bees:
            twap = twap_raw * VWAP_NIFTYBEES_SCALE
        else:
            twap = twap_raw
        spot_raw = _get_current_spot(index_ticker) or twap
        return {
            "vwap":         round(twap, 2),
            "spot":         round(spot_raw, 2),
            "method":       "TWAP (EMA of typical price — volume unavailable)",
            "is_available": True,
        }

    return {
        "vwap":         None,
        "spot":         _get_current_spot(index_ticker),
        "method":       "UNAVAILABLE",
        "is_available": False,
    }


def check_vwap_gate(
    model_direction: str,           # "UP" or "DOWN"
    index_ticker: str = NIFTY_TICKER,
    vwap_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fix #5 + Blueprint Layer 4 (corrected — no "CDSL spot" reference):
    Gate the ML model's trade signal using VWAP confirmation.

    Rules:
      If model is DOWN (PUT): Index spot MUST trade BELOW VWAP.
      If model is UP  (CALL): Index spot MUST trade ABOVE VWAP.
      If NOT confirmed: returns gate_passed=False → wait for 09:25 candle.
    """
    if vwap_data is None:
        vwap_data = compute_intraday_vwap(index_ticker)

    vwap  = vwap_data.get("vwap")
    spot  = vwap_data.get("spot")
    avail = vwap_data.get("is_available", False)

    if not avail or vwap is None or spot is None:
        return {
            "gate_passed":    False,
            "reason":         "VWAP data unavailable — wait for 09:25 candle or trade cautiously",
            "vwap":           vwap,
            "spot":           spot,
            "spot_vs_vwap":   None,
            "method":         vwap_data.get("method", "UNAVAILABLE"),
        }

    spot_vs_vwap = round((spot - vwap) / vwap * 100, 4) if vwap > 0 else 0.0

    if model_direction == "UP":
        gate_passed = spot > vwap
        direction_desc = "CE (CALL)"
        condition_desc = f"Index spot ({spot:.2f}) {'ABOVE' if gate_passed else 'BELOW'} VWAP ({vwap:.2f})"
        reason = (f"✅ VWAP confirmed for UP/CE: Spot {spot:.2f} > VWAP {vwap:.2f}"
                  if gate_passed else
                  f"❌ VWAP not confirmed for UP/CE: Spot {spot:.2f} BELOW VWAP {vwap:.2f} — wait 09:25")
    else:  # DOWN
        gate_passed = spot < vwap
        direction_desc = "PE (PUT)"
        condition_desc = f"Index spot ({spot:.2f}) {'BELOW' if gate_passed else 'ABOVE'} VWAP ({vwap:.2f})"
        reason = (f"✅ VWAP confirmed for DOWN/PE: Spot {spot:.2f} < VWAP {vwap:.2f}"
                  if gate_passed else
                  f"❌ VWAP not confirmed for DOWN/PE: Spot {spot:.2f} ABOVE VWAP {vwap:.2f} — wait 09:25")

    return {
        "gate_passed":    gate_passed,
        "reason":         reason,
        "condition_desc": condition_desc,
        "direction_desc": direction_desc,
        "vwap":           vwap,
        "spot":           spot,
        "spot_vs_vwap":   spot_vs_vwap,
        "method":         vwap_data.get("method"),
    }
