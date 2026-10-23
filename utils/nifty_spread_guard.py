"""
utils/nifty_spread_guard.py
───────────────────────────
Standalone Quantitative Spread Guard & Microstructure Execution Engine for System 2.

Features:
  1. Configurable Multi-Threshold Spread Guard:
     - Absolute spread cap (₹): max acceptable difference between Ask and Bid.
     - Relative spread cap (%): max spread as percentage of mid-price / LTP.
     - Quote freshness threshold (s): maximum allowable quote age before stand-down.
  2. Realistic Executable Price Model:
     - Buy-side execution: Ask + half-spread/adverse slippage.
     - Sell-side execution: Bid - half-spread/adverse slippage.
  3. Official 2026 Indian F&O Statutory Transaction Costs:
     - STT: 0.10% on sell-side option premium turnover (effective Oct 1, 2024 / 2026 rules).
     - Exchange Transaction Charge (NSE): 0.0505% of premium turnover.
     - GST: 18% on (Exchange Charges + Brokerage).
     - SEBI Turnover Fee: 0.0001% (₹10 / crore) of premium turnover.
     - Stamp Duty: 0.003% on buy-side premium turnover.
     - Brokerage: Flat ₹20 per executed order (standard discount broker rate).

Design Constraint:
  - This is a research & analysis engine. It does NOT submit orders or alter live alerting flows.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import time


@dataclass
class SpreadGuardConfig:
    """Configurable thresholds for option spread guard."""
    max_spread_abs: float = 1.50         # Absolute maximum spread in ₹ (Nifty default: 1.50)
    max_spread_pct: float = 2.00         # Relative maximum spread as % of mid/LTP (2.0%)
    max_quote_age_sec: float = 10.0      # Maximum quote staleness in seconds
    tick_size: float = 0.05              # Exchange tick size (₹0.05 for NSE options)
    slippage_ticks: float = 1.0          # Adverse slippage buffer in ticks (1 tick = ₹0.05)
    brokerage_per_order: float = 20.0    # Flat brokerage fee per executed order in ₹


# Symbol-specific default profiles
SPREAD_CONFIG_NIFTY = SpreadGuardConfig(
    max_spread_abs=1.50,
    max_spread_pct=2.00,
    max_quote_age_sec=10.0,
    tick_size=0.05,
    slippage_ticks=1.0,
    brokerage_per_order=20.0
)

SPREAD_CONFIG_SENSEX = SpreadGuardConfig(
    max_spread_abs=3.00,
    max_spread_pct=2.00,
    max_quote_age_sec=10.0,
    tick_size=0.05,
    slippage_ticks=2.0,
    brokerage_per_order=20.0
)


def get_spread_config(ticker: str) -> SpreadGuardConfig:
    """Returns asset-specific spread guard configuration."""
    if "BSESN" in ticker or "SENSEX" in ticker.upper():
        return SPREAD_CONFIG_SENSEX
    return SPREAD_CONFIG_NIFTY


def evaluate_spread_guard(
    bid: Optional[float],
    ask: Optional[float],
    ltp: float,
    quote_timestamp: Optional[float] = None,
    config: Optional[SpreadGuardConfig] = None
) -> Dict[str, Any]:
    """
    Evaluates whether an option contract satisfies liquidity & spread viability.
    
    Returns:
      is_tradable: bool
      spread_research_result: str ('OBSERVED_VALID', 'UNOBSERVABLE', 'STALE_QUOTE', 'WIDE_SPREAD')
      rejection_reason: Optional[str]
      spread_abs: Optional[float]
      spread_pct: Optional[float]
      mid_price: float
      quote_age_sec: Optional[float]
      quote_freshness: str ('FRESH', 'STALE', 'UNKNOWN')
    """
    cfg = config or SPREAD_CONFIG_NIFTY
    now = time.time()

    # Track quote age and freshness without defaulting to local receipt time
    if quote_timestamp is not None and quote_timestamp > 0:
        quote_age = round(now - quote_timestamp, 2)
        quote_freshness = "FRESH" if quote_age <= cfg.max_quote_age_sec else "STALE"
    else:
        quote_age = None
        quote_freshness = "UNKNOWN"

    # 1. Missing authentic bid/ask -> UNOBSERVABLE spread research result
    if bid is None or ask is None or bid <= 0.0 or ask <= 0.0:
        return {
            "is_tradable": False,
            "spread_research_result": "UNOBSERVABLE",
            "research_status": "UNOBSERVABLE",
            "rejection_reason": "UNOBSERVABLE (Authentic top-of-book bid/ask absent from feed)",
            "spread_abs": None,
            "spread_pct": None,
            "mid_price": ltp,
            "quote_age_sec": quote_age,
            "quote_freshness": quote_freshness,
            "effective_entry_price": ltp,
        }

    # 2. Crossed Book Anomaly
    if ask < bid:
        return {
            "is_tradable": False,
            "spread_research_result": "UNOBSERVABLE",
            "research_status": "UNOBSERVABLE",
            "rejection_reason": "CROSSED_BOOK (Ask < Bid anomaly)",
            "spread_abs": round(ask - bid, 2),
            "spread_pct": None,
            "mid_price": ltp,
            "quote_age_sec": quote_age,
            "quote_freshness": quote_freshness,
            "effective_entry_price": ask,
        }

    spread_abs = round(ask - bid, 2)
    mid_price = round((ask + bid) / 2.0, 2)
    ref_price = mid_price if mid_price > 0 else (ltp if ltp > 0 else 1.0)
    spread_pct = round((spread_abs / ref_price) * 100.0, 3)

    # 3. Quote Staleness Check (only if exchange timestamp was provided)
    if quote_freshness == "STALE":
        return {
            "is_tradable": False,
            "spread_research_result": "STALE_QUOTE",
            "research_status": "STALE_QUOTE",
            "rejection_reason": f"STALE_QUOTE (Age {quote_age:.1f}s > {cfg.max_quote_age_sec}s)",
            "spread_abs": spread_abs,
            "spread_pct": spread_pct,
            "mid_price": mid_price,
            "quote_age_sec": quote_age,
            "quote_freshness": quote_freshness,
            "effective_entry_price": ask,
        }

    # 4. Absolute Spread Check
    if spread_abs > cfg.max_spread_abs:
        return {
            "is_tradable": False,
            "spread_research_result": "WIDE_SPREAD",
            "research_status": "WIDE_SPREAD",
            "rejection_reason": f"WIDE_ABSOLUTE_SPREAD (₹{spread_abs:.2f} > ₹{cfg.max_spread_abs:.2f})",
            "spread_abs": spread_abs,
            "spread_pct": spread_pct,
            "mid_price": mid_price,
            "quote_age_sec": quote_age,
            "quote_freshness": quote_freshness,
            "effective_entry_price": ask,
        }

    # 5. Relative Spread Check
    if spread_pct > cfg.max_spread_pct:
        return {
            "is_tradable": False,
            "spread_research_result": "WIDE_SPREAD",
            "research_status": "WIDE_SPREAD",
            "rejection_reason": f"WIDE_RELATIVE_SPREAD ({spread_pct:.2f}% > {cfg.max_spread_pct:.2f}%)",
            "spread_abs": spread_abs,
            "spread_pct": spread_pct,
            "mid_price": mid_price,
            "quote_age_sec": quote_age,
            "quote_freshness": quote_freshness,
            "effective_entry_price": ask,
        }

    # 6. Executable buy price with adverse slippage
    effective_entry = round(ask + (cfg.slippage_ticks * cfg.tick_size), 2)

    return {
        "is_tradable": True,
        "spread_research_result": "OBSERVED_VALID",
        "research_status": "OBSERVED_VALID",
        "rejection_reason": None,
        "spread_abs": spread_abs,
        "spread_pct": spread_pct,
        "mid_price": mid_price,
        "quote_age_sec": quote_age,
        "quote_freshness": quote_freshness,
        "effective_entry_price": effective_entry,
    }


def compute_executable_exit_price(
    bid: float,
    ltp: float,
    slippage_ticks: float = 1.0,
    tick_size: float = 0.05
) -> float:
    """Computes realistic executable sell-side exit price (Bid minus slippage)."""
    ref_bid = bid if bid > 0 else (ltp - 0.20)
    return round(max(0.05, ref_bid - (slippage_ticks * tick_size)), 2)


def calculate_fo_transaction_costs(
    buy_premium: float,
    sell_premium: float,
    quantity: int,
    brokerage_per_order: float = 20.0
) -> Dict[str, float]:
    """
    Computes exact statutory Indian F&O transaction costs (effective 2026 exchange tax schedule).
    
    Turnover definition:
      Buy Turnover  = buy_premium × quantity
      Sell Turnover = sell_premium × quantity
      Total Turnover = Buy Turnover + Sell Turnover
    """
    buy_turnover = buy_premium * quantity
    sell_turnover = sell_premium * quantity
    total_turnover = buy_turnover + sell_turnover

    # 1. Brokerage (flat per leg)
    brokerage = brokerage_per_order * 2.0  # entry + exit

    # 2. STT (Securities Transaction Tax) = 0.10% on SELL turnover only
    stt = round(sell_turnover * 0.0010, 2)

    # 3. Exchange Transaction Charges = 0.0505% of total premium turnover
    exchange_charges = round(total_turnover * 0.000505, 2)

    # 4. GST = 18% on (Brokerage + Exchange Charges)
    gst = round((brokerage + exchange_charges) * 0.18, 2)

    # 5. SEBI Turnover Charges = ₹10 / crore (0.000001 = 0.0001%)
    sebi_charges = round(total_turnover * 0.000001, 2)

    # 6. Stamp Duty = 0.003% on BUY turnover only
    stamp_duty = round(buy_turnover * 0.00003, 2)

    total_cost = round(brokerage + stt + exchange_charges + gst + sebi_charges + stamp_duty, 2)

    return {
        "brokerage": brokerage,
        "stt": stt,
        "exchange_charges": exchange_charges,
        "gst": gst,
        "sebi_charges": sebi_charges,
        "stamp_duty": stamp_duty,
        "total_cost": total_cost,
        "cost_per_qty": round(total_cost / quantity, 2) if quantity > 0 else 0.0,
    }
