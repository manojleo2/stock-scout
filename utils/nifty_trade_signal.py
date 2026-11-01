"""
utils/nifty_trade_signal.py
────────────────────────────
Trade Signal Generator — final output with all 5 fixes applied.

Pipeline:
  1. ML model conviction (from nifty_sensex_model.py)
  2. Live OI/PCR conviction modifier (Fix #3 — Group B, post-model only)
  3. VWAP gate (Fix #5)
  4. Dynamic Target/SL (Fix #4 — % of entry premium)
  5. Formatted signal output for dashboard + Telegram

Fix #4: Dynamic Target = entry × 1.20, SL = entry × 0.90, Trail at +12% → cost,
        Lock at +18% → SL moves to +10% profit. Scales across Nifty/Sensex.
"""

import sys
import logging
import math
from typing import Dict, Any, Optional
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np

from config import (
    NIFTY_TICKER, SENSEX_TICKER,
    NIFTY_LOT_SIZE, SENSEX_LOT_SIZE,
    NIFTY_STRIKE_STEP, SENSEX_STRIKE_STEP,
    NIFTY_TARGET_PCT, NIFTY_SL_PCT,
    NIFTY_TRAIL_TRIGGER_PCT, NIFTY_TRAIL_LOCK_PCT, NIFTY_TRAIL_LOCK_PROFIT_PCT,
    NIFTY_HARD_CUTOFF_TIME,
    PCR_BEARISH_THRESHOLD, PCR_BULLISH_THRESHOLD,
    PCR_CONVICTION_BOOST, MAX_PCR_ADJUSTMENT,
)
from utils.nifty_oi_pcr import (
    get_live_nifty_oi_summary, compute_pcr_conviction_modifier,
    get_groww_account_status,
)
from utils.nifty_vwap_gate import check_vwap_gate, compute_intraday_vwap
from utils.nifty_sensex_model import estimate_atm_premium
from utils.nifty_confluence import compute_confluence_score, get_market_phase

logger = logging.getLogger("nifty_trade_signal")

# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Target / SL Calculator (Fix #4)
# ─────────────────────────────────────────────────────────────────────────────

def compute_dynamic_levels(entry_premium: float, lot_size: int) -> Dict[str, Any]:
    """
    Fix #4: Replace fixed ₹10 target / ₹5 SL with % of entry premium.
    Scales correctly across:
      - Nifty near ₹24,500 (premium ~₹80–150)
      - Sensex near ₹80,000 (premium ~₹200–400)
      - Expiry days (premium ₹20–50)

    Rules:
      Target        = entry × 1.20    (+20%)
      Stop Loss     = entry × 0.90    (−10%)
      Trail trigger = entry × 1.12    (+12%) → move SL to entry (cost)
      Lock profit   = entry × 1.18    (+18%) → SL moves to entry × 1.10 (+10%)
    """
    target_premium       = round(entry_premium * (1 + NIFTY_TARGET_PCT / 100), 2)
    sl_premium           = round(entry_premium * (1 - NIFTY_SL_PCT / 100), 2)
    trail_trigger_premium= round(entry_premium * (1 + NIFTY_TRAIL_TRIGGER_PCT / 100), 2)
    lock_trigger_premium = round(entry_premium * (1 + NIFTY_TRAIL_LOCK_PCT / 100), 2)
    locked_sl_premium    = round(entry_premium * (1 + NIFTY_TRAIL_LOCK_PROFIT_PCT / 100), 2)

    gain_per_lot  = round((target_premium - entry_premium) * lot_size, 0)
    loss_per_lot  = round((entry_premium - sl_premium)     * lot_size, 0)
    rr_ratio      = round(gain_per_lot / loss_per_lot, 2) if loss_per_lot > 0 else 99.0

    return {
        "entry_premium":        entry_premium,
        "target_premium":       target_premium,
        "sl_premium":           sl_premium,
        "trail_trigger_premium":trail_trigger_premium,
        "lock_trigger_premium": lock_trigger_premium,
        "locked_sl_premium":    locked_sl_premium,
        "gain_per_lot":         int(gain_per_lot),
        "loss_per_lot":         int(loss_per_lot),
        "rr_ratio":             rr_ratio,
        "target_pct":           NIFTY_TARGET_PCT,
        "sl_pct":               NIFTY_SL_PCT,
        "trail_trigger_pct":    NIFTY_TRAIL_TRIGGER_PCT,
        "lock_trigger_pct":     NIFTY_TRAIL_LOCK_PCT,
        "lock_profit_pct":      NIFTY_TRAIL_LOCK_PROFIT_PCT,
        "lot_size":             lot_size,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ATM Strike Calculator
# ─────────────────────────────────────────────────────────────────────────────

def get_atm_strike(spot: float, ticker: str = NIFTY_TICKER) -> int:
    step = NIFTY_STRIKE_STEP if ticker == NIFTY_TICKER else SENSEX_STRIKE_STEP
    return int(round(spot / step) * step)


def get_lot_size(ticker: str) -> int:
    return NIFTY_LOT_SIZE if ticker == NIFTY_TICKER else SENSEX_LOT_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Master Signal Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_trade_signal(
    model_result: dict,
    ticker: str = NIFTY_TICKER,
    fetch_live_oi: bool = True,
    check_vwap: bool = True,
    overnight_cues: Optional[dict] = None,
    other_index_result: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Generate complete trade signal from ML model output.

    Steps:
      1. Validate model result
      2. Get live OI/PCR conviction modifier (Fix #3)
      3. Check VWAP gate (Fix #5)
      4. Compute ATM strike + entry premium estimate
      5. Compute dynamic Target/SL (Fix #4)
      6. Assemble final signal dict

    Returns: full trade signal dict (+ formatting strings for dashboard)
    """
    if model_result.get("status") != "success":
        return {
            "signal": "NO_SIGNAL",
            "reason": model_result.get("message", "Model error"),
        }

    direction        = model_result["direction"]           # "UP" or "DOWN"
    base_conviction  = model_result["conviction_pct"]
    is_tradeable     = model_result["is_tradeable"]
    regime           = model_result["regime"]
    latest_close     = model_result["latest_close"]
    atr_14           = model_result["atr_14"]
    days_to_expiry   = model_result["days_to_nifty_expiry"] if ticker == NIFTY_TICKER \
                       else model_result["days_to_sensex_expiry"]
    lot_size         = get_lot_size(ticker)
    strike_step      = NIFTY_STRIKE_STEP if ticker == NIFTY_TICKER else SENSEX_STRIKE_STEP

    # ── Step 1: OI/PCR Live Conviction Modifier (Fix #3) ─────────────────────
    oi_summary  = {}
    oi_modifier = {"adjustment_pct": 0.0, "signals": [], "description": "OI fetch skipped"}
    if fetch_live_oi:
        try:
            oi_summary  = get_live_nifty_oi_summary(spot=latest_close, strike_step=strike_step, ticker=ticker)
            oi_modifier = compute_pcr_conviction_modifier(
                oi_parsed=oi_summary,
                model_direction=direction,
                pcr_bearish_threshold=PCR_BEARISH_THRESHOLD,
                pcr_bullish_threshold=PCR_BULLISH_THRESHOLD,
                boost_per_signal=PCR_CONVICTION_BOOST,
                max_total_adjustment=MAX_PCR_ADJUSTMENT,
            )
        except Exception as e:
            logger.warning(f"OI/PCR fetch failed: {e}")

    final_conviction = round(base_conviction + oi_modifier.get("adjustment_pct", 0.0), 2)
    final_conviction = float(np.clip(final_conviction, 0.0, 99.9))

    # Re-check tradeability with final conviction
    required = model_result.get("required_threshold", 62.0)
    is_tradeable_final = final_conviction >= required

    # ── Step 2: VWAP Gate (Fix #5) ───────────────────────────────────────────
    vwap_result = {}
    if check_vwap:
        try:
            vwap_data   = compute_intraday_vwap(index_ticker=ticker)
            vwap_result = check_vwap_gate(direction, ticker, vwap_data)
        except Exception as e:
            logger.warning(f"VWAP gate failed: {e}")
            vwap_result = {"gate_passed": False, "reason": f"VWAP error: {e}"}

    vwap_confirmed = vwap_result.get("gate_passed", False)

    # ── Step 3: ATM Strike + Entry Premium ───────────────────────────────────
    spot         = vwap_result.get("spot") or latest_close
    atm_strike   = get_atm_strike(spot, ticker)
    is_call      = (direction == "UP")

    # Use Real ATM CE/PE LTP from Groww/NSE if available, fallback to BSM estimate
    live_ltp = oi_summary.get("atm_ce_ltp", 0.0) if is_call else oi_summary.get("atm_pe_ltp", 0.0)
    is_live_premium = False
    if live_ltp and float(live_ltp) > 0.0:
        entry_premium = float(live_ltp)
        is_live_premium = True
    else:
        entry_premium = estimate_atm_premium(
            spot=spot, days_to_expiry=days_to_expiry,
            atr=atr_14, strike_step=strike_step, is_call=is_call,
        )

    # ── Step 4: Dynamic Target / SL (Fix #4) ─────────────────────────────────
    levels = compute_dynamic_levels(entry_premium=entry_premium, lot_size=lot_size)

    # ── Step 5: Instrument name ───────────────────────────────────────────────
    ticker_clean = "NIFTY" if ticker == NIFTY_TICKER else "SENSEX"
    option_type  = "CE" if is_call else "PE"
    contract_name= f"{ticker_clean} {atm_strike} {option_type}"

    # ── Step 6: Multi-Factor Confluence & Action ─────────────────────────────
    if overnight_cues is None:
        try:
            from utils.nifty_overnight_cues import get_overnight_cues
            overnight_cues = get_overnight_cues()
        except Exception:
            overnight_cues = None

    confluence = compute_confluence_score(
        model_result=model_result,
        vwap_data=vwap_data if 'vwap_data' in locals() else {},
        oi_summary=oi_summary,
        overnight_cues=overnight_cues,
        other_index_result=other_index_result,
    )
    confluence_score = confluence["total_confluence"]
    confluence_thresh = confluence["threshold"]
    is_confluence_pass = confluence["is_passed"]

    if not vwap_confirmed:
        action = "⏳ WAIT — VWAP not confirmed (check next candle)"
        signal = "WAIT_VWAP"
        signal_emoji = "⏳"
    elif not is_confluence_pass:
        action = f"⚪ SCANNING — Confluence {confluence_score:.1f}/{confluence_thresh:.0f} pts"
        signal = "NO_SIGNAL"
        signal_emoji = "⚪"
    else:
        signal = "BUY_CE" if is_call else "BUY_PE"
        signal_emoji = "🟢" if is_call else "🔴"
        action = f"{'BUY CALL (CE)' if is_call else 'BUY PUT (PE)'} — {contract_name}"

    now_str = datetime.now().strftime("%d %b %Y %H:%M")

    return {
        # ── Core Signal ───────────────────────────────────────────────────────
        "signal":           signal,
        "signal_emoji":     signal_emoji,
        "action":           action,
        "ticker":           ticker,
        "ticker_clean":     ticker_clean,
        "direction":        direction,
        "option_type":      option_type,
        "contract":         contract_name,
        "atm_strike":       atm_strike,
        "spot":             round(spot, 2),
        "timestamp":        now_str,
        # ── Conviction ────────────────────────────────────────────────────────
        "base_conviction":  base_conviction,
        "oi_adjustment":    oi_modifier.get("adjustment_pct", 0.0),
        "final_conviction": final_conviction,
        "required_threshold": required,
        "confidence":       model_result["confidence"],
        "is_tradeable":     is_tradeable_final,
        # ── Multi-Factor Confluence (New Institutional Engine) ────────────────
        "confluence":       confluence,
        "confluence_score": confluence_score,
        "confluence_thresh": confluence_thresh,
        # ── Regime ────────────────────────────────────────────────────────────
        "regime":           regime,
        "regime_detail":    model_result.get("regime_detail", ""),
        # ── VWAP Gate ─────────────────────────────────────────────────────────
        "vwap":             vwap_result.get("vwap"),
        "vwap_confirmed":   vwap_confirmed,
        "vwap_reason":      vwap_result.get("reason", ""),
        "vwap_method":      vwap_result.get("method", ""),
        # ── Premium & Levels (Fix #4) ─────────────────────────────────────────
        "entry_premium":    levels["entry_premium"],
        "is_live_premium":  is_live_premium,
        "target_premium":   levels["target_premium"],
        "sl_premium":       levels["sl_premium"],
        "trail_trigger":    levels["trail_trigger_premium"],
        "lock_trigger":     levels["lock_trigger_premium"],
        "locked_sl":        levels["locked_sl_premium"],
        "gain_per_lot":     levels["gain_per_lot"],
        "loss_per_lot":     levels["loss_per_lot"],
        "rr_ratio":         levels["rr_ratio"],
        "target_pct":       levels["target_pct"],
        "sl_pct":           levels["sl_pct"],
        "trail_trigger_pct": levels["trail_trigger_pct"],
        "lock_trigger_pct": levels["lock_trigger_pct"],
        "lock_profit_pct":  levels["lock_profit_pct"],
        "lot_size":         lot_size,
        "days_to_expiry":   days_to_expiry,
        # ── Real ATM Strike & Live Prices ─────────────────────────────────────
        "atm_ce_ltp":       oi_summary.get("atm_ce_ltp", 0.0),
        "atm_pe_ltp":       oi_summary.get("atm_pe_ltp", 0.0),
        "groww_account":    get_groww_account_status(),
        # ── OI/PCR Summary ────────────────────────────────────────────────────
        "pcr_oi":           oi_summary.get("pcr_oi", "N/A"),
        "pcr_vol":          oi_summary.get("pcr_vol", "N/A"),
        "iv_skew":          oi_summary.get("iv_skew", "N/A"),
        "max_pain_strike":  oi_summary.get("max_pain_strike", "N/A"),
        "oi_signals":       oi_modifier.get("signals", []),
        "oi_description":   oi_modifier.get("description", ""),
        "vix":              oi_summary.get("vix", "N/A"),
        # ── Rules Reminder ────────────────────────────────────────────────────
        "hard_cutoff":      NIFTY_HARD_CUTOFF_TIME,
    }


def format_signal_console(sig: dict) -> str:
    """Pretty-print signal to console / Telegram."""
    if not sig or (sig.get("signal") == "NO_SIGNAL" and "ticker_clean" not in sig):
        return f"⚠️ Signal generation halted: {sig.get('reason', 'No signal details') if sig else 'Empty response'}"

    prem_type = "(Live LTP)" if sig.get("is_live_premium") else "(est.)"
    tgt_pct = sig.get("target_pct", 30.0)
    sl_pct = sig.get("sl_pct", 15.0)
    trail_pct = sig.get("trail_trigger_pct", 18.0)
    lock_pct = sig.get("lock_trigger_pct", 25.0)
    lock_profit = sig.get("lock_profit_pct", 15.0)
    acc = sig.get("groww_account", {})

    lines = [
        "═" * 55,
        f" 🎯 {sig['ticker_clean']} INTRADAY SIGNAL — {sig['timestamp']}",
        "═" * 55,
        f" Direction    : {'UP 📈' if sig['direction'] == 'UP' else 'DOWN 📉'} ({sig['final_conviction']:.1f}% conviction)",
        f" Contract     : {sig['contract']}  |  Spot: {sig['spot']}",
        f" ATM Strikes  : {sig['atm_strike']} CE: ₹{sig.get('atm_ce_ltp', 0.0):.2f} | {sig['atm_strike']} PE: ₹{sig.get('atm_pe_ltp', 0.0):.2f}",
        f" Entry Premium: ₹{sig['entry_premium']:.2f} {prem_type}",
        f" Target       : ₹{sig['target_premium']:.2f} (+{sig['target_premium'] - sig['entry_premium']:.2f} = +{tgt_pct:.0f}%)",
        f" Stop Loss    : ₹{sig['sl_premium']:.2f}  (−{sig['entry_premium'] - sig['sl_premium']:.2f} = −{sl_pct:.0f}%)",
        f" Trail Rule   : At ₹{sig['trail_trigger']:.2f} (+{trail_pct:.0f}%) → SL to cost (₹{sig['entry_premium']:.2f})",
        f"              : At ₹{sig['lock_trigger']:.2f} (+{lock_pct:.0f}%) → lock +{lock_profit:.0f}% (SL ₹{sig['locked_sl']:.2f})",
        f" Gain/Lot     : ₹{sig['gain_per_lot']:,.0f}  |  Loss/Lot: ₹{sig['loss_per_lot']:,.0f}  |  R:R {sig['rr_ratio']:.1f}:1",
        f" Hard Cutoff  : {sig['hard_cutoff']}",
        "─" * 55,
        f" Groww Account: Cash ₹{acc.get('balance', 0):,.2f} | Allowed: {acc.get('lots', 1)} Lot(s) ({acc.get('source', 'Offline')})",
        f" Regime       : {sig['regime']}  |  India VIX: {sig.get('vix', 'N/A')}",
        f" VWAP Gate    : {'✅ Confirmed' if sig['vwap_confirmed'] else '❌ NOT confirmed'} ({sig.get('vwap_method', 'N/A')})",
        f" PCR OI       : {sig['pcr_oi']}  |  IV Skew: {sig['iv_skew']}",
        f" OI Modifier  : {sig['oi_adjustment']:+.1f}%  — {sig['oi_description']}",
        f" Base → Final : {sig['base_conviction']:.1f}% → {sig['final_conviction']:.1f}% (need ≥{sig['required_threshold']:.0f}%)",
        "─" * 55,
        f" {sig['signal_emoji']} ACTION: {sig['action']}",
        "═" * 55,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from utils.nifty_sensex_model import train_and_predict
    print("Training model and generating live signal for NIFTY...")
    m_res = train_and_predict(NIFTY_TICKER, period="5y")
    sig = generate_trade_signal(m_res, ticker=NIFTY_TICKER, fetch_live_oi=True, check_vwap=True)
    print(format_signal_console(sig))
