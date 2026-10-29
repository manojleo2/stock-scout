"""
utils/nifty_confluence.py
─────────────────────────
Multi-Factor Confluence & Real-Time Scanning Engine for NIFTY & SENSEX Options.

Eliminates "Single-Shot Analysis Paralysis":
  1. Multi-Factor Confluence Scoring (0 to 100 points):
     - Factor 1: ML Model Direction & Edge        (Max 30 pts)
     - Factor 2: Intraday VWAP & Price Alignment  (Max 25 pts)
     - Factor 3: Live PCR & Options Chain Flow   (Max 20 pts)
     - Factor 4: Pre-Market & Global Macro Bias   (Max 15 pts)
     - Factor 5: Cross-Index (Nifty/Sensex) Sync  (Max 10 pts)

  2. Dynamic Regime Thresholds:
     - TRENDING  : 60.0 pts
     - CHOP      : 65.0 pts (Replaces the rigid 80% static wall)
     - VOLATILE  : 65.0 pts
     - FRIDAY    : 65.0 pts
     - EXPIRY    : 68.0 pts

  3. Continuous Full-Day Scanning Window (09:18 AM – 02:30 PM IST):
     - Pre-market Briefing: 09:00 – 09:15 AM
     - Opening Window      : 09:18 – 11:00 AM
     - Mid-day Scanner     : 11:00 AM – 01:00 PM (Filtered)
     - Afternoon Window    : 01:00 PM – 02:30 PM (European open / trend expansion)
     - Cutoff Exit         : 03:05 PM (Hard square-off)
"""

import logging
from datetime import datetime, time as dtime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple, List

IST = timezone(timedelta(hours=5, minutes=30))

logger = logging.getLogger("nifty_confluence")

# ── Dynamic Thresholds ────────────────────────────────────────────────────────
CONFLUENCE_THRESHOLD_TRENDING = 60.0
CONFLUENCE_THRESHOLD_CHOP     = 65.0
CONFLUENCE_THRESHOLD_VOLATILE = 65.0
CONFLUENCE_THRESHOLD_FRIDAY   = 65.0
CONFLUENCE_THRESHOLD_EXPIRY   = 68.0

# ── Daily Scanner Windows ─────────────────────────────────────────────────────
TIME_PRE_MARKET_START = dtime(9, 0)
TIME_PRE_MARKET_END   = dtime(9, 15)
TIME_MARKET_OPEN      = dtime(9, 15)
TIME_SCANNER_START    = dtime(9, 18)
TIME_LUNCH_START      = dtime(11, 30)
TIME_LUNCH_END        = dtime(12, 45)
TIME_SCANNER_END      = dtime(14, 30)
TIME_HARD_CUTOFF      = dtime(15, 5)
TIME_MARKET_CLOSE     = dtime(15, 30)


def get_market_phase(now_time: Optional[dtime] = None) -> Dict[str, Any]:
    """
    Returns current active market phase for Nifty/Sensex intraday operations.
    """
    if now_time is None:
        now_time = datetime.now(IST).time()

    if now_time < TIME_PRE_MARKET_START:
        return {
            "phase": "PRE_MARKET_IDLE",
            "badge": "🌙 Market Offline (Pre-Open at 09:00 AM)",
            "can_scan": False,
            "description": "Waiting for 09:00 AM IST pre-open cues.",
        }
    elif TIME_PRE_MARKET_START <= now_time < TIME_PRE_MARKET_END:
        return {
            "phase": "PRE_MARKET_ACTIVE",
            "badge": "🌅 Pre-Market Cues Gathering (09:00 – 09:15 AM)",
            "can_scan": False,
            "description": "Analyzing GIFT Nifty, US closes, and Asian pre-open settles.",
        }
    elif TIME_PRE_MARKET_END <= now_time < TIME_SCANNER_START:
        return {
            "phase": "OPENING_DISCOVERY",
            "badge": "⏳ Opening Discovery (09:15 – 09:18 AM)",
            "can_scan": False,
            "description": "Allowing opening 3-minute volatility to settle before VWAP locks.",
        }
    elif TIME_SCANNER_START <= now_time <= TIME_SCANNER_END:
        is_lunch = (TIME_LUNCH_START <= now_time <= TIME_LUNCH_END)
        return {
            "phase": "ACTIVE_SCANNING",
            "badge": "⚡ Continuous Scanner Active (09:18 – 14:30 IST)" if not is_lunch else "🥪 Mid-Day Scanner (High-Edge Only)",
            "can_scan": True,
            "is_lunch_window": is_lunch,
            "description": "Continuously evaluating 5-second ticks for high-confluence entry.",
        }
    elif TIME_SCANNER_END < now_time < TIME_HARD_CUTOFF:
        return {
            "phase": "LATE_DAY_HOLD",
            "badge": "⏸️ Scanner Closed (Holding / Managing Open Positions)",
            "can_scan": False,
            "description": "No new entries permitted after 02:30 PM. Managing open trailing stops.",
        }
    elif TIME_HARD_CUTOFF <= now_time < TIME_MARKET_CLOSE:
        return {
            "phase": "HARD_CUTOFF",
            "badge": "🛑 03:05 PM Hard Cutoff Zone (All Risk Squared Off)",
            "can_scan": False,
            "description": "Mandatory square-off window to prevent overnight gap risk.",
        }
    else:
        return {
            "phase": "POST_MARKET",
            "badge": "🌙 Market Closed (EOD Self-Learning & Ledger Finalized)",
            "can_scan": False,
            "description": "Market session closed for today.",
        }


def compute_confluence_score(
    model_result: Dict[str, Any],
    vwap_data: Dict[str, Any],
    oi_summary: Dict[str, Any],
    overnight_cues: Optional[Dict[str, Any]] = None,
    other_index_result: Optional[Dict[str, Any]] = None,
    current_time: Optional[dtime] = None,
) -> Dict[str, Any]:
    """
    Computes institutional Multi-Factor Confluence Score (0 to 100 points).
    Replaces static single-gate failure with balanced multi-pillar confirmation.
    """
    if current_time is None:
        current_time = datetime.now(IST).time()

    direction = model_result.get("direction", "NEUTRAL")
    conviction = float(model_result.get("conviction_pct", 50.0))
    regime = model_result.get("regime", "CHOP")
    is_friday = bool(model_result.get("is_friday", False))
    is_expiry_day = bool(model_result.get("is_expiry_day", False))

    factors_breakdown: List[Dict[str, Any]] = []

    # ── Factor 1: ML Model Edge (Max 30 pts) ───────────────────────────────────
    ml_pts = 0.0
    if model_result.get("status") == "success":
        # Base conviction: 50% = 0 pts, 60% = 12 pts, 70% = 24 pts, 75%+ = 30 pts
        edge = max(0.0, conviction - 50.0)
        ml_pts = min(25.0, edge * 1.2)
        confidence = model_result.get("confidence", "Low / Neutral")
        if confidence == "Super High":
            ml_pts += 5.0
        elif confidence == "High":
            ml_pts += 3.0
        elif confidence == "Moderate":
            ml_pts += 1.0
        ml_pts = min(30.0, round(ml_pts, 1))

    factors_breakdown.append({
        "factor": "ML Model Edge",
        "score": ml_pts,
        "max": 30.0,
        "detail": f"{direction} ({conviction:.1f}% conviction, {model_result.get('confidence', 'Neutral')})",
    })

    # ── Factor 2: Intraday VWAP & Price Alignment (Max 25 pts) ────────────────
    vwap_pts = 0.0
    vwap_val = float(vwap_data.get("vwap", 0.0) or 0.0)
    spot = float(oi_summary.get("spot", 0.0) or vwap_data.get("spot", 0.0) or model_result.get("latest_close", 0.0))
    vwap_confirmed = False

    if vwap_val > 0 and spot > 0:
        pct_dist = abs(spot - vwap_val) / vwap_val * 100.0
        if direction == "UP" and spot >= vwap_val:
            vwap_confirmed = True
            vwap_pts += 15.0
            if 0.02 <= pct_dist <= 0.35:
                vwap_pts += 10.0  # Sweet spot: breakout with room to run
            elif pct_dist < 0.02:
                vwap_pts += 7.0   # Right on the line
            elif pct_dist <= 0.50:
                vwap_pts += 4.0   # Slightly extended
        elif direction == "DOWN" and spot <= vwap_val:
            vwap_confirmed = True
            vwap_pts += 15.0
            if 0.02 <= pct_dist <= 0.35:
                vwap_pts += 10.0
            elif pct_dist < 0.02:
                vwap_pts += 7.0
            elif pct_dist <= 0.50:
                vwap_pts += 4.0

    factors_breakdown.append({
        "factor": "VWAP Alignment",
        "score": vwap_pts,
        "max": 25.0,
        "detail": f"{'✅ Confirmed' if vwap_confirmed else '❌ Failed'} (Spot: {spot:.1f}, VWAP: {vwap_val:.1f})",
    })

    # ── Factor 3: Live PCR & Options Flow (Max 20 pts) ────────────────────────
    pcr_pts = 0.0
    pcr_oi = float(oi_summary.get("pcr_oi", 0.0) or 0.0)
    if pcr_oi > 0:
        if direction == "UP":
            if pcr_oi <= 0.80:
                pcr_pts += 14.0  # Oversold / strong call absorption
            elif pcr_oi <= 1.00:
                pcr_pts += 10.0
            elif pcr_oi <= 1.20:
                pcr_pts += 6.0
            else:
                pcr_pts += 2.0   # Overbought resistance
        else:  # DOWN
            if pcr_oi >= 1.25:
                pcr_pts += 14.0  # Heavy call writing / bearish dominance
            elif pcr_oi >= 1.00:
                pcr_pts += 10.0
            elif pcr_oi >= 0.80:
                pcr_pts += 6.0
            else:
                pcr_pts += 2.0

        # Live option chain quote bonus (real LTP available)
        if oi_summary.get("atm_ce_ltp") and oi_summary.get("atm_pe_ltp"):
            pcr_pts += 6.0
    else:
        # Fallback neutral score if PCR data offline
        pcr_pts = 10.0

    pcr_pts = min(20.0, pcr_pts)
    factors_breakdown.append({
        "factor": "PCR & Options Flow",
        "score": pcr_pts,
        "max": 20.0,
        "detail": f"PCR OI: {pcr_oi:.2f}" if pcr_oi > 0 else "PCR Offline (Neutral)",
    })

    # ── Factor 4: Pre-Market & Global Macro Bias (Max 15 pts) ─────────────────
    macro_pts = 8.0  # Default neutral
    if overnight_cues:
        sp_chg = 0.0
        nq_chg = 0.0
        for mkt in overnight_cues.get("global_markets", []):
            label = mkt.get("label", "").lower()
            chg = float(mkt.get("change", 0.0) or 0.0)
            if "s&p" in label:
                sp_chg = chg
            elif "nasdaq" in label:
                nq_chg = chg

        avg_us = (sp_chg + nq_chg) / 2.0
        gap_pct = float(overnight_cues.get("sgx_gap_pct", 0.0) or 0.0)

        if direction == "UP":
            if avg_us > 0.3 and gap_pct >= 0.0:
                macro_pts = 15.0
            elif avg_us > 0.0 or gap_pct > 0.0:
                macro_pts = 11.0
            elif avg_us < -0.5:
                macro_pts = 3.0
            else:
                macro_pts = 7.0
        else:  # DOWN
            if avg_us < -0.3 and gap_pct <= 0.0:
                macro_pts = 15.0
            elif avg_us < 0.0 or gap_pct < 0.0:
                macro_pts = 11.0
            elif avg_us > 0.5:
                macro_pts = 3.0
            else:
                macro_pts = 7.0

    factors_breakdown.append({
        "factor": "Pre-Market & Macro Cues",
        "score": macro_pts,
        "max": 15.0,
        "detail": f"Global US: {avg_us:+.2f}%, SGX Gap: {gap_pct:+.2f}%" if overnight_cues else "Neutral/Historical Base",
    })

    # ── Factor 5: Cross-Index Sync (Max 10 pts) ──────────────────────────────
    sync_pts = 5.0
    if other_index_result and other_index_result.get("status") == "success":
        other_dir = other_index_result.get("direction")
        if other_dir == direction:
            sync_pts = 10.0
        else:
            sync_pts = 2.0

    factors_breakdown.append({
        "factor": "Cross-Index Sync",
        "score": sync_pts,
        "max": 10.0,
        "detail": f"{'Aligned' if sync_pts == 10.0 else 'Divergent/Solo'}",
    })

    # ── Total Confluence & Threshold Determination ────────────────────────────
    total_confluence = round(sum(f["score"] for f in factors_breakdown), 1)
    total_confluence = float(min(100.0, max(0.0, total_confluence)))

    # Determine dynamic threshold
    if is_expiry_day:
        threshold = CONFLUENCE_THRESHOLD_EXPIRY
    elif is_friday:
        threshold = CONFLUENCE_THRESHOLD_FRIDAY
    elif regime == "VOLATILE":
        threshold = CONFLUENCE_THRESHOLD_VOLATILE
    elif regime == "CHOP":
        threshold = CONFLUENCE_THRESHOLD_CHOP
    else:
        threshold = CONFLUENCE_THRESHOLD_TRENDING

    # Lunch hours discount penalty if not strong enough
    market_phase = get_market_phase(current_time)
    if market_phase.get("is_lunch_window", False):
        threshold += 5.0  # Lunch lull requires extra edge

    is_passed = (total_confluence >= threshold) and vwap_confirmed

    return {
        "total_confluence": total_confluence,
        "threshold": threshold,
        "is_passed": is_passed,
        "vwap_confirmed": vwap_confirmed,
        "regime": regime,
        "direction": direction,
        "factors": factors_breakdown,
        "market_phase": market_phase,
    }


def evaluate_early_defense_exit(
    active_position: Dict[str, Any],
    current_spot: float,
    current_option_ltp: float,
    current_vwap: float,
) -> Tuple[bool, str]:
    """
    Early Defense Engine: Saves capital by cutting losing trades before full -15% SL.
    
    Conditions to trigger Early Defense:
      1. Adverse VWAP breakdown: Spot entered above VWAP, but now closed firmly on wrong side.
      2. Unwinding momentum: Option premium has lost ≥ 7.0% AND spot violated VWAP.
    """
    if not active_position or current_option_ltp <= 0:
        return False, ""

    entry_premium = active_position.get("entry_premium", 0.0)
    direction = active_position.get("direction", "UP")

    if entry_premium <= 0:
        return False, ""

    pnl_pct = (current_option_ltp - entry_premium) / entry_premium * 100.0

    # Rule 1: Pre-emptive cut if premium drops to -7% AND VWAP is lost
    if pnl_pct <= -7.0 and current_vwap > 0:
        if direction == "UP" and current_spot < current_vwap:
            reason = f"🛡️ Early Defense Triggered: CALL lost VWAP ({current_spot:.1f} < {current_vwap:.1f}) & premium down {pnl_pct:.1f}%. Cut early to prevent -15% SL."
            return True, reason
        elif direction == "DOWN" and current_spot > current_vwap:
            reason = f"🛡️ Early Defense Triggered: PUT lost VWAP ({current_spot:.1f} > {current_vwap:.1f}) & premium down {pnl_pct:.1f}%. Cut early to prevent -15% SL."
            return True, reason

    # Rule 2: Hard momentum invalidation at -10% if 0.25% beyond VWAP
    if pnl_pct <= -10.0 and current_vwap > 0:
        if (direction == "UP" and current_spot < current_vwap) or (direction == "DOWN" and current_spot > current_vwap):
            reason = f"🛡️ Momentum Invalidation: Option down {pnl_pct:.1f}% with adverse VWAP. Capital protected from full SL."
            return True, reason

    return False, ""
