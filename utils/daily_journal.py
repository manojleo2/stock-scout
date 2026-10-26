"""
utils/daily_journal.py
──────────────────────
Daily Trade & Decision Journal for Nifty Options Sentinel.
Tracks every trading session:
  - Date and Day of Week
  - Decision: TRADE EXECUTED vs CAPITAL PRESERVED (NO TRADE)
  - Detailed Post-Mortem Narrative: Why the trade was taken OR why it was skipped
  - Conviction, Regime, VWAP gate status, and P&L
"""

import os
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import streamlit as st

IST = timezone(timedelta(hours=5, minutes=30))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOURNAL_FILE = os.path.join(PROJECT_ROOT, "daily_decision_journal.json")


def load_daily_journal() -> List[Dict[str, Any]]:
    """Loads all recorded daily trading session summaries."""
    if os.path.exists(JOURNAL_FILE):
        try:
            with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            return []
    return []


def save_daily_journal(records: List[Dict[str, Any]]) -> bool:
    """Atomic write of daily decision journal."""
    try:
        temp_file = f"{JOURNAL_FILE}.tmp_{os.getpid()}"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, JOURNAL_FILE)
        return True
    except Exception:
        return False


def get_today_entry(model_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generates today's dynamic pre-market or live decision summary."""
    now_ist = datetime.now(IST)
    today_str = now_ist.strftime("%Y-%m-%d")
    weekday_name = now_ist.strftime("%A")

    if model_result and model_result.get("status") == "success":
        direction = model_result.get("direction", "NEUTRAL")
        conviction = model_result.get("conviction_pct", 50.0)
        threshold = model_result.get("required_threshold", 62.0)
        regime = model_result.get("regime", "CHOP")
        is_tradeable = model_result.get("is_tradeable", False)
        spot = model_result.get("latest_close", 0.0)

        if is_tradeable:
            decision = "STANDBY_FOR_ENTRY"
            badge = "⏳ Standby for 09:20 VWAP Confirmation"
            summary = (
                f"Model forecast is {direction} with {conviction:.1f}% conviction (meets ≥{threshold:.1f}% threshold). "
                f"Sentinel is standing by for 09:18-09:20 AM VWAP gate confirmation before entry."
            )
            why = f"High conviction ({conviction:.1f}%) meets {regime} threshold ({threshold:.1f}%)."
        else:
            decision = "NO_TRADE"
            badge = "🛡️ Capital Preserved (No Trade)"
            summary = (
                f"Today's ML model directional forecast is {direction} with {conviction:.1f}% conviction. "
                f"The required threshold for {regime} regime is ≥{threshold:.1f}%. "
                f"Because conviction is sub-threshold, the entry gatekeeper strictly blocks trade execution "
                f"to preserve capital and prevent whipsaws in low-edge consolidation."
            )
            why = f"Model conviction ({conviction:.1f}%) is below the required {regime} cutoff (≥{threshold:.1f}%)."

        return {
            "date": today_str,
            "day_of_week": weekday_name,
            "ticker": "^NSEI",
            "ticker_name": "NIFTY 50",
            "contract": f"NIFTY {round(spot / 50) * 50} {'CE' if direction == 'UP' else 'PE'}",
            "decision": decision,
            "decision_badge": badge,
            "direction": f"{direction} {'📈' if direction == 'UP' else '📉'}",
            "conviction_pct": conviction,
            "required_threshold_pct": threshold,
            "regime": regime,
            "vwap_status": "MONITORING",
            "pcr_oi": 1.0,
            "pnl_rs": 0.0,
            "pnl_points": 0.0,
            "capital_status": "100% Cash Protected (₹0 Loss)",
            "summary": summary,
            "why_no_trade": why,
            "what_was_expected": f"Expected {direction} bias if volume/conviction confirms.",
            "what_had_happened": "Session in progress. Capital protected in 100% cash buffer."
        }

    return {
        "date": today_str,
        "day_of_week": weekday_name,
        "ticker": "^NSEI",
        "ticker_name": "NIFTY 50",
        "contract": "NIFTY Cash Buffer",
        "decision": "STANDBY",
        "decision_badge": "⏳ Pre-Market Standby",
        "direction": "NEUTRAL",
        "conviction_pct": 0.0,
        "required_threshold_pct": 62.0,
        "regime": "STANDBY",
        "vwap_status": "UNAVAILABLE",
        "pcr_oi": 1.0,
        "pnl_rs": 0.0,
        "pnl_points": 0.0,
        "capital_status": "100% Cash Protected",
        "summary": "Pre-market evaluation in progress. Awaiting market open.",
        "why_no_trade": "Market not open yet.",
        "what_was_expected": "Awaiting morning session data.",
        "what_had_happened": "Pre-market standby."
    }


def render_daily_journal_section(model_result: Optional[Dict[str, Any]] = None):
    """Renders the comprehensive daily decision journal on the Streamlit dashboard."""
    st.markdown("## 📖 Daily Trade & Decision Journal")
    st.caption(
        "Complete historical record of every session: Why trades were taken, "
        "why trades were skipped, model conviction gates, and capital protection outcomes."
    )

    records = load_daily_journal()
    now_ist = datetime.now(IST)
    today_str = now_ist.strftime("%Y-%m-%d")

    # If today's session is not yet recorded, prepend dynamic today's entry
    has_today = any(r.get("date") == today_str for r in records)
    display_records = list(records)
    if not has_today:
        today_entry = get_today_entry(model_result)
        display_records.insert(0, today_entry)

    # ── Summary KPI metrics ──
    total_days = len(display_records)
    trades_taken = sum(1 for r in display_records if r.get("decision") in ["TRADE_EXECUTED", "TRADE_TAKEN"])
    trades_skipped = sum(1 for r in display_records if r.get("decision") in ["NO_TRADE", "SKIPPED"])
    total_net_pnl = sum(r.get("pnl_rs", 0.0) for r in display_records)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Sessions Tracked", f"{total_days} Days")
    with k2:
        st.metric("Trades Executed", f"{trades_taken}", "High Conviction Only")
    with k3:
        st.metric("Capital Protected (Skipped)", f"{trades_skipped} Days", "100% Cash Discipline")
    with k4:
        pnl_color = "normal" if total_net_pnl >= 0 else "inverse"
        st.metric("Net Realized Gain", f"₹{total_net_pnl:+,.2f}", f"Cumulative P&L", delta_color=pnl_color)

    st.markdown("---")

    # ── Render Each Daily Session Card ──
    for rec in display_records:
        date_val = rec.get("date", "—")
        day_of_week = rec.get("day_of_week", "")
        badge = rec.get("decision_badge", "—")
        decision = rec.get("decision", "NO_TRADE")
        is_trade = (decision in ["TRADE_EXECUTED", "TRADE_TAKEN"])
        pnl_rs = rec.get("pnl_rs", 0.0)

        # Card header styling
        icon = "🎯" if is_trade else "🛡️"
        status_color = "green" if is_trade else "orange" if pnl_rs == 0 else "red"

        with st.container():
            col_date, col_badge, col_pnl = st.columns([2, 3, 2])
            with col_date:
                st.markdown(f"### {icon} **{date_val}** ({day_of_week})")
            with col_badge:
                if is_trade:
                    st.success(f"**{badge}**")
                else:
                    st.info(f"**{badge}**")
            with col_pnl:
                if is_trade:
                    st.metric("Realized P&L", f"+₹{pnl_rs:,.2f}", f"+{rec.get('pnl_points', 0):.2f} pts")
                else:
                    st.metric("Capital Impact", "₹0.00 (Protected)", "100% Cash Buffer")

            # Session Details Table / Breakdown
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"**Contract:** `{rec.get('contract', '—')}`")
            with c2:
                st.markdown(f"**Direction:** {rec.get('direction', '—')}")
            with c3:
                conv = rec.get('conviction_pct', 0.0)
                req = rec.get('required_threshold_pct', 62.0)
                st.markdown(f"**Conviction:** `{conv:.1f}%` (Need `≥{req:.0f}%`)")
            with c4:
                st.markdown(f"**Regime:** `{rec.get('regime', '—')}`")

            # Narrative Summary Box
            st.markdown(f"> **Daily Post-Mortem & Decision Summary:**\n> {rec.get('summary', '—')}")

            with st.expander(f"🔍 Detailed Audit Breakdown — {date_val}"):
                ca, cb = st.columns(2)
                with ca:
                    st.markdown(f"**Why Decision Was Made:**\n{rec.get('why_no_trade', '—')}")
                    st.markdown(f"**VWAP Status:** `{rec.get('vwap_status', '—')}` | **PCR OI:** `{rec.get('pcr_oi', '—')}`")
                with cb:
                    st.markdown(f"**What Was Expected:**\n{rec.get('what_was_expected', '—')}")
                    st.markdown(f"**What Actually Happened:**\n{rec.get('what_had_happened', '—')}")
                    st.markdown(f"**Capital Status:** `{rec.get('capital_status', '—')}`")

            st.divider()
