"""
views/nifty_predictor_view.py
──────────────────────────────
Streamlit UI for the Nifty & Sensex Intraday Options Predictor.
Lean Trading Terminal — Option Premium First.

Architecture:
  1. Top Bar: Index Selector, Capital Discipline Rule (1 Lot), Sentinel Live Status
  2. Hero Option Trading Card:
     - Prominent Action Banner (BUY CALL / BUY PUT / WAIT VWAP / SCANNING)
     - 4 Option Premium Metrics: Contract & LTP, Target (+30%), SL (-15%), Trailing SL (Cost & Lock)
     - 1-Line Trade Rationale (Spot vs VWAP, PCR, Confluence, Cutoff 03:05 PM)
     - Real-Time P&L Tracking (when Sentinel trade is active)
  3. Collapsible Diagnostics:
     - 🌙 Pre-Market Briefing & Global Cues
     - 📊 Live OI & Strike Chain Analysis
     - 🤖 Institutional Ensemble & Confluence Diagnostics
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime, timezone, timedelta, time as dt_time

IST = timezone(timedelta(hours=5, minutes=30))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.nifty_overnight_cues import get_overnight_cues
from utils.nifty_sensex_model import train_and_predict
from utils.nifty_trade_signal import generate_trade_signal, format_signal_console
from utils.daily_journal import render_daily_journal_section
from config import NIFTY_TICKER, SENSEX_TICKER


# ─────────────────────────────────────────────────────────────────────────────
# Cached wrappers (Sub-second disk cache fallback)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _cached_overnight_cues():
    return get_overnight_cues()


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_model(ticker):
    return train_and_predict(ticker=ticker, period="3y")


def _get_sentinel_state():
    state_file = os.path.join(PROJECT_ROOT, "live_sentinel_state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Render: Hero Option Trading Card (Option Premium First)
# ─────────────────────────────────────────────────────────────────────────────

def _render_hero_trading_card(sig: dict, sentinel_state: dict = None):
    signal = sig.get("signal", "NO_SIGNAL")
    contract = sig.get("contract", "N/A")
    entry_prem = sig.get("entry_premium", 0.0)
    target_prem = sig.get("target_premium", 0.0)
    sl_prem = sig.get("sl_premium", 0.0)
    gain_lot = sig.get("gain_per_lot", 0)
    loss_lot = sig.get("loss_per_lot", 0)
    diff_tgt = target_prem - entry_prem
    diff_sl = entry_prem - sl_prem
    trail_trig = sig.get("trail_trigger", 0.0)
    lock_trig = sig.get("lock_trigger", 0.0)
    lot_size = sig.get("lot_size", 65)
    direction = sig.get("direction", "UP")
    spot = sig.get("spot", 0.0)
    vwap_val = sig.get("vwap", 0.0)
    vwap_str = f"₹{vwap_val:,.2f}" if vwap_val else "N/A"
    conf_data = sig.get("confluence", {})
    conf_score = conf_data.get("total_confluence", sig.get("confluence_score", 0.0))
    conf_thresh = conf_data.get("threshold", sig.get("confluence_thresh", 60.0))
    pcr_val = sig.get("pcr_oi", "N/A")

    # 1. Action Header Banner
    if signal == "BUY_CE":
        st.success(f"## 🟢 BUY CALL: {contract}")
    elif signal == "BUY_PE":
        st.error(f"## 🔴 BUY PUT: {contract}")
    elif signal == "WAIT_VWAP":
        st.warning(f"## ⏳ WAIT — VWAP Not Confirmed ({contract})")
    else:
        st.info(f"## ⚪ SCANNING — Preserving Capital ({contract})")

    # 2. Check if there's an active trade in Sentinel
    active_pos = sentinel_state.get("active_position") if sentinel_state else None
    if active_pos:
        pnl = active_pos.get("pnl_rs", 0.0)
        pnl_pct = active_pos.get("pnl_pct", 0.0)
        pnl_badge = f"+₹{pnl:,.0f} (+{pnl_pct:.1f}%)" if pnl >= 0 else f"-₹{abs(pnl):,.0f} ({pnl_pct:.1f}%)"
        alert_fn = st.success if pnl >= 0 else st.error
        alert_fn(
            f"🎯 **ACTIVE TRADE RUNNING:** {active_pos.get('contract')} | "
            f"Entry: ₹{active_pos.get('entry_premium', 0):.2f} | Current LTP: ₹{active_pos.get('current_ltp', 0):.2f} | "
            f"Live P&L: **{pnl_badge}** | "
            f"Trailing SL: ₹{active_pos.get('current_sl', 0):.2f} ({active_pos.get('trail_state')}) | "
            f"Target: ₹{active_pos.get('target_premium', 0):.2f}"
        )

    # 3. Four Core Metric Columns (Option Premium First)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label=f"🎯 {contract} ({'Live LTP' if sig.get('is_live_premium') else 'Est. Premium'})",
            value=f"₹{entry_prem:.2f}",
            delta=f"ATM Strike: {sig.get('atm_strike', '-')}",
            delta_color="off",
        )
    with c2:
        st.metric(
            label="Target (+30%)",
            value=f"₹{target_prem:.2f}",
            delta=f"+₹{diff_tgt:.2f} (+₹{gain_lot:,.0f} / lot)",
            delta_color="normal",
        )
    with c3:
        st.metric(
            label="Stop Loss (-15%)",
            value=f"₹{sl_prem:.2f}",
            delta=f"−₹{diff_sl:.2f} (−₹{loss_lot:,.0f} / lot)",
            delta_color="inverse",
        )
    with c4:
        st.metric(
            label="Trailing Protection",
            value=f"Trail @ ₹{trail_trig:.2f}",
            delta=f"Lock +15% @ ₹{lock_trig:.2f}",
            delta_color="off",
        )

    # 4. 1-Line Trade Rationale Box
    direction_word = "ABOVE" if direction == "UP" else "BELOW"
    vwap_gate_status = "✅ Confirmed" if sig.get("vwap_confirmed") else "❌ Unconfirmed"
    st.info(
        f"📌 **Trade Rationale:** Spot (₹{spot:,.2f}) trades {direction_word} VWAP ({vwap_str} • {vwap_gate_status}) | "
        f"PCR: **{pcr_val}** | Confluence: **{conf_score:.1f}/{conf_thresh:.0f} pts** | "
        f"Discipline: **1 Lot ({lot_size} Qty)** | Hard Cutoff: **03:05 PM**"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Render: Collapsible Supporting Sections
# ─────────────────────────────────────────────────────────────────────────────

def _render_premarket_expander(cues: dict):
    with st.expander("🌙 Pre-Market Briefing & Global Cues", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            chg = cues.get("nifty_chg_pct", 0.0)
            st.metric("Nifty 50", cues.get("nifty_last", "N/A"),
                      f"{chg:+.2f}%", delta_color="normal")
        with col2:
            sgx = cues.get("sgx_gap_pct", 0.0)
            color = "🟢" if sgx > 0 else ("🔴" if sgx < 0 else "⚪")
            st.metric("SGX Gap", f"{color} {sgx:+.3f}%",
                      f"Est: {cues.get('sgx_nifty_est', 'N/A')}")
        with col3:
            st.metric(f"India VIX {cues.get('vix_badge','')}", cues.get("vix", "N/A"),
                      f"{cues.get('vix_chg_pct', 0.0):+.2f}%",
                      delta_color="inverse")
        with col4:
            bn = cues.get("banknifty_chg_pct", 0.0)
            st.metric("BankNifty", f"{bn:+.2f}%", delta_color="normal")

        global_mkts = cues.get("global_markets", [])
        if global_mkts:
            cols = st.columns(len(global_mkts))
            for idx, mkt in enumerate(global_mkts):
                with cols[idx]:
                    st.metric(
                        f"{mkt['emoji']} {mkt['label']}",
                        mkt.get("last", "N/A"),
                        f"{mkt.get('change', 0.0):+.2f}%",
                        delta_color="normal",
                    )
        st.caption(f"**Overnight Sentiment:** {cues.get('overnight_sentiment', 'NEUTRAL ⚪')} "
                   f"(Avg global: {cues.get('avg_global_bias_pct', 0.0):+.3f}%)")


def _render_oi_expander(sig: dict):
    with st.expander("📊 Live OI & Strike Chain Analysis", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            pcr = sig.get("pcr_oi", "N/A")
            pcr_v = float(pcr) if isinstance(pcr, (float, int)) else None
            color = ("🔴 Bearish" if pcr_v and pcr_v > 1.30 else
                     "🟢 Bullish" if pcr_v and pcr_v < 0.75 else "🟡 Neutral")
            st.metric("PCR OI", f"{pcr}", color)
        with col2:
            st.metric("PCR Volume", sig.get("pcr_vol", "N/A"))
        with col3:
            st.metric("IV Skew", sig.get("iv_skew", "N/A"))
        with col4:
            adj = sig.get("oi_adjustment", 0.0)
            st.metric("OI Modifier", f"{adj:+.1f}%",
                      f"{sig.get('base_conviction', 0):.1f}% → {sig.get('final_conviction', 0):.1f}%")

        # ATM Strike Comparison
        atm_s = sig.get("atm_strike", "N/A")
        ce_ltp = sig.get("atm_ce_ltp", 0.0)
        pe_ltp = sig.get("atm_pe_ltp", 0.0)
        c_ce, c_pe = st.columns(2)
        with c_ce:
            is_sel_ce = (sig.get("option_type") == "CE")
            st.metric(
                f"{'🟢 ' if is_sel_ce else ''}{sig.get('ticker_clean')} {atm_s} CE",
                f"₹{ce_ltp:.2f}" if ce_ltp > 0 else "Offline",
                "🎯 Selected Trade Contract" if is_sel_ce else "Alternative Strike",
            )
        with c_pe:
            is_sel_pe = (sig.get("option_type") == "PE")
            st.metric(
                f"{'🔴 ' if is_sel_pe else ''}{sig.get('ticker_clean')} {atm_s} PE",
                f"₹{pe_ltp:.2f}" if pe_ltp > 0 else "Offline",
                "🎯 Selected Trade Contract" if is_sel_pe else "Alternative Strike",
            )


def _render_diagnostics_expander(model_result: dict, sig: dict):
    with st.expander("🤖 Institutional Ensemble & Confluence Diagnostics", expanded=False):
        metrics = model_result.get("test_metrics", {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Accuracy", f"{metrics.get('accuracy', 0):.1f}%")
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.1f}%")
        with col3:
            st.metric("Brier Score", f"{metrics.get('brier', 0):.4f}")
        with col4:
            st.metric("Models Used", metrics.get("n_models", 4))

        regime = model_result.get("regime", "—")
        rd = model_result.get("regime_detail", "")
        badge = "🟢" if regime == "TRENDING" else ("🟡" if regime == "VOLATILE" else "🔴")
        st.info(f"{badge} **Regime: {regime}** — {rd}")

        # Multi-factor confluence breakdown
        conf_data = sig.get("confluence", {})
        if conf_data and "factors" in conf_data:
            st.markdown("##### 🏛️ Confluence Factor Breakdown")
            c_f1, c_f2 = st.columns(2)
            factors = conf_data.get("factors", [])
            for idx, f in enumerate(factors):
                target_col = c_f1 if idx % 2 == 0 else c_f2
                with target_col:
                    st.markdown(f"**{f['factor']}**: `{f['score']:.1f} / {f['max']:.0f} pts` ({f['detail']})")

        render_daily_journal_section(model_result)

        with st.expander("📋 Full Console Signal Output", expanded=False):
            st.code(format_signal_console(sig), language="text")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry (called from app.py)
# ─────────────────────────────────────────────────────────────────────────────

def render_nifty_predictor_tab():
    """Main render function — lean options terminal."""
    # ── Top Bar ───────────────────────────────────────────────────────────────
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.title("🎯 Nifty & Sensex Options Terminal")
        st.caption("1-Lot Capital Safety Engine | Dynamic Target (+30%) & SL (-15%) | Live VWAP Gated")
    with col_t2:
        now_str = datetime.now(IST).strftime("%I:%M:%S %p IST")
        st.markdown(f"<div style='text-align: right; padding-top: 15px; color: gray;'>🕒 {now_str}</div>", unsafe_allow_html=True)

    # ── Broker Status & Capital Discipline Ribbon ──────────────────────────────
    sentinel_state = _get_sentinel_state()
    cash = 24300.0
    lots = 1
    if sentinel_state and sentinel_state.get("cash_margin"):
        cash = sentinel_state.get("cash_margin")
        lots = sentinel_state.get("allowed_lots", 1)

    st.success(
        f"⚡ **Groww Live Broker Active** | UCC: `9338012620` | "
        f"Available Margin: **₹{cash:,.2f}** | "
        f"Enforced Lot Size: **{lots} Lot(s)** (Capital Safety Rule Enforced: ≤₹30k = 1 Lot)"
    )

    # ── Index Selector & Refresh ──────────────────────────────────────────────
    ticker_opts = {"Nifty 50 (^NSEI)": NIFTY_TICKER, "Sensex (^BSESN)": SENSEX_TICKER}
    c_sel, c_btn = st.columns([4, 1])
    with c_sel:
        sel = st.selectbox("Select Index", list(ticker_opts.keys()), label_visibility="collapsed")
    with c_btn:
        run_btn = st.button("🔄 Refresh Signal", use_container_width=True)

    ticker = ticker_opts[sel]
    if run_btn:
        st.cache_data.clear()

    # ── Pre-market cues (cached, sub-second) ───────────────────────────────────
    cues = _cached_overnight_cues()

    # ── Model Prediction (sub-second disk cache fallback) ──────────────────────
    model_result = _cached_model(ticker)
    if model_result.get("status") != "success":
        st.error(f"Model error: {model_result.get('message', 'Unknown')}")
        return

    # Market auto-refresh window: 09:00 AM to 03:00 PM (15:00) IST on weekdays
    now_ist = datetime.now(IST)
    is_weekday = (now_ist.weekday() < 5)
    is_live_window = is_weekday and (dt_time(9, 0) <= now_ist.time() <= dt_time(15, 0))
    auto_refresh_rate = "5s" if is_live_window else None

    @st.fragment(run_every=auto_refresh_rate)
    def _render_live_stream_fragment(ticker_sym: str, model_data: dict, overnight_cues_data: dict):
        current_sentinel = _get_sentinel_state()

        # Generate live trade signal
        try:
            live_sig = generate_trade_signal(
                model_data,
                ticker=ticker_sym,
                fetch_live_oi=True,
                check_vwap=True,
                overnight_cues=overnight_cues_data,
            )
        except Exception as e:
            st.error(f"Signal generation failed: {e}")
            return

        # ── 1. Hero Option Trading Card ───────────────────────────────────────
        _render_hero_trading_card(live_sig, current_sentinel)
        st.divider()

        # ── 2. Collapsible Diagnostics ────────────────────────────────────────
        _render_premarket_expander(overnight_cues_data)
        _render_oi_expander(live_sig)
        _render_diagnostics_expander(model_data, live_sig)

    # Execute continuous live stream fragment
    _render_live_stream_fragment(ticker, model_result, cues)
