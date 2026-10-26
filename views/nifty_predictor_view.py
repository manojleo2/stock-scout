"""
views/nifty_predictor_view.py
──────────────────────────────
Streamlit UI for the Nifty & Sensex Intraday Options Predictor.

Sections:
  1. Pre-Market Briefing (overnight cues, VIX, SGX gap)
  2. Trade Signal Panel (CE/PE, entry, target, SL, VWAP gate)
  3. OI/PCR Live Panel (conviction modifier)
  4. Model Stats (test accuracy, regime, feature importance)
  5. Nifty vs Sensex cross-confirmation panel
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.nifty_overnight_cues import get_overnight_cues
from utils.nifty_sensex_model import train_and_predict
from utils.nifty_trade_signal import generate_trade_signal, format_signal_console
from utils.nifty_oi_pcr import get_live_nifty_oi_summary
from utils.daily_journal import render_daily_journal_section
from config import NIFTY_TICKER, SENSEX_TICKER


# ─────────────────────────────────────────────────────────────────────────────
# Cached wrappers (TTL 60s for signal, 5min for cues)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _cached_overnight_cues():
    return get_overnight_cues()


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_model(ticker):
    return train_and_predict(ticker=ticker, period="3y")


@st.cache_data(ttl=60, show_spinner=False)
def _cached_signal(ticker, model_json):
    import json
    model_result = json.loads(model_json)
    return generate_trade_signal(model_result, ticker=ticker,
                                  fetch_live_oi=True, check_vwap=True)


def _model_to_json(model_result: dict) -> str:
    import json
    safe = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else
                bool(v) if isinstance(v, np.bool_) else v)
            for k, v in model_result.items()
            if not isinstance(v, dict)}
    return json.dumps(safe)


# ─────────────────────────────────────────────────────────────────────────────
# Render: Pre-Market Briefing
# ─────────────────────────────────────────────────────────────────────────────

def _render_premarket(cues: dict):
    st.subheader("🌙 Pre-Market Briefing")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        chg = cues.get("nifty_chg_pct", 0.0)
        st.metric("Nifty 50", cues.get("nifty_last", "N/A"),
                  f"{chg:+.2f}%", delta_color="normal")
    with col2:
        sgx = cues.get("sgx_gap_pct", 0.0)
        color = "🟢" if sgx > 0 else ("🔴" if sgx < 0 else "⚪")
        st.metric("SGX Gap (NIFTYBEES proxy)", f"{color} {sgx:+.3f}%",
                  f"Est Nifty: {cues.get('sgx_nifty_est', 'N/A')}")
    with col3:
        st.metric(f"India VIX  {cues.get('vix_badge','')}", cues.get("vix", "N/A"),
                  f"{cues.get('vix_chg_pct', 0.0):+.2f}%",
                  delta_color="inverse")
    with col4:
        bn = cues.get("banknifty_chg_pct", 0.0)
        st.metric("BankNifty", f"{bn:+.2f}%", delta_color="normal")

    with st.expander("🌍 Global Markets Overnight"):
        cols = st.columns(len(cues.get("global_markets", [])))
        for idx, mkt in enumerate(cues.get("global_markets", [])):
            with cols[idx]:
                st.metric(
                    f"{mkt['emoji']} {mkt['label']}",
                    mkt.get("last", "N/A"),
                    f"{mkt.get('change', 0.0):+.2f}%",
                    delta_color="normal",
                )
    sentiment = cues.get("overnight_sentiment", "NEUTRAL ⚪")
    st.info(f"**Overnight Sentiment:** {sentiment} "
            f"(Avg global: {cues.get('avg_global_bias_pct', 0.0):+.3f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Render: Trade Signal Panel
# ─────────────────────────────────────────────────────────────────────────────

def _render_signal(sig: dict, ticker: str):
    ticker_clean = sig.get("ticker_clean", "NIFTY")

    # ── Header ────────────────────────────────────────────────────────────────
    direction = sig.get("direction", "UP")
    conv      = sig.get("final_conviction", 0.0)
    emoji     = "🟢📈" if direction == "UP" else "🔴📉"
    st.markdown(f"## {emoji} {ticker_clean} Intraday Signal")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Direction", f"{'UP (CALL/CE)' if direction == 'UP' else 'DOWN (PUT/PE)'}")
        st.metric("Contract", sig.get("contract", "—"))
    with col2:
        st.metric("Conviction", f"{conv:.1f}%",
                  f"(need ≥{sig.get('required_threshold', 62):.0f}%)")
        st.metric("Model Confidence", sig.get("confidence", "—"))
    with col3:
        st.metric("Spot (est.)", f"₹{sig.get('spot', 0):,.2f}")
        st.metric("VWAP", f"₹{sig.get('vwap', 0):,.2f}" if sig.get("vwap") else "N/A")

    # ── Action Banner ─────────────────────────────────────────────────────────
    signal = sig.get("signal", "NO_SIGNAL")
    if signal == "NO_SIGNAL":
        st.error(f"⚪ **SKIP** — {sig.get('action', 'Low conviction')}")
    elif signal == "WAIT_VWAP":
        st.warning(f"⏳ **WAIT** — {sig.get('action', 'VWAP not confirmed')}")
    else:
        st.success(f"✅ **{sig.get('action', '')}**")

    # VWAP gate status
    if sig.get("vwap_confirmed"):
        st.success(f"**VWAP Gate:** ✅ {sig.get('vwap_reason', '')}")
    else:
        st.warning(f"**VWAP Gate:** ❌ {sig.get('vwap_reason', '')}")
    st.caption(f"Method: {sig.get('vwap_method', 'N/A')}")

    # ── Groww Live Account Status ─────────────────────────────────────────────
    acc = sig.get("groww_account", {})
    if acc.get("connected"):
        st.success(
            f"⚡ **Groww Live Broker Pipe Active** | UCC: `{acc.get('ucc')}` | "
            f"Available Cash: **₹{acc.get('balance', 0):,.2f}** | "
            f"Allowed Lot Size: **{acc.get('lots', 1)} Lot(s)** (Capital Discipline Rule)"
        )
    elif acc.get("alert"):
        st.error(acc.get("alert"))

    # ── ATM Strike Points (Live Zero-Delay) ────────────────────────────────────
    atm_s = sig.get("atm_strike")
    ce_ltp = sig.get("atm_ce_ltp", 0.0)
    pe_ltp = sig.get("atm_pe_ltp", 0.0)
    st.markdown(f"#### 🎯 ATM Strike Points ({ticker_clean} {atm_s})")
    c_ce, c_pe = st.columns(2)
    with c_ce:
        is_sel_ce = (sig.get("option_type") == "CE")
        st.metric(
            f"{'🟢 ' if is_sel_ce else ''}{ticker_clean} {atm_s} CE (Call Option)",
            f"₹{ce_ltp:.2f}" if ce_ltp > 0 else "Offline",
            "🎯 Selected Trade Contract" if is_sel_ce else "Alternative Strike"
        )
    with c_pe:
        is_sel_pe = (sig.get("option_type") == "PE")
        st.metric(
            f"{'🔴 ' if is_sel_pe else ''}{ticker_clean} {atm_s} PE (Put Option)",
            f"₹{pe_ltp:.2f}" if pe_ltp > 0 else "Offline",
            "🎯 Selected Trade Contract" if is_sel_pe else "Alternative Strike"
        )

    # ── Premium Levels (Fix #4 dynamic) ──────────────────────────────────────
    st.markdown("#### 📊 Selected Contract Risk & Targets (Dynamic — Fix #4)")
    prem_tag = "Live LTP" if sig.get("is_live_premium") else "est."
    tgt_pct = sig.get("target_pct", 30.0)
    sl_pct = sig.get("sl_pct", 15.0)
    trail_pct = sig.get("trail_trigger_pct", 18.0)
    lock_pct = sig.get("lock_trigger_pct", 25.0)
    lock_profit = sig.get("lock_profit_pct", 15.0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"Entry Premium ({prem_tag})", f"₹{sig.get('entry_premium', 0):.2f}")
        diff_sl = sig.get('entry_premium', 0) - sig.get('sl_premium', 0)
        st.metric("Stop Loss", f"₹{sig.get('sl_premium', 0):.2f}",
                  f"−₹{diff_sl:.2f} (−{sl_pct:.0f}%)")
    with col2:
        diff_tgt = sig.get('target_premium', 0) - sig.get('entry_premium', 0)
        st.metric("Target", f"₹{sig.get('target_premium', 0):.2f}",
                  f"+₹{diff_tgt:.2f} (+{tgt_pct:.0f}%)")
        st.metric("R:R Ratio", f"{sig.get('rr_ratio', 0):.1f} : 1")
    with col3:
        st.metric("Trail SL to Cost at", f"₹{sig.get('trail_trigger', 0):.2f} (+{trail_pct:.0f}%)")
        st.metric(f"Lock +{lock_profit:.0f}% Profit at", f"₹{sig.get('lock_trigger', 0):.2f} (+{lock_pct:.0f}%)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lot Size", sig.get("lot_size", 65))
    with col2:
        st.metric("Max Gain / Lot", f"₹{sig.get('gain_per_lot', 0):,.0f}")
    with col3:
        st.metric("Max Loss / Lot", f"₹{sig.get('loss_per_lot', 0):,.0f}")

    st.info(f"⏰ **Hard Cutoff: {sig.get('hard_cutoff', '15:05')}** — Exit at market regardless of P&L.")


# ─────────────────────────────────────────────────────────────────────────────
# Render: OI/PCR Panel
# ─────────────────────────────────────────────────────────────────────────────

def _render_oi_panel(sig: dict):
    st.markdown("#### 🔍 Live OI/PCR (Post-Model Modifier)")
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
        st.metric("OI Conviction Adj.", f"{adj:+.1f}%",
                  f"{sig.get('base_conviction', 0):.1f}% → {sig.get('final_conviction', 0):.1f}%")

    if sig.get("oi_signals"):
        for s in sig["oi_signals"]:
            st.caption(f"• {s}")
    else:
        st.caption("No OI modifier signals triggered.")


# ─────────────────────────────────────────────────────────────────────────────
# Render: Model Stats Panel
# ─────────────────────────────────────────────────────────────────────────────

def _render_model_stats(model_result: dict):
    st.markdown("#### 🤖 Model Statistics")
    metrics = model_result.get("test_metrics", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test Accuracy", f"{metrics.get('accuracy', 0):.1f}%")
    with col2:
        st.metric("Precision", f"{metrics.get('precision', 0):.1f}%")
    with col3:
        st.metric("Brier Score", f"{metrics.get('brier', 0):.4f}")
    with col4:
        st.metric("Models Used", metrics.get("n_models", 2))

    regime = model_result.get("regime", "—")
    rd     = model_result.get("regime_detail", "")
    badge  = "🟢" if regime == "TRENDING" else ("🟡" if regime == "VOLATILE" else "🔴")
    st.info(f"{badge} **Regime: {regime}** — {rd}")

    importances = model_result.get("feature_importances", {})
    if importances:
        with st.expander("📊 Top Feature Importances"):
            top = dict(list(importances.items())[:15])
            df_imp = pd.DataFrame(list(top.items()), columns=["Feature", "Importance"])
            df_imp = df_imp.sort_values("Importance", ascending=True)
            st.bar_chart(df_imp.set_index("Feature"))


def _render_sentinel_monitor():
    state_file = os.path.join(PROJECT_ROOT, "live_sentinel_state.json")
    if os.path.exists(state_file):
        try:
            import json
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            st.markdown("### ⚡ Real-Time 5-Second Sentinel Stream")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Sentinel Feed", "🟢 ACTIVE (5s)", f"Tick: {state.get('timestamp')}")
            with c2:
                st.metric("Tick Latency", f"{state.get('latency_ms', 0):.0f} ms", "Zero-Delay Groww Pipe")
            with c3:
                atm = state.get('atm_strike', 0)
                ce = state.get('atm_ce_ltp', 0.0)
                pe = state.get('atm_pe_ltp', 0.0)
                st.metric(f"Live ATM {atm}", f"CE: ₹{ce:.2f} | PE: ₹{pe:.2f}")
            with c4:
                cash = state.get('cash_margin', 0.0)
                lots = state.get('allowed_lots', 1)
                st.metric("Groww Account", f"₹{cash:,.2f}" if cash > 0 else "Connected", f"Max: {lots} Lot(s)")

            pos = state.get("active_position")
            if pos:
                pnl = pos.get('pnl_rs', 0.0)
                pnl_color = "🟢" if pnl >= 0 else "🔴"
                st.success(
                    f"🎯 **ACTIVE TRADE:** {pos.get('contract')} | "
                    f"Entry: ₹{pos.get('entry_premium', 0):.2f} | Current LTP: ₹{pos.get('current_ltp', 0):.2f} | "
                    f"P&L: **{pnl_color} ₹{pnl:+,.0f} ({pos.get('pnl_pct', 0):+.1f}%)** | "
                    f"Current SL: ₹{pos.get('current_sl', 0):.2f} ({pos.get('trail_state')}) | "
                    f"Target: ₹{pos.get('target_premium', 0):.2f}"
                )
            else:
                st.caption(
                    f"🔍 Sentinel scanning {state.get('ticker_name')} every 5s | "
                    f"Spot: ₹{state.get('spot', 0):,.2f} | VWAP: {state.get('vwap_status')} | "
                    f"Bias: {state.get('direction')} ({state.get('conviction', 0):.1f}%)"
                )
            st.divider()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry (called from app.py)
# ─────────────────────────────────────────────────────────────────────────────

def render_nifty_predictor_tab():
    """Main render function — call from app.py Streamlit tab."""
    st.title("🎯 Nifty & Sensex Intraday Options Predictor")
    st.caption("Powered by 4-model ML ensemble (RF + HGB + LGBM + XGBoost) "
               "| 46 features | Groups A+C+D+E | VWAP gated | OI/PCR live modifier")

    # ── Pre-market briefing ───────────────────────────────────────────────────
    with st.spinner("Fetching overnight cues..."):
        cues = _cached_overnight_cues()
    _render_premarket(cues)
    st.divider()

    # ── Ticker selector ───────────────────────────────────────────────────────
    ticker_opts = {"Nifty 50 (^NSEI)": NIFTY_TICKER, "Sensex (^BSESN)": SENSEX_TICKER}
    col1, col2 = st.columns([2, 1])
    with col1:
        sel = st.selectbox("Select Index", list(ticker_opts.keys()))
    with col2:
        run_btn = st.button("🔄 Refresh Signal", use_container_width=True)
    ticker = ticker_opts[sel]

    if run_btn:
        st.cache_data.clear()

    # ── Run model + signal ────────────────────────────────────────────────────
    with st.spinner(f"Training ensemble on {sel}..."):
        model_result = _cached_model(ticker)

    if model_result.get("status") != "success":
        st.error(f"Model error: {model_result.get('message', 'Unknown')}")
        return

    other_ticker = SENSEX_TICKER if ticker == NIFTY_TICKER else NIFTY_TICKER
    other_label  = "Sensex" if ticker == NIFTY_TICKER else "Nifty"

    @st.fragment(run_every="5s")
    def _render_live_stream_fragment(ticker_sym: str, model_data: dict, other_sym: str, other_name: str):
        # ── 5-Second Sentinel Stream ──
        _render_sentinel_monitor()

        now_ist_str = datetime.now(IST).strftime("%I:%M:%S %p IST")
        st.caption(f"⚡ **Live 5s Reactive Stream Active** • Quotes auto-refreshing continuously without page reload • Last tick: `{now_ist_str}`")

        # ── Live Trade Signal + VWAP Gate + OI/PCR ──
        try:
            live_sig = generate_trade_signal(model_data, ticker=ticker_sym,
                                             fetch_live_oi=True, check_vwap=True)
        except Exception as e:
            st.error(f"Signal generation failed: {e}")
            return

        # ── Cross-confirmation bonus ──
        try:
            other_res = _cached_model(other_sym)
            if other_res.get("status") == "success":
                other_d = other_res.get("direction", "—")
                main_d  = model_data.get("direction", "—")
                if other_d == main_d:
                    live_sig["final_conviction"] = min(live_sig["final_conviction"] + 5.0, 99.9)
                    live_sig["oi_description"] += f" | {other_name} cross-confirms."
        except Exception:
            pass

        _render_signal(live_sig, ticker_sym)
        st.divider()
        _render_oi_panel(live_sig)

        # ── Console output for debugging ──
        with st.expander("📋 Full Signal Text (Console Format)"):
            st.code(format_signal_console(live_sig), language="text")

    # Execute 5-second continuous live stream fragment
    _render_live_stream_fragment(ticker, model_result, other_ticker, other_label)
    st.divider()
    _render_model_stats(model_result)
    st.divider()
    render_daily_journal_section(model_result)
