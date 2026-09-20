import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import datetime as dt

from utils.data_loader import get_stock_data
from utils.opening_predictor import predict_opening_gap
from utils.gap_analysis import analyze_intraday_gap_and_zones
from utils.market_calendar import get_market_dates, is_trading_holiday
from utils.opening_audit import (
    record_opening_gap_prediction,
    evaluate_opening_gap_outcomes,
    load_saved_gap_audit_history,
    get_locked_opening_gap_snapshot
)
from utils.ui_theme import apply_custom_theme
from config import STOCK_NAME_MAP, MIN_GAP_CONVICTION_THRESHOLD

# ── HDFCBANK Standalone Specialist (Step B) — completely separate from CDSL ──
from utils.hdfc_specialist import predict_hdfc_opening_gap
from utils.hdfc_learning_audit import (
    record_hdfc_prediction,
    evaluate_hdfc_outcomes,
    load_hdfc_audit_history,
    get_locked_hdfc_snapshot,
)

HDFC_SYMBOL = "HDFCBANK.NS"


def get_ist_time() -> dt.datetime:
    """Return current Indian Standard Time (UTC + 5:30)."""
    now_utc = dt.datetime.now(dt.timezone.utc)
    return now_utc + dt.timedelta(hours=5, minutes=30)

def render_opening_prediction_page():
    apply_custom_theme()

    st.markdown("<div class='glowing-header'>🔮 3:05 PM Next-Morning Opening Gap Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>Intraday Session Machine Learning Model for Next-Day 9:15 AM Gap Direction & Options Entry (Put / Call)</div>", unsafe_allow_html=True)

    # Calculate current IST time
    ist_time = get_ist_time()
    ist_str = ist_time.strftime("%I:%M:%S %p IST (%a, %d %b %Y)")
    ist_hour = ist_time.hour
    ist_minute = ist_time.minute
    total_minutes = ist_hour * 60 + ist_minute

    # Check official NSE trading holiday
    is_today_holiday, today_holiday_name = is_trading_holiday(ist_time.date())

    # Time-Gating Status Logic:
    # 3:00 PM = 15:00 = 900 min
    # 3:05 PM = 15:05 = 905 min
    # 3:30 PM = 15:30 = 930 min
    is_weekday = ist_time.weekday() < 5
    is_live_trading_window = is_weekday and (not is_today_holiday) and (905 <= total_minutes <= 930)
    is_prep_window = is_weekday and (not is_today_holiday) and (900 <= total_minutes < 905)
    is_market_closed = (not is_weekday) or is_today_holiday or (total_minutes > 930) or (total_minutes < 555)

    # Time Gate Banner & Mode Override
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        if is_today_holiday:
            st.info(
                f"🏖️ **MARKET CLOSED TODAY ({today_holiday_name.upper()})** ({ist_str})\n\n"
                f"NSE & BSE are closed for **{today_holiday_name}**. Model displays forecast targeting the next active market opening."
            )
        elif is_live_trading_window:
            st.success(
                f"🟢 **LIVE 3:05 PM OPTIONS WINDOW ACTIVE** ({ist_str})\n\n"
                "Full intraday price action processed. Enter Put / Call options before 3:25 PM IST to capture tomorrow's 9:15 AM opening gap!"
            )
        elif is_prep_window:
            remaining_sec = (905 * 60) - (ist_hour * 3600 + ist_minute * 60 + ist_time.second)
            st.warning(
                f"⏳ **PREPARING FINAL 3:05 PM INTRADAY SNAPSHOT** ({ist_str})\n\n"
                f"Model unlocking in **{max(remaining_sec, 0)} seconds**. Processing final institutional volume and closing wicks..."
            )
        elif is_market_closed:
            st.info(
                f"🌙 **MARKET CLOSED / POST-SESSION** ({ist_str})\n\n"
                "Reviewing latest closing session data. Forecast indicates expected 9:15 AM opening move for the next market session."
            )
        else:
            st.info(
                f"⏳ **SESSION IN PROGRESS** ({ist_str})\n\n"
                "Opening gap predictor requires full session data up to 3:00 PM for maximum accuracy. Official 3:05 PM window unlocks at 3:05 PM IST."
            )

    with col_t2:
        bypass_gate = st.checkbox(
            "🔓 Unlock Preview Mode",
            value=True if (not is_live_trading_window) else False,
            help="Allows inspection and backtesting of the 3:05 PM opening gap model anytime outside the live 3:05-3:10 PM window."
        )

    # If before 3:05 PM on a trading day and user hasn't bypassed
    if not is_live_trading_window and not bypass_gate and not is_market_closed:
        st.warning(
            "🔒 **Time-Gate Active**: Please return between **3:05 PM and 3:10 PM IST** for high-confidence options entry signals, "
            "or check **'Unlock Preview Mode'** above to preview current indicators."
        )
        return

    st.markdown("---")

    # Stock & Horizon Selectors
    # Always pin the two specialist stocks at the top, then append any additional watchlist stocks
    SPECIALIST_STOCKS = ["CDSL.NS", "HDFCBANK.NS"]
    watchlist = st.session_state.get("watchlist", ["CDSL.NS", "NSDL.BO"])
    # Merge: specialists first, then any extra watchlist stocks not already in the specialist list
    extra = [s for s in watchlist if s not in SPECIALIST_STOCKS]
    predictor_options = SPECIALIST_STOCKS + extra

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        selected_symbol = st.selectbox(
            "Select Stock for Opening Gap Prediction",
            options=predictor_options,
            format_func=lambda s: f"{STOCK_NAME_MAP.get(s, s)} ({s})"

        )
    with col_s2:
        period = st.selectbox("Training Data Horizon", options=["1y", "2y", "3y"], index=1)

    if not selected_symbol:
        st.warning("Please select a stock.")
        return

    # Fetch raw data for market dates
    df_raw = get_stock_data(selected_symbol, period=period)
    dates_info = get_market_dates(df_raw)

    # 1. Trading Target & Timing Header
    latest_close_val = df_raw['Close'].iloc[-1] if not df_raw.empty else 0.0
    if is_market_closed:
        st.subheader(f"📅 Target Market Open: {dates_info['next_date_str']} (9:15 AM – 9:20 AM IST)")
        st.markdown(
            f"<div style='background: rgba(56, 189, 248, 0.08); border-left: 4px solid #38bdf8; padding: 10px 14px; border-radius: 6px; margin-bottom: 15px; font-size: 0.92rem;'>"
            f"✅ <strong>Session Complete ({dates_info['last_date_str']}):</strong> Market closed at <strong>₹{latest_close_val:,.2f}</strong>. "
            f"The 3:05 PM Gap Predictor below has incorporated today's full session data to forecast the <strong>9:15 AM opening gap for tomorrow ({dates_info['next_date_str']})</strong>.<br>"
            f"📜 <em>To review today's ({dates_info['last_date_str']}) audited opening gap outcome and past performance, see the Audit Ledger table below.</em>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.subheader(f"📅 Target Market Open: {dates_info['next_date_str']} (9:15 AM – 9:20 AM IST)")
        st.caption("Predicting overnight gap direction: Will tomorrow's 9:15 AM Open be GREATER than today's 3:30 PM Close?")

    if dates_info.get('holiday_alert'):
        st.warning(f"🏖️ **Market Holiday Notice:** {dates_info['holiday_alert']}")

    # Fetch real-time intraday price action
    intraday = analyze_intraday_gap_and_zones(selected_symbol)
    if intraday.get("status") == "success":
        c_p1, c_p2, c_p3, c_p4, c_p5 = st.columns(5)
        c_p1.metric("Current Trading Price (LTP)", f"₹{intraday['current_price']:,.2f}", f"{intraday['day_change_pct']:+.2f}%")
        c_p2.metric("Today's 9:15 AM Open", f"₹{intraday['open_price']:,.2f}")
        c_p3.metric("Intraday Range (H - L)", f"₹{intraday['day_high']:,.2f} - ₹{intraday['day_low']:,.2f}")
        c_p4.metric("Session VWAP", f"₹{intraday['vwap']:,.2f}", f"{intraday['vwap_diff_pct']:+.2f}% vs VWAP")
        c_p5.metric("Session Regime", intraday.get("gap_badge", "CONSOLIDATION"))

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. SMART ROUTING: CDSL → CDSL Specialist | HDFCBANK → HDFC Specialist
    #    CDSL predictor (opening_predictor.py) is NEVER touched by HDFC logic.
    # ─────────────────────────────────────────────────────────────────────────
    target_open_date = dates_info['next_date_str']
    is_hdfc = (selected_symbol == HDFC_SYMBOL)

    # Safe defaults — prevent NameError if any routing path encounters an exception
    result        = {"status": "error", "message": "Model not yet initialized."}
    is_frozen     = False
    snapshot_time = ist_time.strftime("%I:%M %p IST")

    if is_hdfc:
        # ── HDFCBANK Standalone Specialist Path ───────────────────────────
        locked_snapshot = get_locked_hdfc_snapshot(target_open_date)
        if locked_snapshot and not is_live_trading_window:
            result        = locked_snapshot
            is_frozen     = True
            snapshot_time = locked_snapshot.get("prediction_time", "3:10 PM IST")
        else:
            with st.spinner("🏦 Running HDFCBANK Autonomous Self-Learning Specialist Model..."):
                result = predict_hdfc_opening_gap(period=period)
            is_frozen     = False
            snapshot_time = ist_time.strftime("%I:%M %p IST")
            if result.get("status") == "success":
                record_hdfc_prediction(target_open_date, result)
                try:
                    from utils.paper_trading import record_simulated_gap_entry
                    record_simulated_gap_entry(selected_symbol, target_open_date, result)
                except Exception:
                    pass

        # Banking Radar telemetry widget (HDFC-only)
        if result.get("status") == "success":
            radar   = result.get("banking_radar", {})
            bn_ret  = radar.get("banknifty_ret1", 0.0)
            hdfc_bn = radar.get("hdfc_vs_bn", 0.0)
            us10y   = radar.get("us_10y_ret1", 0.0)
            d_bn    = radar.get("days_to_bn_expiry", "—")
            d_mo    = radar.get("days_to_monthly", "—")
            is_wed  = radar.get("is_wednesday", False)
            learning_reason = result.get("gap_reason", "")

            st.markdown(
                "<div style='background:rgba(251,191,36,0.12);border-left:5px solid #fbbf24;"
                "padding:10px 16px;border-radius:6px;margin-bottom:14px'>"
                "🏦 <strong>HDFCBANK AUTONOMOUS SELF-LEARNING SPECIALIST</strong> — "
                "Banking Microstructure + Bank Nifty Co-Integration + US 10-Year Yield Cues"
                "</div>", unsafe_allow_html=True
            )
            r1, r2, r3, r4 = st.columns(4)
            bn_color = "#00E676" if bn_ret >= 0 else "#FF5252"
            bn_icon  = "🟢" if bn_ret >= 0 else "🔴"
            us_color = "#FF5252" if us10y > 0.1 else ("#00E676" if us10y < -0.1 else "#FFB300")
            r1.markdown(
                f"<div style='background:rgba(0,0,0,0.3);border:1px solid #334155;padding:10px;border-radius:8px;text-align:center'>"
                f"<div style='font-size:0.78rem;color:#94a3b8'>Bank Nifty Lead-Lag</div>"
                f"<div style='font-size:1.1rem;font-weight:700;color:{bn_color}'>{bn_icon} {bn_ret:+.2f}%</div>"
                f"<div style='font-size:0.73rem;color:#64748b'>{'Aligned ✓' if abs(hdfc_bn)<0.2 else f'Spread {hdfc_bn:+.2f}%'}</div>"
                f"</div>", unsafe_allow_html=True)
            r2.markdown(
                f"<div style='background:rgba(0,0,0,0.3);border:1px solid #334155;padding:10px;border-radius:8px;text-align:center'>"
                f"<div style='font-size:0.78rem;color:#94a3b8'>US 10-Yr Yield Δ</div>"
                f"<div style='font-size:1.1rem;font-weight:700;color:{us_color}'>{us10y:+.3f}%</div>"
                f"<div style='font-size:0.73rem;color:#64748b'>{'⚠️ FII Outflow Risk' if us10y>0.1 else ('✅ FII Supportive' if us10y<-0.1 else '⚪ Neutral')}</div>"
                f"</div>", unsafe_allow_html=True)
            r3.markdown(
                f"<div style='background:rgba(0,0,0,0.3);border:1px solid #334155;padding:10px;border-radius:8px;text-align:center'>"
                f"<div style='font-size:0.78rem;color:#94a3b8'>Expiry Regime</div>"
                f"<div style='font-size:1.1rem;font-weight:700;color:#818cf8'>{d_bn}d BN / {d_mo}d Stock</div>"
                f"<div style='font-size:0.73rem;color:#64748b'>{'⚠️ BN Wed Expiry' if is_wed else 'Normal Session'}</div>"
                f"</div>", unsafe_allow_html=True)
            r4.markdown(
                f"<div style='background:rgba(0,0,0,0.3);border:1px solid #334155;padding:10px;border-radius:8px;text-align:center'>"
                f"<div style='font-size:0.78rem;color:#94a3b8'>Self-Learning Status</div>"
                f"<div style='font-size:0.85rem;font-weight:700;color:#38bdf8'>{'✅ Balanced' if '✅' in learning_reason else '🔄 Adaptive'}</div>"
                f"<div style='font-size:0.73rem;color:#64748b'>Rolling 10-session window</div>"
                f"</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    else:
        # ── CDSL Specialist Path — 100% Untouched ─────────────────────────
        locked_snapshot = get_locked_opening_gap_snapshot(selected_symbol, target_open_date)
        if locked_snapshot and not is_live_trading_window:
            result        = locked_snapshot["pred_result"]
            is_frozen     = True
            snapshot_time = locked_snapshot.get("prediction_time", "3:10 PM IST")
        else:
            with st.spinner(f"Computing 3:05 PM Opening Gap Ensemble Model for {selected_symbol}..."):
                result = predict_opening_gap(selected_symbol, period=period)
            is_frozen     = False
            snapshot_time = ist_time.strftime("%I:%M %p IST")
            if result.get("status") == "success":
                record_opening_gap_prediction(selected_symbol, target_open_date, result, lock_snapshot=True)
                try:
                    from utils.paper_trading import record_simulated_gap_entry
                    record_simulated_gap_entry(selected_symbol, target_open_date, result)
                except Exception:
                    pass

    if result.get("status") != "success":
        st.error(f"Opening gap prediction failed: {result.get('message')}")
        return

    # Display Frozen Snapshot Badge if applicable
    if is_frozen:
        st.markdown(
            f"<div style='background: rgba(14, 165, 233, 0.12); border-left: 5px solid #38bdf8; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;'>"
            f"🔒 <strong>OFFICIAL 3:05 PM SNAPSHOT (LOCKED AT {snapshot_time})</strong><br>"
            f"<span style='color: #cbd5e1; font-size: 0.92rem;'>"
            f"This prediction was sealed during the 3:05 PM – 3:15 PM options entry window and is <strong>permanently frozen overnight</strong>. "
            f"Overnight US market moves or morning pre-market noise will not alter these probabilities so you can reliably trade and audit the 3:05 PM signal."
            f"</span></div>",
            unsafe_allow_html=True
        )

    # 3. Forecast Result Cards
    options_call   = result.get("options_call", {})
    action_text    = options_call.get("action", "HOLD")
    bias_color     = options_call.get("bias_color", "#38bdf8")

    st.subheader("🎯 3:05 PM Gap Forecast & Put/Call Direction")

    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    with col_g1:
        st.metric("Expected Opening Gap", result['direction'])
    with col_g2:
        st.metric("Gap Up Probability", f"{result['probability_up_pct']}%")
    with col_g3:
        st.metric("Gap Down Probability", f"{result['probability_down_pct']}%")
    with col_g4:
        st.metric("Confidence Rating", result['confidence'])

    # 4. ACTIONABLE OPTIONS TRADING CARD
    prob_up_val    = float(result.get('probability_up_pct', 50.0))
    conviction_val = max(prob_up_val, 100.0 - prob_up_val)
    active_threshold = options_call.get("active_threshold", MIN_GAP_CONVICTION_THRESHOLD)
    if conviction_val < active_threshold:
        st.markdown(
            f"<div style='background: rgba(245, 158, 11, 0.15); border-left: 6px solid #f59e0b; padding: 14px 18px; border-radius: 8px; margin-bottom: 16px;'>"
            f"<h3 style='margin:0; color: #f59e0b; font-size: 1.25rem;'>🛡️ Capital Preserved: Filter Active (&lt;{active_threshold:.0f}% Conviction)</h3>"
            f"<p style='margin: 6px 0 0 0; color: #e2e8f0; font-size: 0.95rem;'>"
            f"Current model conviction is <strong>{conviction_val:.1f}%</strong> (below the validated <strong>{active_threshold:.0f}% conviction threshold</strong>). "
            f"197-day walk-forward backtests confirm that carrying overnight positions below this threshold is vulnerable to overnight theta decay. "
            f"<strong>100% Cash preservation advised — no overnight trade will be placed.</strong>"
            f"</p></div>",
            unsafe_allow_html=True
        )

    st.markdown("### ⚡ Actionable Put / Call Options Entry Blueprint")
    with st.container(border=True):
        st.markdown(
            f"<div style='background: {bias_color}22; border-left: 6px solid {bias_color}; padding: 12px 18px; border-radius: 8px; margin-bottom: 12px;'>"
            f"<h2 style='margin:0; color: {bias_color}; font-size: 1.6rem;'>{action_text} — {options_call.get('strategy', 'Overnight Strategy')}</h2>"
        f"<p style='margin: 4px 0 0 0; color: #cbd5e1; font-size: 0.95rem;'>Targeting tomorrow's 9:15 AM opening gap move on <strong>{selected_symbol}</strong> (LTP: ₹{result.get('current_price', 0):,.2f})</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        o1, o2, o3, o4 = st.columns(4)
        o1.metric("🎯 Recommended Strike", options_call.get("suggested_strike", "N/A"))
        o2.metric("🛡️ Alternative / Safe Strike", options_call.get("alt_strike", "N/A"))
        o3.metric("⏰ Options Entry Window", options_call.get("entry_window", "3:10 PM - 3:20 PM"))
        o4.metric("🏁 Options Exit Window", options_call.get("exit_window", "9:15 AM - 9:20 AM"))

        # Microstructure Badges — handles both CDSL (days_to_expiry) and HDFC (days_to_monthly_expiry)
        days_exp  = options_call.get("days_to_monthly_expiry",
                    options_call.get("days_to_expiry",
                    result.get("days_to_expiry", 0)))
        is_exp_wk = options_call.get("is_expiry_week",  result.get("is_expiry_week", False))
        is_fri    = options_call.get("is_friday",        result.get("is_friday", False))
        active_thresh = options_call.get("active_threshold", MIN_GAP_CONVICTION_THRESHOLD)

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(
                f"<div style='background: rgba(14, 165, 233, 0.15); border: 1px solid #38bdf8; padding: 8px 12px; border-radius: 6px; font-size: 0.88rem; text-align: center;'>"
                f"📅 <strong>Monthly Expiry:</strong> {days_exp} days remaining"
                f"{' <span style=\"color:#f59e0b;\">(Expiry Week)</span>' if is_exp_wk else ''}"
                f"</div>",
                unsafe_allow_html=True
            )
        with m_col2:
            if is_fri:
                st.markdown(
                    f"<div style='background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; padding: 8px 12px; border-radius: 6px; font-size: 0.88rem; text-align: center;'>"
                    f"🛡️ <strong>Friday Weekend Shield:</strong> Active (≥{active_thresh:.0f}% Threshold)"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='background: rgba(34, 197, 94, 0.15); border: 1px solid #22c55e; padding: 8px 12px; border-radius: 6px; font-size: 0.88rem; text-align: center;'>"
                    f"🛡️ <strong>Weekday Threshold:</strong> ≥{active_thresh:.0f}% Conviction"
                    f"</div>",
                    unsafe_allow_html=True
                )
        with m_col3:
            if is_exp_wk:
                st.markdown(
                    f"<div style='background: rgba(168, 85, 247, 0.15); border: 1px solid #a855f7; padding: 8px 12px; border-radius: 6px; font-size: 0.88rem; text-align: center;'>"
                    f"⚔️ <strong>Expiry ITM Armor:</strong> Active (Delta ~0.70)"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='background: rgba(100, 116, 139, 0.15); border: 1px solid #64748b; padding: 8px 12px; border-radius: 6px; font-size: 0.88rem; text-align: center;'>"
                    f"🎯 <strong>Standard ATM Delta:</strong> ~0.50 Delta Sizing"
                    f"</div>",
                    unsafe_allow_html=True
                )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.warning(f"⚠️ **Risk Management Rule:** {options_call.get('risk_guideline', 'Follow disciplined risk limits.')}")
        if result.get("gap_reason"):
            st.caption(f"🔄 **Historical Gap Feedback:** {result['gap_reason']}")

    # Telegram Push Button
    stock_display = STOCK_NAME_MAP.get(selected_symbol, selected_symbol)
    if st.button(f"📱 Send 3:05 PM Gap Prediction for {selected_symbol} to My Phone via Telegram", use_container_width=True):
        with st.spinner("Pushing 3:05 PM opening gap alert to Telegram..."):
            try:
                import importlib
                import utils.notifications as notif_mod
                importlib.reload(notif_mod)
                send_fn = getattr(notif_mod, 'send_opening_gap_alert_notification', None)
                if callable(send_fn):
                    success, err_msg = send_fn(
                        symbol=selected_symbol,
                        name=stock_display,
                        gap_direction=result['direction'],
                        prob_up=result['probability_up_pct'],
                        confidence=result['confidence'],
                        options_call=options_call,
                        target_date=dates_info['next_date_str'],
                        current_price=result['current_price']
                    )
                else:
                    success, err_msg = False, "Telegram notification module is reloading."
            except Exception as e_notif:
                success, err_msg = False, str(e_notif)

        if success:
            st.success("🎉 3:05 PM Opening Gap Alert sent to your phone Telegram!")
        else:
            st.error(f"❌ Alert error: {err_msg}")

    # 5. Probability Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result['probability_up_pct'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Overnight Gap Up Probability for {dates_info['next_date_str']} (9:15 AM Open)", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#26a69a" if result['probability_up_pct'] >= 50 else "#ef5350"},
            'steps': [
                {'range': [0, 40], 'color': "rgba(239, 83, 80, 0.3)"},
                {'range': [40, 60], 'color': "rgba(255, 235, 59, 0.3)"},
                {'range': [60, 100], 'color': "rgba(38, 166, 154, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig_gauge.update_layout(height=260, template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")

    # 6. AUDIT TRAIL — Routes to correct specialist ledger
    st.subheader("🕵️ 3:05 PM Opening Gap Prediction vs Actual 9:15 AM Open Audit")

    if is_hdfc:
        # ── HDFCBANK Independent Audit Ledger ─────────────────────────────
        st.caption("🏦 HDFCBANK Self-Learning Audit Ledger — tracks outcomes, adapts calibration using a rolling 10-session window.")
        hdfc_history = evaluate_hdfc_outcomes()

        if hdfc_history:
            completed_h    = [a for a in hdfc_history if a.get("is_correct") is not None]
            correct_h      = sum(1 for a in completed_h if a.get("is_correct") is True)
            total_h        = len(completed_h)
            hit_rate_h     = round((correct_h / total_h * 100.0), 1) if total_h > 0 else 0.0

            gh1, gh2, gh3, gh4 = st.columns(4)
            gh1.metric("HDFC Gap Hit Rate",  f"{hit_rate_h}%" if total_h > 0 else "Pending Data")
            gh2.metric("Audited Sessions",   f"{total_h} Sessions")
            gh3.metric("Verified Gap Hits",  f"✅ {correct_h}")
            gh4.metric("Gap Divergences",    f"❌ {total_h - correct_h}")

            # "What Happened & Why" Self-Learning Panel
            recent_lessons = []
            for record in reversed(hdfc_history[-5:]):
                if record.get("is_correct") is not None:
                    diag = record.get("divergence_reasons", [""])
                    recent_lessons.append({
                        "date":     record.get("target_date"),
                        "result":   "✅ Correct" if record.get("is_correct") else "❌ Diverged",
                        "predicted": record.get("predicted_gap_direction"),
                        "actual_pct": record.get("actual_gap_pct", 0.0),
                        "action":   record.get("options_action", ""),
                        "diagnosis": diag[0] if diag else "",
                        "learning_note": record.get("learning_note", ""),
                    })

            if recent_lessons:
                st.markdown("#### 🧠 What Happened & Why — Self-Learning Memory")
                st.caption("Recent sessions and the root-cause analysis that fed into today's calibration.")
                for lesson in recent_lessons:
                    result_color = "#00E676" if "✅" in lesson["result"] else "#FF5252"
                    actual_str = f"{lesson['actual_pct']:+.2f}%" if lesson.get("actual_pct") is not None else "Pending"
                    st.markdown(
                        f"<div style='background:rgba(0,0,0,0.25);border-left:4px solid {result_color};"
                        f"padding:10px 14px;border-radius:6px;margin-bottom:8px'>"
                        f"<strong>{lesson['date']}</strong> &nbsp;|&nbsp; {lesson['result']} &nbsp;|&nbsp; "
                        f"Predicted: <em>{lesson['predicted']}</em> &nbsp;→&nbsp; Actual: <strong>{actual_str}</strong>"
                        f"<br><span style='color:#94a3b8;font-size:0.85rem'>{lesson['diagnosis']}</span>"
                        f"{'<br><span style=\"color:#38bdf8;font-size:0.82rem\">' + lesson['learning_note'] + '</span>' if lesson.get('learning_note') else ''}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            # Full HDFC audit table
            df_hdfc_audit = pd.DataFrame([
                {
                    "Target Date": a.get("target_date"),
                    "Predicted Gap": a.get("predicted_gap_direction"),
                    "Probability": f"{a.get('probability_up_pct')}% Up",
                    "Options Action": a.get("options_action", "N/A"),
                    "3 PM Baseline": f"₹{a.get('baseline_3pm_close'):,.2f}" if a.get("baseline_3pm_close") else "N/A",
                    "Actual 9:15 AM": f"₹{a.get('actual_915_open'):,.2f}" if a.get("actual_915_open") else "⏳ Pending",
                    "Actual Gap %": f"{a.get('actual_gap_pct'):+.2f}%" if a.get("actual_gap_pct") is not None else "⏳",
                    "Learning Offset": f"{a.get('learning_offset_applied', 0.0):+.1f}%",
                    "Verdict": (
                        "✅ Hit" if a.get("is_correct") is True else
                        ("🛡️ Preserved" if "NEUTRAL" in str(a.get("options_action", "")) else
                         ("❌ Diverged" if a.get("is_correct") is False else "⏳ Awaiting 9:15 AM"))
                    )
                } for a in reversed(hdfc_history)
            ])
            st.dataframe(df_hdfc_audit, use_container_width=True, hide_index=True)
        else:
            st.info("🏦 HDFC Bank audit ledger is ready. Predictions will be automatically tracked and self-learning will begin from your first trade session.")

    else:
        # ── CDSL Audit Ledger — 100% Untouched ────────────────────────────
        st.caption("Dedicated CDSL audit ledger tracking whether the predicted opening gap matched tomorrow's actual 9:15 AM open price.")
        gap_audit_history = evaluate_opening_gap_outcomes()

        if gap_audit_history:
            completed = [a for a in gap_audit_history if a.get("is_correct") is not None]
            correct_count   = sum(1 for a in completed if a.get("is_correct") is True)
            total_completed = len(completed)
            hit_rate_pct    = round((correct_count / total_completed * 100.0), 1) if total_completed > 0 else 0.0

            ga1, ga2, ga3, ga4 = st.columns(4)
            ga1.metric("Opening Gap Hit Rate", f"{hit_rate_pct}%" if total_completed > 0 else "Pending Data")
            ga2.metric("Audited Sessions",     f"{total_completed} Sessions")
            ga3.metric("Verified Gap Hits",    f"✅ {correct_count}")
            ga4.metric("Gap Divergences",      f"❌ {total_completed - correct_count}")

            df_gap_audit = pd.DataFrame([
                {
                    "Target Open Date": a.get("target_date"),
                    "Stock": a.get("symbol"),
                    "Predicted Gap": a.get("predicted_gap_direction"),
                    "Gap Probability": f"{a.get('probability_up_pct')}% Up",
                    "Options Action": a.get("options_action", "N/A"),
                    "3:05 PM Baseline": f"₹{a.get('baseline_3pm_close'):,.2f}" if a.get('baseline_3pm_close') else "N/A",
                    "Actual 9:15 AM Open": f"₹{a.get('actual_915_open'):,.2f}" if a.get('actual_915_open') else "Pending 9:15 AM...",
                    "Actual Gap %": f"{'+' if (a.get('actual_gap_pct') or 0) >= 0 else ''}{a.get('actual_gap_pct')}%" if a.get('actual_gap_pct') is not None else "Pending...",
                    "Verification": (
                        "✅ Verified Hit" if a.get("is_correct") is True else
                        ("🛡️ Capital Preserved" if "NEUTRAL" in str(a.get("options_action", "")) else
                         ("❌ Diverged" if a.get("is_correct") is False else "⏳ Awaiting 9:15 AM Open"))
                    )
                } for a in reversed(gap_audit_history)
            ])
            st.dataframe(df_gap_audit, use_container_width=True, hide_index=True)
        else:
            st.info("Opening gap forecasts are logged automatically. Each day at 9:15 AM, actual open prices will evaluate hit rate accuracy.")


    st.markdown("---")

    # 7. Model Decision Drivers & Technical Features
    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        st.subheader("📊 Opening Gap Model Metrics")
        st.metric("Test Gap Accuracy", f"{result['test_accuracy_pct']}%")
        m1, m2 = st.columns(2)
        m1.metric("Gap Precision", f"{result.get('precision_pct', 'N/A')}%")
        m2.metric("Gap Recall", f"{result.get('recall_pct', 'N/A')}%")
        st.caption("Evaluated strictly on out-of-sample chronological test split (last 20% of trading sessions).")
        st.caption("Specialized features: Close-High position, Intraday Range %, Open-to-Close return, Wick ratios.")

    with col_f2:
        st.subheader("🔑 Top Overnight Gap Drivers")
        importances = result.get('feature_importances', {})
        df_imp = pd.DataFrame({
            "Feature": list(importances.keys()),
            "Importance": list(importances.values())
        }).sort_values(by="Importance", ascending=True).tail(10)

        fig_imp = go.Figure(go.Bar(
            x=df_imp['Importance'],
            y=df_imp['Feature'],
            orientation='h',
            marker_color='#818cf8'
        ))
        fig_imp.update_layout(
            height=280, template="plotly_dark",
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Feature Importance Weight",
            yaxis_title="Driver"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

if __name__ == "__main__" or True:
    render_opening_prediction_page()
