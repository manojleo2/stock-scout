import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import datetime as dt

from utils.data_loader import get_stock_data
from utils.opening_predictor import predict_opening_gap
from utils.gap_analysis import analyze_intraday_gap_and_zones
from utils.market_calendar import get_market_dates
from utils.opening_audit import (
    record_opening_gap_prediction,
    evaluate_opening_gap_outcomes,
    load_saved_gap_audit_history
)
from utils.ui_theme import apply_custom_theme
from config import STOCK_NAME_MAP

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

    # Time-Gating Status Logic:
    # 3:00 PM = 15:00 = 900 min
    # 3:05 PM = 15:05 = 905 min
    # 3:30 PM = 15:30 = 930 min
    is_weekday = ist_time.weekday() < 5
    is_live_trading_window = is_weekday and (905 <= total_minutes <= 930)
    is_prep_window = is_weekday and (900 <= total_minutes < 905)
    is_market_closed = (not is_weekday) or (total_minutes > 930) or (total_minutes < 555)

    # Time Gate Banner & Mode Override
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        if is_live_trading_window:
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
                "Reviewing latest closing session data. Forecast indicates expected 9:15 AM opening move for the next market day."
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
    watchlist = st.session_state.get("watchlist", ["CDSL.NS", "NSDL.BO"])
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        selected_symbol = st.selectbox(
            "Select Stock for Opening Gap Prediction",
            options=watchlist,
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
    st.subheader(f"📅 Target Market Open: {dates_info['next_date_str']} (9:15 AM – 9:20 AM IST)")
    st.caption("Predicting overnight gap direction: Will tomorrow's 9:15 AM Open be GREATER than today's 3:30 PM Close?")

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

    # 2. Train and Predict Opening Gap
    with st.spinner(f"Computing 3:05 PM Opening Gap Ensemble Model for {selected_symbol}..."):
        result = predict_opening_gap(selected_symbol, period=period)

    if result.get("status") != "success":
        st.error(f"Opening gap prediction failed: {result.get('message')}")
        return

    # Automatically Record Prediction into Opening Gap Audit
    record_opening_gap_prediction(selected_symbol, dates_info['next_date_str'], result)

    # 3. Forecast Result Cards
    options_call = result.get("options_call", {})
    action_text = options_call.get("action", "HOLD")
    bias_color = options_call.get("bias_color", "#38bdf8")

    st.subheader(f"🎯 3:05 PM Gap Forecast & Put/Call Direction")

    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    with col_g1:
        st.metric("Expected Opening Gap", result['direction'])
    with col_g2:
        st.metric("Gap Up Probability", f"{result['probability_up_pct']}%")
    with col_g3:
        st.metric("Gap Down Probability", f"{result['probability_down_pct']}%")
    with col_g4:
        st.metric("Confidence Rating", result['confidence'])

    # 4. ACTIONABLE OPTIONS TRADING CARD (The User's Primary Tool)
    st.markdown("### ⚡ Actionable Put / Call Options Entry Blueprint")
    with st.container(border=True):
        st.markdown(
            f"<div style='background: {bias_color}22; border-left: 6px solid {bias_color}; padding: 12px 18px; border-radius: 8px; margin-bottom: 12px;'>"
            f"<h2 style='margin:0; color: {bias_color}; font-size: 1.6rem;'>{action_text} — {options_call.get('strategy', 'Overnight Strategy')}</h2>"
            f"<p style='margin: 4px 0 0 0; color: #cbd5e1; font-size: 0.95rem;'>Targeting tomorrow's 9:15 AM opening gap move on <strong>{selected_symbol}</strong> (LTP: ₹{result['current_price']:,.2f})</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        o1, o2, o3, o4 = st.columns(4)
        o1.metric("🎯 Recommended Strike", options_call.get("suggested_strike", "N/A"))
        o2.metric("🛡️ Alternative / Safe Strike", options_call.get("alt_strike", "N/A"))
        o3.metric("⏰ Options Entry Window", options_call.get("entry_window", "3:10 PM - 3:20 PM"))
        o4.metric("🏁 Options Exit Window", options_call.get("exit_window", "9:15 AM - 9:20 AM"))

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

    # 6. OPENING GAP AUDIT TRAIL & HIT RATE
    st.subheader("🕵️ 3:05 PM Opening Gap Prediction vs Actual 9:15 AM Open Audit")
    st.caption("Dedicated audit ledger tracking whether the predicted opening gap matched tomorrow's actual 9:15 AM open price.")

    # Evaluate completed sessions
    gap_audit_history = evaluate_opening_gap_outcomes()

    if gap_audit_history:
        completed = [a for a in gap_audit_history if a.get("is_correct") is not None]
        correct_count = sum(1 for a in completed if a.get("is_correct") is True)
        total_completed = len(completed)
        hit_rate_pct = round((correct_count / total_completed * 100.0), 1) if total_completed > 0 else 0.0

        ga1, ga2, ga3, ga4 = st.columns(4)
        ga1.metric("Opening Gap Hit Rate", f"{hit_rate_pct}%" if total_completed > 0 else "Pending Data")
        ga2.metric("Audited Sessions", f"{total_completed} Sessions")
        ga3.metric("Verified Gap Hits", f"✅ {correct_count}")
        ga4.metric("Gap Divergences", f"❌ {total_completed - correct_count}")

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
                "Verification": "✅ Verified Hit" if a.get("is_correct") is True else ("❌ Diverged" if a.get("is_correct") is False else "⏳ Awaiting 9:15 AM Open")
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
