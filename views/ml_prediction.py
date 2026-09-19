import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.data_loader import get_stock_data
from utils.ml_model import train_and_predict
from utils.nifty_correlation import analyze_nifty_impact
from utils.macro_factors import get_latest_macro_summary
from utils.market_calendar import get_market_dates, get_daily_ups_downs_history
from utils.prediction_audit import (
    record_prediction, evaluate_and_update_audit_outcomes, load_saved_audit_history
)
from utils.ui_theme import apply_custom_theme
from config import STOCK_NAME_MAP, AI_MIN_CONVICTION_THRESHOLD

def render_ml_prediction_page():
    apply_custom_theme()

    st.markdown("<div class='glowing-header'>🤖 AI & Ensemble Directional Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>Multi-Factor ML Classifier (RandomForest + Gradient Boosting + Global Cues + India VIX)</div>", unsafe_allow_html=True)

    st.warning(
        "⚠️ **Disclaimer:** Stock predictions are probabilistic decision-support signals based on technical indicators, "
        "Nifty momentum, India VIX volatility, and overnight global cues. They do NOT guarantee future price action."
    )

    watchlist = st.session_state.get("watchlist", ["CDSL.NS", "NSDL.BO"])

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        selected_symbol = st.selectbox(
            "Select Stock for Prediction & History",
            options=watchlist,
            format_func=lambda s: f"{STOCK_NAME_MAP.get(s, s)} ({s})"
        )
    with col_s2:
        period = st.selectbox("Training Horizon", options=["1y", "2y", "3y"], index=1)

    if not selected_symbol:
        st.warning("Please select a stock.")
        return

    # Fetch raw data for market dates & history
    df_raw = get_stock_data(selected_symbol, period=period)
    dates_info = get_market_dates(df_raw)

    # 1. Trading Calendar & Next Date Target Banner
    st.subheader("📅 Trading Calendar & Target Session")
    
    cal1, cal2, cal3 = st.columns(3)
    with cal1:
        st.metric("Last Closed Trading Session", dates_info['last_date_str'])
    with cal2:
        st.metric("Next Market Trading Session", dates_info['next_date_str'])
    with cal3:
        st.metric("Target Forecast Date", dates_info['next_date_str'])

    if dates_info.get('holiday_alert'):
        st.warning(f"🏖️ **Market Holiday Notice:** {dates_info['holiday_alert']}")

    if dates_info.get('is_today_holiday'):
        st.info(f"🏖️ **Today is an official NSE/BSE Holiday ({dates_info['today_holiday_name']})**. The market is closed today. Predictions displayed below apply to the next active session ({dates_info['next_date_str']}).")

    st.markdown("---")

    # 2. Train / Retrieve Model Prediction (Instant via @st.cache_data)
    with st.spinner(f"Computing Supercharged Ensemble AI model for {selected_symbol}..."):
        result = train_and_predict(selected_symbol, period=period)
        nifty_impact = analyze_nifty_impact(selected_symbol, period=period)

    if result.get("status") != "success":
        st.error(f"Prediction failed: {result.get('message')}")
        return

    # Automatically Record Current Prediction into Persistent Audit Log
    record_prediction(selected_symbol, dates_info['next_date_str'], result)

    # 3. Real-time Global & Volatility Macro Banner
    st.subheader("🌐 Overnight Global Cues, Volatility & Hourly News Bias")
    macro = get_latest_macro_summary()
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        sp_val = macro['sp500_change_pct']
        st.metric("S&P 500 (US Market)", f"{sp_val}%", delta=f"{sp_val}%", delta_color="normal")
    with col_m2:
        vix_val = macro['vix_level']
        st.metric("India VIX (Volatility)", f"{vix_val}", delta=f"{macro['vix_change_pct']}%", delta_color="inverse")
    with col_m3:
        st.metric("Volatility Regime", f"{macro['vix_badge']} {macro['vix_status']}")
    with col_m4:
        news_info = result.get("news_info", {})
        news_bias_score = news_info.get("score", 0.0)
        news_badge = news_info.get("badge", "⚪ Neutral")
        st.metric("Hourly News Sentiment Bias", f"{news_badge} ({news_bias_score:+})", f"{news_info.get('count', 0)} articles fetched")

    st.markdown("---")

    # 3. Target Date Forecast Result Card
    st.subheader(f"🎯 Prediction for Next Trading Session ({dates_info['next_date_str']})")
    st.caption(f"Stock: **{STOCK_NAME_MAP.get(selected_symbol, selected_symbol)}** | Last Close: **₹{result['latest_close']}** on {dates_info['last_date_str']}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Forecast Direction", result['direction'])
    with c2:
        st.metric("Up Probability", f"{result['probability_up_pct']}%")
    with c3:
        st.metric("Down Probability", f"{result['probability_down_pct']}%")
    with c4:
        st.metric("Confidence Level", result['confidence'])

    # Push Forecast Alert to Telegram Phone Button
    from utils.notifications import send_prediction_alert_notification
    if st.button(f"📱 Send AI Forecast for {selected_symbol} to My Phone via Telegram", use_container_width=True):
        with st.spinner("Pushing forecast alert to Telegram..."):
            stock_name = STOCK_NAME_MAP.get(selected_symbol, selected_symbol)
            success, err_msg = send_prediction_alert_notification(
                selected_symbol,
                stock_name,
                result['direction'],
                result['probability_up_pct'],
                result['confidence'],
                news_bias_score,
                dates_info['next_date_str']
            )
        if success:
            st.success("🎉 Pre-Market AI Forecast sent to your phone Telegram!")
        else:
            st.error(f"❌ Alert error: {err_msg}")

    # Gauge Chart for Probability
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = result['probability_up_pct'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Directional Probability for {dates_info['next_date_str']} (%)", 'font': {'size': 16}},
        gauge = {
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
    fig_gauge.update_layout(height=280, template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # 3b. Actionable Quant Execution Blueprint Card (Swing / Breakout)
    quant_bp = result.get("quant_blueprint", {})
    action_text = quant_bp.get("action", "HOLD")
    bias_color = quant_bp.get("bias_color", "#38bdf8")
    conviction_val = quant_bp.get("conviction_pct", 50.0)

    st.markdown("### ⚡ Actionable Quant Execution Blueprint (Swing & Breakout)")
    
    # Horizon Clarity Notice
    st.markdown(
        f"<div style='background: rgba(56, 189, 248, 0.08); border-left: 4px solid #38bdf8; padding: 8px 14px; border-radius: 6px; margin-bottom: 12px; font-size: 0.90rem;'>"
        f"🕒 <strong>Trade Horizon Clarity:</strong> Unlike the 3:05 PM Gap Predictor (Overnight Gap), this model forecasts the <strong>Full Next-Day Direction (Close-to-Close)</strong>.<br>"
        f"🧭 <em>Recommended Trade Horizon: Next-Day Intraday Breakout or 1–3 Day Swing Carry (Cash / Futures / Stock Equity).</em>"
        f"</div>",
        unsafe_allow_html=True
    )

    if conviction_val < AI_MIN_CONVICTION_THRESHOLD:
        st.markdown(
            f"<div style='background: rgba(245, 158, 11, 0.15); border-left: 6px solid #f59e0b; padding: 12px 16px; border-radius: 8px; margin-bottom: 14px;'>"
            f"<h3 style='margin:0; color: #f59e0b; font-size: 1.15rem;'>🛡️ Capital Preserved: Filter Active (&lt;{AI_MIN_CONVICTION_THRESHOLD:.0f}% Conviction)</h3>"
            f"<p style='margin: 4px 0 0 0; color: #e2e8f0; font-size: 0.92rem;'>"
            f"Current directional conviction is <strong>{conviction_val:.1f}%</strong>. Market order flow is in a low-edge consolidation regime. "
            f"<strong>100% Cash preservation advised — avoid placing swing trades in chop.</strong>"
            f"</p></div>",
            unsafe_allow_html=True
        )

    with st.container(border=True):
        st.markdown(
            f"<div style='background: {bias_color}22; border-left: 6px solid {bias_color}; padding: 12px 18px; border-radius: 8px; margin-bottom: 12px;'>"
            f"<h2 style='margin:0; color: {bias_color}; font-size: 1.5rem;'>{action_text} — {quant_bp.get('strategy', 'Strategy')}</h2>"
            f"<p style='margin: 4px 0 0 0; color: #cbd5e1; font-size: 0.92rem;'>Quantitative ATR levels for <strong>{selected_symbol}</strong> (LTP: ₹{result['latest_close']:,.2f} | 14-ATR: ₹{quant_bp.get('atr_14', 0):,.2f})</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        qc1, qc2, qc3, qc4, qc5 = st.columns(5)
        qc1.metric("🎯 Entry Level", quant_bp.get("entry_level", "N/A"))
        sl_v = quant_bp.get('stop_loss')
        qc2.metric("🛡️ Stop-Loss (1.0x ATR)", f"₹{sl_v:,.2f}" if sl_v else "N/A", f"-₹{quant_bp.get('sl_points', 0):,.2f}" if sl_v else "")
        t1_v = quant_bp.get('target_1')
        qc3.metric("🏁 Target 1 (1.5x ATR)", f"₹{t1_v:,.2f}" if t1_v else "N/A", f"+₹{quant_bp.get('tp1_points', 0):,.2f}" if t1_v else "")
        t2_v = quant_bp.get('target_2')
        qc4.metric("🚀 Target 2 (2.5x ATR)", f"₹{t2_v:,.2f}" if t2_v else "N/A", f"+₹{quant_bp.get('tp2_points', 0):,.2f}" if t2_v else "")
        qc5.metric("⚖️ Risk : Reward", quant_bp.get("risk_reward_ratio", "1.5 : 1"))

        # Microstructure & Relative Alpha Badges
        a_col1, a_col2, a_col3, a_col4 = st.columns(4)
        alpha5 = result.get('alpha_5d_pct', 0.0)
        alpha20 = result.get('alpha_20d_pct', 0.0)
        bpi = result.get('buying_pressure_index', 1.0)
        trend_c = result.get('trend_convergence', 0.0)

        with a_col1:
            st.metric("5D Alpha vs Nifty 50", f"{alpha5:+.2f}%", "Outperforming" if alpha5 > 0 else "Underperforming")
        with a_col2:
            st.metric("20D Alpha vs Nifty 50", f"{alpha20:+.2f}%", "Outperforming" if alpha20 > 0 else "Underperforming")
        with a_col3:
            st.metric("Buying Pressure Index", f"{bpi:.2f}x", "Accumulation" if bpi >= 1.0 else "Distribution")
        with a_col4:
            st.metric("Trend Alignment", "🟢 Bullish" if trend_c > 0.3 else ("🔴 Bearish" if trend_c < -0.3 else "⚪ Neutral"))

        st.caption(f"🧭 **Recommended Trade Horizon:** Next-Day Intraday Breakout or 1–3 Day Swing Carry (Cash / Futures / Stock Equity)")
        st.warning(f"⚠️ **Execution Rule:** {quant_bp.get('execution_guideline', 'Adhere to strict risk limits.')}")
        if result.get("feedback_reason"):
            st.info(f"🔄 **Continuous Learning Feedback Recalibration:** {result['feedback_reason']}")

    st.markdown("---")

    # 4. PREDICTION vs ACTUAL AUDIT LOG & ROOT CAUSE INSPECTOR
    st.subheader("🕵️ Prediction vs Actual Audit Log & Root Cause Analyzer")
    st.caption("Track historical prediction accuracy and inspect why the opposite movement occurred when a prediction diverged.")

    # Evaluate completed market sessions
    audit_history = evaluate_and_update_audit_outcomes()

    if audit_history:
        # Calculate Hit Rate Accuracy
        completed = [a for a in audit_history if a.get("is_correct") is not None]
        correct_count = sum(1 for a in completed if a.get("is_correct") is True)
        total_completed = len(completed)
        hit_rate_pct = round((correct_count / total_completed * 100.0), 1) if total_completed > 0 else 0.0

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Historical AI Hit Rate", f"{hit_rate_pct}%" if total_completed > 0 else "Pending Data")
        a2.metric("Total Predictions Audited", f"{total_completed} Days")
        a3.metric("Correct Predictions", f"✅ {correct_count}")
        a4.metric("Diverged Predictions", f"❌ {total_completed - correct_count}")

        # Summary Audit Table
        df_audit = pd.DataFrame([
            {
                "Target Date": a.get("target_date"),
                "Stock": a.get("symbol"),
                "AI Forecast": a.get("predicted_direction"),
                "Probability": f"{a.get('probability_up_pct')}%",
                "Actual Outcome": a.get("actual_direction", "Pending..."),
                "Actual Change": f"{'+' if (a.get('actual_change_pct') or 0)>=0 else ''}{a.get('actual_change_pct')}%" if a.get("actual_change_pct") is not None else "Pending...",
                "Status": "✅ Verified Hit" if a.get("is_correct") is True else ("❌ Diverged" if a.get("is_correct") is False else "⏳ Awaiting Session Close")
            } for a in reversed(audit_history)
        ])
        st.dataframe(df_audit, use_container_width=True, hide_index=True)

        # Root Cause Inspector for Missed Predictions
        diverged_list = [a for a in reversed(audit_history) if a.get("is_correct") is False]
        if diverged_list:
            with st.expander("🔍 Inspect Root Cause: Why Did the Opposite Happen?", expanded=True):
                selected_audit_date = st.selectbox(
                    "Select Diverged Prediction Date to Inspect",
                    options=[f"{a['target_date']} - {a['symbol']} (Predicted {a['predicted_direction']}, Actual {a['actual_direction']})" for a in diverged_list]
                )
                
                # Match selected record
                target_rec = next((a for a in diverged_list if f"{a['target_date']} - {a['symbol']}" in selected_audit_date), None)
                if target_rec:
                    st.markdown(f"#### 🧐 Root Cause Post-Mortem Analysis for `{target_rec['symbol']}` on {target_rec['target_date']}")
                    st.markdown(f"- **AI Forecast:** `{target_rec['predicted_direction']}` ({target_rec['probability_up_pct']}% Probability)")
                    st.markdown(f"- **Actual Market Outcome:** `{target_rec['actual_direction']}` ({target_rec['actual_change_pct']}% Change)")
                    st.markdown("##### Key Divergence Drivers & Parameter Factors:")

                    for r in target_rec.get("divergence_reasons", []):
                        st.markdown(r)

                    if target_rec.get("top_features"):
                        st.markdown("##### Top Parameter Factors Evaluated on Prediction Date:")
                        df_feat = pd.DataFrame(target_rec["top_features"], columns=["Parameter Factor", "Importance Weight"])
                        st.dataframe(df_feat, use_container_width=True, hide_index=True)

        # 4b. Deep Daily Market Journal & Granular Snapshot Explorer
        from utils.daily_journal import load_daily_journal
        journal = load_daily_journal()
        if journal:
            with st.expander("🔬 Deep Daily Market Journal & Granular Parameter Snapshot", expanded=True):
                j_symbol_entries = [e for e in journal if e.get("symbol") == selected_symbol and e.get("open") is not None]
                if j_symbol_entries:
                    selected_j_date = st.selectbox(
                        "Select Session Journal Date to Inspect",
                        options=[f"{e['date']} - Open: ₹{e['open']}, High: ₹{e['high']}, Low: ₹{e['low']}, Close: ₹{e['close']} ({e['actual_direction']})" for e in reversed(j_symbol_entries)]
                    )
                    j_rec = next((e for e in j_symbol_entries if f"{e['date']} -" in selected_j_date), j_symbol_entries[-1])
                    if j_rec:
                        st.markdown(f"#### 📖 Daily Market Activity Snapshot: `{j_rec['symbol']}` on {j_rec['date']}")
                        
                        m_o1, m_o2, m_o3, m_o4, m_o5 = st.columns(5)
                        m_o1.metric("Open Price", f"₹{j_rec.get('open', 0):,.2f}")
                        m_o2.metric("Intraday High", f"₹{j_rec.get('high', 0):,.2f}")
                        m_o3.metric("Intraday Low", f"₹{j_rec.get('low', 0):,.2f}")
                        m_o4.metric("Close Price", f"₹{j_rec.get('close', 0):,.2f}", f"{j_rec.get('day_change_pct', 0):+.2f}%")
                        m_o5.metric("Volume Surge", f"{j_rec.get('vol_vs_10d_sma', 1.0)}x avg", f"{j_rec.get('volume', 0):,} shares")

                        c_w1, c_w2, c_w3 = st.columns(3)
                        c_w1.metric("Lower Wick (Support Defense)", f"{j_rec.get('lower_wick_pct', 0)}%")
                        c_w2.metric("Candle Body", f"{j_rec.get('body_pct', 0)}%")
                        c_w3.metric("Upper Wick (Profit Rejection)", f"{j_rec.get('upper_wick_pct', 0)}%")

                        st.markdown(f"**Tags & Classification:** `{'`, `'.join(j_rec.get('divergence_tags', []))}`")
                        st.info(j_rec.get("post_mortem_narrative", "No post-mortem narrative."))
    else:
        st.info("Predictions are being logged. As trading sessions complete, historical accuracy and root-cause analyses will automatically populate here.")

    st.markdown("---")

    # 5. Date-Wise Daily Ups & Downs History Log
    st.subheader(f"🗓️ Date-Wise Daily Ups & Downs History ({selected_symbol})")
    st.caption("Historical day-by-day closing prices, daily movements, and volume trends")

    max_hist_days = st.slider("Historical Trading Days to Display", min_value=10, max_value=60, value=20)
    df_history = get_daily_ups_downs_history(df_raw, max_days=max_hist_days)

    if not df_history.empty:
        st.dataframe(df_history, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 6. Nifty Sensitivity & Feature Drivers
    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        st.subheader("📊 Model Validation Metrics")
        st.metric("Out-of-Sample Test Accuracy", f"{result['test_accuracy_pct']}%")
        
        m_p1, m_p2 = st.columns(2)
        m_p1.metric("Precision Score", f"{result.get('precision_pct', 'N/A')}%")
        m_p2.metric("Recall Score", f"{result.get('recall_pct', 'N/A')}%")

        st.caption(f"Ensemble Model: RandomForest + HistGradientBoosting Classifier")
        st.caption(f"Evaluated on last 20% chronological trading days ({result['test_sample_count']} days)")

    with col_f2:
        st.subheader("🔑 Top Prediction Drivers")
        importances = result['feature_importances']
        df_imp = pd.DataFrame({
            "Feature": list(importances.keys()),
            "Importance": list(importances.values())
        }).sort_values(by="Importance", ascending=True).tail(10)

        fig_imp = go.Figure(go.Bar(
            x=df_imp['Importance'],
            y=df_imp['Feature'],
            orientation='h',
            marker_color='#42a5f5'
        ))
        fig_imp.update_layout(
            height=320, template="plotly_dark",
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Importance Score",
            yaxis_title="Feature Factor"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

if __name__ == "__main__":
    render_ml_prediction_page()
