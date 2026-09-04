import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.data_loader import get_stock_data
from utils.ml_model import train_and_predict
from utils.nifty_correlation import analyze_nifty_impact
from utils.macro_factors import get_latest_macro_summary
from utils.market_calendar import get_market_dates, get_daily_ups_downs_history
from config import STOCK_NAME_MAP

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

    st.markdown("---")

    # 2. Real-time Global & Volatility Macro Banner
    st.subheader("🌐 Overnight Global Cues & Volatility Climate")
    macro = get_latest_macro_summary()
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        sp_val = macro['sp500_change_pct']
        st.metric("S&P 500 (US Market)", f"{sp_val}%", delta=f"{sp_val}%", delta_color="normal")
    with col_m2:
        vix_val = macro['vix_level']
        st.metric("India VIX (Market Volatility)", f"{vix_val}", delta=f"{macro['vix_change_pct']}%", delta_color="inverse")
    with col_m3:
        st.metric("Market Volatility Regime", f"{macro['vix_badge']} {macro['vix_status']}")

    st.markdown("---")

    # Train ML Model
    with st.spinner(f"Training Ensemble ML model for {selected_symbol}..."):
        result = train_and_predict(selected_symbol, period=period)
        nifty_impact = analyze_nifty_impact(selected_symbol, period=period)

    if result.get("status") != "success":
        st.error(f"Prediction failed: {result.get('message')}")
        return

    # 3. Target Date Forecast Result Card
    st.subheader(f"🎯 Prediction for Next Trading Session ({dates_info['next_date_str']})")
    st.caption(f"Stock: **{STOCK_NAME_MAP.get(selected_symbol, selected_symbol)}** | Last Close: **₹{result['latest_close']}** on {dates_info['last_date_str']}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Forecasted Direction", result['direction'])
    with c2:
        st.metric(f"Up Probability ({dates_info['next_date'].strftime('%a, %b %d')})", f"{result['probability_up_pct']}%")
    with c3:
        st.metric(f"Down Probability ({dates_info['next_date'].strftime('%a, %b %d')})", f"{result['probability_down_pct']}%")
    with c4:
        st.metric("Signal Confidence", result['confidence'])

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

    st.markdown("---")

    # 4. Date-Wise Daily Ups & Downs History Log
    st.subheader(f"🗓️ Date-Wise Daily Ups & Downs History ({selected_symbol})")
    st.caption("Historical day-by-day closing prices, daily movements, and volume trends")

    max_hist_days = st.slider("Historical Trading Days to Display", min_value=10, max_value=60, value=20)
    df_history = get_daily_ups_downs_history(df_raw, max_days=max_hist_days)

    if not df_history.empty:
        st.dataframe(df_history, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 5. Nifty Sensitivity & Feature Drivers
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

if __name__ == "__main__" or True:
    render_ml_prediction_page()
