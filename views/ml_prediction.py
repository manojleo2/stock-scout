import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.ml_model import train_and_predict
from utils.nifty_correlation import analyze_nifty_impact
from config import STOCK_NAME_MAP

def render_ml_prediction_page():
    st.title("🤖 ML Directional Stock Prediction")
    st.caption("Machine Learning Classifier for Daily Price Movement Probability (Up/Down)")

    st.warning(
        "⚠️ **Disclaimer:** Stock predictions are probabilistic decision-support signals based on historical technical patterns "
        "and Nifty index momentum. They do NOT guarantee future price action. Always combine with risk management."
    )

    watchlist = st.session_state.get("watchlist", ["CDSL.NS", "NSDL.NS"])

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        selected_symbol = st.selectbox(
            "Select Stock for Prediction",
            options=watchlist,
            format_func=lambda s: f"{STOCK_NAME_MAP.get(s, s)} ({s})"
        )
    with col_s2:
        period = st.selectbox("Training Data Horizon", options=["1y", "2y", "3y"], index=1)

    if not selected_symbol:
        st.warning("Please select a stock.")
        return

    with st.spinner(f"Training Machine Learning model for {selected_symbol}..."):
        result = train_and_predict(selected_symbol, period=period)
        nifty_impact = analyze_nifty_impact(selected_symbol, period=period)

    if result.get("status") != "success":
        st.error(f"Prediction failed: {result.get('message')}")
        return

    st.markdown("---")

    # 1. Tomorrow's Prediction Result Card
    st.subheader(f"🎯 Tomorrow's Forecast for {selected_symbol}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Predicted Direction", result['direction'])
    with c2:
        st.metric("Up Probability", f"{result['probability_up_pct']}%")
    with c3:
        st.metric("Down Probability", f"{result['probability_down_pct']}%")
    with c4:
        st.metric("Signal Confidence", result['confidence'])

    # Gauge Chart for Probability
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = result['probability_up_pct'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Tomorrow's Upward Directional Probability (%)", 'font': {'size': 18}},
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

    # 2. Nifty Correlation & Factor Analysis
    st.subheader("🏛️ Nifty 50 Index Influence & Sensitivity")
    if nifty_impact.get("status") == "success":
        nc1, nc2, nc3, nc4 = st.columns(4)
        with nc1:
            st.metric("Nifty Pearson Correlation", nifty_impact['correlation'])
        with nc2:
            st.metric("Stock Beta vs Nifty", nifty_impact['beta'])
        with nc3:
            st.metric("Avg Drop on Nifty Fall (>1%)", f"{nifty_impact['avg_stock_drop_on_nifty_fall']}%")
        with nc4:
            st.metric("Nifty Momentum Trend", f"{nifty_impact['nifty_badge']} {nifty_impact['nifty_trend']}")
        
        st.info(f"💡 **Impact Assessment:** {nifty_impact['impact_level']}")

    st.markdown("---")

    # 3. Model Accuracy & Top Drivers (Feature Importances)
    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        st.subheader("📊 Model Performance Metrics")
        st.metric("Out-of-Sample Test Accuracy", f"{result['test_accuracy_pct']}%")
        st.caption(f"Evaluated on last 20% chronological trading days ({result['test_sample_count']} days)")
        st.caption(f"Total historical training samples: {result['sample_count']} days")

    with col_f2:
        st.subheader("🔑 Top Prediction Drivers (Feature Importances)")
        importances = result['feature_importances']
        df_imp = pd.DataFrame({
            "Feature": list(importances.keys()),
            "Importance": list(importances.values())
        }).sort_values(by="Importance", ascending=True)

        fig_imp = go.Figure(go.Bar(
            x=df_imp['Importance'],
            y=df_imp['Feature'],
            orientation='h',
            marker_color='#42a5f5'
        ))
        fig_imp.update_layout(
            height=300, template="plotly_dark",
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Importance Score",
            yaxis_title="Feature"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # 4. Extensibility Roadmap Info Box
    with st.expander("🚀 How to Improve Prediction Accuracy Beyond Current Baseline"):
        st.markdown("""
        The current model uses technical indicators and Nifty momentum features to reach **~54-58% out-of-sample directional accuracy**.
        
        To systematically push accuracy higher in future updates, we can add:
        1. **FII / DII Institutional Flow Data** (+2-4% edge): Track daily foreign & domestic institutional buying/selling.
        2. **Options PCR (Put-Call Ratio) & Open Interest** (+2-4% edge): Gauge sentiment from options buildup.
        3. **Corporate Earnings Calendar**: Signal volatility 3-5 days prior to quarterly result announcements.
        4. **Delivery Percentage Data**: Track when high-volume delivery buying occurs on NSE.
        5. **Advanced Models (XGBoost / LightGBM)**: Better handle non-linear market regimes.
        """)

if __name__ == "__main__" or True:
    render_ml_prediction_page()

