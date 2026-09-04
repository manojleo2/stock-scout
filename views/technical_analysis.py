import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from utils.data_loader import get_stock_data
from utils.indicators import calculate_technical_indicators, generate_composite_signals
from utils.ui_theme import apply_custom_theme
from config import STOCK_NAME_MAP

def render_technical_analysis_page():
    apply_custom_theme()

    st.markdown("<div class='glowing-header'>📈 Interactive Technical Charts</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>Multi-Panel TradingView Candlesticks, Moving Averages, RSI Oscillators & MACD</div>", unsafe_allow_html=True)

    watchlist = st.session_state.get("watchlist", ["CDSL.NS", "NSDL.BO"])

    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
        selected_symbol = st.selectbox(
            "Select Stock for Analysis",
            options=watchlist,
            format_func=lambda s: f"{STOCK_NAME_MAP.get(s, s)} ({s})"
        )
    with col_s2:
        period = st.selectbox("Timeframe", options=["3m", "6m", "1y", "2y"], index=2)
    with col_s3:
        show_macd = st.checkbox("Show MACD Panel", value=True)

    if not selected_symbol:
        st.warning("Please select a stock.")
        return

    # Fetch and process data
    df_raw = get_stock_data(selected_symbol, period=period)
    if df_raw.empty:
        st.error(f"Could not load data for {selected_symbol}.")
        return

    df = calculate_technical_indicators(df_raw)

    # 1. Composite Indicator Signals Summary Banner
    signals = generate_composite_signals(df)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Technical Summary Signal", signals['summary'])
    with c2:
        st.metric("Bullish Indicators", f"🟢 {signals['bullish_count']}")
    with c3:
        st.metric("Bearish Indicators", f"🔴 {signals['bearish_count']}")
    with c4:
        st.metric("Neutral Indicators", f"⚪ {signals['neutral_count']}")

    # 2. Multi-panel Plotly Chart Construction
    rows = 4 if show_macd else 3
    row_heights = [0.5, 0.15, 0.15, 0.2] if show_macd else [0.6, 0.2, 0.2]
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=(
            f"{selected_symbol} Price & Technical Overlays",
            "Volume",
            "RSI (14)",
            "MACD (12, 26, 9)"
        ) if show_macd else (
            f"{selected_symbol} Price & Technical Overlays",
            "Volume",
            "RSI (14)"
        )
    )

    # Row 1: Candlestick + Moving Averages + Bollinger Bands
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="OHLC",
        increasing_line_color='#00E676', decreasing_line_color='#FF5252'
    ), row=1, col=1)

    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='#FFB300', width=1.5), name='SMA 50'), row=1, col=1)
    if 'SMA_200' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='#00E5FF', width=1.5), name='SMA 200'), row=1, col=1)
    if 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='#90A4AE', dash='dash'), name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='#90A4AE', dash='dash'), name='BB Lower'), row=1, col=1)

    # Row 2: Volume Bar Chart
    colors = ['#00E676' if c >= o else '#FF5252' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)

    # Row 3: RSI (14)
    if 'RSI_14' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], line=dict(color='#B388FF', width=1.5), name="RSI"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#FF5252", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00E676", row=3, col=1)

    # Row 4: MACD Panel (Optional)
    if show_macd and 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#00E5FF', width=1.5), name="MACD"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='#FF4081', width=1.5), name="Signal"), row=4, col=1)
        
        hist_colors = ['#00E676' if h >= 0 else '#FF5252' for h in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=hist_colors, name="Histogram"), row=4, col=1)

    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(
        height=900 if show_macd else 750,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor='#0e131f',
        plot_bgcolor='#111827',
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 3. Detailed Signal Breakdown Table
    st.markdown("### 📋 Indicator Breakdown Matrix")
    if signals.get('details'):
        df_details = pd.DataFrame(signals['details'])
        st.dataframe(df_details, use_container_width=True, hide_index=True)

if __name__ == "__main__" or True:
    render_technical_analysis_page()
