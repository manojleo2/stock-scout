import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from utils.data_loader import get_stock_data
from utils.indicators import calculate_technical_indicators, generate_composite_signals
from config import STOCK_NAME_MAP

def render_technical_analysis_page():
    st.title("📈 Interactive Technical Analysis & Charts")
    st.caption("Detailed Candlesticks, Moving Averages, RSI, MACD, and Bollinger Bands")

    watchlist = st.session_state.get("watchlist", ["CDSL.NS", "NSDL.NS"])

    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
        selected_symbol = st.selectbox(
            "Select Stock to Analyze",
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
    
    st.markdown("---")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Technical Summary", signals['summary'])
    with c2:
        st.metric("Bullish Signals", f"🟢 {signals['bullish_count']}")
    with c3:
        st.metric("Bearish Signals", f"🔴 {signals['bearish_count']}")
    with c4:
        st.metric("Neutral Signals", f"⚪ {signals['neutral_count']}")

    # 2. Multi-panel Plotly Chart Construction
    rows = 4 if show_macd else 3
    row_heights = [0.5, 0.15, 0.15, 0.2] if show_macd else [0.6, 0.2, 0.2]
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=(
            f"{selected_symbol} Price & Overlays",
            "Volume",
            "RSI (14)",
            "MACD (12, 26, 9)"
        ) if show_macd else (
            f"{selected_symbol} Price & Overlays",
            "Volume",
            "RSI (14)"
        )
    )

    # Row 1: Candlestick + Moving Averages + Bollinger Bands
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="OHLC"
    ), row=1, col=1)

    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1.5), name='SMA 50'), row=1, col=1)
    if 'SMA_200' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='deepskyblue', width=1.5), name='SMA 200'), row=1, col=1)
    if 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', dash='dash'), name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', dash='dash'), name='BB Lower'), row=1, col=1)

    # Row 2: Volume Bar Chart
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)

    # Row 3: RSI (14)
    if 'RSI_14' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], line=dict(color='purple', width=1.5), name="RSI"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # Row 4: MACD Panel (Optional)
    if show_macd and 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='cyan', width=1.5), name="MACD"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='magenta', width=1.5), name="Signal"), row=4, col=1)
        
        hist_colors = ['#26a69a' if h >= 0 else '#ef5350' for h in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=hist_colors, name="Histogram"), row=4, col=1)

    # Remove weekend trading gaps on Indian markets
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(
        height=900 if show_macd else 750,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 3. Detailed Signal Breakdown Table
    st.subheader("📋 Indicator Breakdown")
    if signals.get('details'):
        df_details = pd.DataFrame(signals['details'])
        st.dataframe(df_details, use_container_width=True, hide_index=True)

if __name__ == "__main__" or True:
    render_technical_analysis_page()

