import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from utils.data_loader import get_stock_data
from utils.indicators import calculate_technical_indicators, generate_composite_signals
from utils.gap_analysis import analyze_intraday_gap_and_zones
from utils.ui_theme import apply_custom_theme
from config import STOCK_NAME_MAP

def render_technical_analysis_page():
    apply_custom_theme()

    st.markdown("<div class='glowing-header'>📈 Interactive Technical Analysis & Intraday Gap Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>Real-Time VWAP, 15-Min Opening Range Breakouts, Buy/Sell Zones & Multi-Panel Charts</div>", unsafe_allow_html=True)

    watchlist = st.session_state.get("watchlist", ["CDSL.NS", "NSDL.BO"])

    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
        selected_symbol = st.selectbox(
            "Select Stock for Technical & Intraday Analysis",
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

    # 1. Run Intraday Gap & Buy/Sell Zone Analysis
    gap_info = analyze_intraday_gap_and_zones(selected_symbol)

    if gap_info.get("status") == "success":
        st.markdown("### ⚡ 9:15 AM Intraday Gap & Momentum Decision Engine")
        
        # 1. High-Level Price & Market State Metrics
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.metric(
                "9:15 AM Opened Price",
                f"₹{gap_info.get('open_price', 'N/A')}",
                f"{'+' if gap_info.get('opening_gap_rs', 0) >= 0 else ''}{gap_info.get('opening_gap_rs', 0)} ({gap_info.get('opening_gap_pct', 0)}% Gap)"
            )
        with g2:
            st.metric(
                "Current Trading Price (LTP)",
                f"₹{gap_info.get('current_price', 'N/A')}",
                f"{'+' if gap_info.get('day_change_rs', 0) >= 0 else ''}{gap_info.get('day_change_rs', 0)} ({gap_info.get('day_change_pct', 0)}% Day)"
            )
        with g3:
            st.metric(
                "Intraday VWAP (Avg Price)",
                f"₹{gap_info.get('vwap', 'N/A')}",
                f"{gap_info.get('vwap_diff_pct', 0)}% vs VWAP"
            )
        with g4:
            st.metric(
                "15-Min Opening Range",
                f"₹{gap_info.get('orb_low', 'N/A')} - ₹{gap_info.get('orb_high', 'N/A')}"
            )

        # 2. Real-Time Actionable Verdict & Summary Banner
        v_color = gap_info.get("verdict_color", "#38bdf8")
        with st.container(border=True):
            st.markdown(f"""
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div style='display:flex; align-items:center; gap:10px;'>
                        <span style='font-size:1.4rem;'>🧭</span>
                        <div>
                            <h4 style='margin:0; font-weight:700; color:{v_color};'>Intraday Momentum: {gap_info.get('gap_signal')}</h4>
                            <span style='color:#94a3b8; font-size:0.85rem;'>Evaluated using real-time 15-minute order flow & Volume Weighted Average Price (VWAP)</span>
                        </div>
                    </div>
                    <span class='badge-up' style='border-color:{v_color}; color:{v_color}!important; font-size:0.95rem; font-weight:700; padding:6px 14px;'>
                        CURRENT ACTION: {gap_info.get('active_verdict', 'WAIT')}
                    </span>
                </div>
                <div style='background:rgba(15, 23, 42, 0.6); padding:12px; border-radius:8px; margin-top:12px; border-left:4px solid {v_color};'>
                    <p style='color:#f1f5f9; font-size:0.92rem; margin:0 0 6px 0; line-height:1.4;'>
                        <b>📌 Market Executive Summary:</b> {gap_info.get('gap_explanation')}
                    </p>
                    <div style='display:flex; justify-content:space-between; align-items:center; margin-top:8px; font-size:0.88rem;'>
                        <span style='color:#cbd5e1;'>🎯 <b>Actionable Advice:</b> {gap_info.get('recommendation')}</span>
                        <span style='color:#cbd5e1;'>⚖️ <b>Risk : Reward Ratio:</b> <strong style='color:#00E676;'>{gap_info.get('risk_reward_ratio')}</strong></span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # 3. Crystal Clear "When to Buy, When to Hold, When to Sell" Decision Matrix
        st.markdown("#### 🎯 Execution Matrix: Exactly When to Buy, Hold & Sell")
        pb = gap_info.get("playbook", {})
        
        c_buy, c_hold, c_sell = st.columns(3)
        
        with c_buy:
            with st.container(border=True):
                st.markdown(f"""
                    <div style='color:#00E676; font-weight:700; font-size:1.05rem; margin-bottom:8px;'>
                        🟢 WHEN TO BUY (Long Trigger)
                    </div>
                    <p style='color:#cbd5e1; font-size:0.85rem; margin-bottom:10px;'>
                        <b>Condition:</b> {pb.get('buy', {}).get('condition', 'Breakout above 15m High & VWAP')}
                    </p>
                    <div style='background:rgba(0, 230, 118, 0.1); padding:8px 10px; border-radius:6px; font-size:0.83rem;'>
                        <div><b>Entry Zone:</b> <code style='color:#00E676;'>{pb.get('buy', {}).get('entry_zone')}</code></div>
                        <div style='margin-top:4px;'><b>Target 1:</b> {pb.get('buy', {}).get('target_1')}</div>
                        <div style='margin-top:4px;'><b>Target 2:</b> {pb.get('buy', {}).get('target_2')}</div>
                        <div style='margin-top:4px;'><b>Stop Loss:</b> <code style='color:#FF5252;'>{pb.get('buy', {}).get('stop_loss')}</code></div>
                    </div>
                """, unsafe_allow_html=True)

        with c_hold:
            with st.container(border=True):
                st.markdown(f"""
                    <div style='color:#FFB300; font-weight:700; font-size:1.05rem; margin-bottom:8px;'>
                        🟡 WHEN TO HOLD (Neutral Zone)
                    </div>
                    <p style='color:#cbd5e1; font-size:0.85rem; margin-bottom:10px;'>
                        <b>Condition:</b> {pb.get('hold', {}).get('condition', 'Price consolidating inside opening range')}
                    </p>
                    <div style='background:rgba(255, 179, 0, 0.1); padding:8px 10px; border-radius:6px; font-size:0.83rem;'>
                        <div><b>Action:</b> {pb.get('hold', {}).get('action')}</div>
                        <div style='margin-top:6px;'><b>Safe Range:</b> <code>₹{gap_info.get('orb_low')} - ₹{gap_info.get('orb_high')}</code></div>
                    </div>
                """, unsafe_allow_html=True)

        with c_sell:
            with st.container(border=True):
                st.markdown(f"""
                    <div style='color:#FF5252; font-weight:700; font-size:1.05rem; margin-bottom:8px;'>
                        🔴 WHEN TO SELL / SHORT / PUT
                    </div>
                    <p style='color:#cbd5e1; font-size:0.85rem; margin-bottom:10px;'>
                        <b>Condition:</b> {pb.get('sell', {}).get('condition', 'Price trades below VWAP or 15m Low')}
                    </p>
                    <div style='background:rgba(255, 82, 82, 0.1); padding:8px 10px; border-radius:6px; font-size:0.83rem;'>
                        <div><b>Action:</b> <code style='color:#FF5252;'>{pb.get('sell', {}).get('action')}</code></div>
                        <div style='margin-top:4px;'><b>Downside Target 1:</b> {pb.get('sell', {}).get('downside_target_1')}</div>
                        <div style='margin-top:4px;'><b>Downside Target 2:</b> {pb.get('sell', {}).get('downside_target_2')}</div>
                        <div style='margin-top:4px;'><b>Stop Loss:</b> <code style='color:#00E676;'>{pb.get('sell', {}).get('stop_loss')}</code></div>
                    </div>
                """, unsafe_allow_html=True)

        # 4. Target Levels Reference Row
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Current Price (LTP)", f"₹{gap_info['current_price']}")
        t2.metric("Intraday Target 1", f"₹{gap_info['suggested_target_1']}")
        t3.metric("Intraday Target 2", f"₹{gap_info['suggested_target_2']}")
        t4.metric("Invalidation Stop Loss", f"₹{gap_info['suggested_stop_loss']}")

    st.markdown("---")

    # Fetch and process daily technical data
    df_raw = get_stock_data(selected_symbol, period=period)
    if df_raw.empty:
        st.error(f"Could not load historical data for {selected_symbol}.")
        return

    df = calculate_technical_indicators(df_raw)

    # 2. Composite Indicator Signals Summary Banner
    signals = generate_composite_signals(df)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Technical Summary Signal", signals['summary'])
    with c2:
        st.metric("Bullish Indicators", f"🟢 {signals['bullish_count']}")
    with c3:
        st.metric("Bearish Indicators", f"🔴 {signals['bearish_count']}")
    with c4:
        st.metric("Neutral Indicators", f"⚪ {signals['neutral_count']}")

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Multi-panel Plotly Chart Construction
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

    # 4. Detailed Signal Breakdown Table
    st.markdown("### 📋 Indicator Breakdown Matrix")
    if signals.get('details'):
        df_details = pd.DataFrame(signals['details'])
        st.dataframe(df_details, use_container_width=True, hide_index=True)

if __name__ == "__main__" or True:
    render_technical_analysis_page()
