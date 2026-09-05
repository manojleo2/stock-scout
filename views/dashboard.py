import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.data_loader import get_stock_fundamentals, get_stock_data
from utils.nifty_correlation import analyze_nifty_impact
from utils.demat_analytics import render_demat_analytics_widget
from utils.ui_theme import apply_custom_theme
from config import BENCHMARK_TICKER, STOCK_NAME_MAP

@st.fragment(run_every="5s")
def render_live_stock_cards_fragment(watchlist: list):
    """
    Auto-refreshes live stock cards and Nifty index every 5 seconds during market hours.
    """
    # 1. Benchmark Index Glass Card (Nifty 50)
    st.markdown("### 🏛️ Market Benchmark Index (Nifty 50)")
    nifty_fund = get_stock_fundamentals(BENCHMARK_TICKER)
    
    col_n1, col_n2, col_n3, col_n4 = st.columns(4)
    with col_n1:
        n_price = nifty_fund.get("Current Price", "N/A")
        n_change = nifty_fund.get("Day Change", "N/A")
        n_pct = nifty_fund.get("Day Change %", "N/A")
        st.metric("Nifty 50 Index", f"₹{n_price}" if n_price != "N/A" else "N/A", f"{n_change} ({n_pct}%)")
    with col_n2:
        st.metric("52-Week High", f"₹{nifty_fund.get('52W High', 'N/A')}")
    with col_n3:
        st.metric("52-Week Low", f"₹{nifty_fund.get('52W Low', 'N/A')}")
    with col_n4:
        if watchlist:
            nifty_impact = analyze_nifty_impact(watchlist[0])
            if nifty_impact.get("status") == "success":
                st.metric(
                    "Nifty Trend Regime", 
                    f"{nifty_impact['nifty_badge']} {nifty_impact['nifty_trend']}",
                    f"Beta: {nifty_impact['beta']}"
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Watchlist Live Stock Cards
    st.markdown("### 📌 Monitored Watchlist Stocks (Live 5s Auto-Refresh)")

    if not watchlist:
        st.info("Your watchlist is empty. Add stocks from the sidebar!")
        return

    for i in range(0, len(watchlist), 2):
        cols = st.columns(2)
        batch = watchlist[i:i+2]
        
        for idx, symbol in enumerate(batch):
            with cols[idx]:
                fund = get_stock_fundamentals(symbol)
                name = fund.get("Name", symbol)
                price = fund.get("Current Price", "N/A")
                chg = fund.get("Day Change", 0.0)
                chg_pct = fund.get("Day Change %", 0.0)

                badge_class = "badge-up" if chg >= 0 else "badge-down"
                badge_text = f"▲ +{chg} (+{chg_pct}%)" if chg >= 0 else f"▼ {chg} ({chg_pct}%)"

                with st.container(border=True):
                    st.markdown(f"""
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <h3 style='margin:0; font-weight:700;'>{name}</h3>
                            <span class='{badge_class}'>{badge_text}</span>
                        </div>
                        <p style='color:#94a3b8; font-size:0.85rem; margin-bottom:15px;'>Ticker: <code>{symbol}</code></p>
                    """, unsafe_allow_html=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("LTP (₹)", f"₹{price}")
                    m2.metric("Market Cap", f"₹{fund.get('Market Cap (Cr ₹)', 'N/A')} Cr" if fund.get('Market Cap (Cr ₹)') != 'N/A' else "N/A")
                    m3.metric("P/E Ratio", f"{fund.get('Trailing P/E', 'N/A')}")

                    m4, m5, m6 = st.columns(3)
                    m4.metric("52W High", f"₹{fund.get('52W High', 'N/A')}")
                    m5.metric("52W Low", f"₹{fund.get('52W Low', 'N/A')}")
                    m6.metric("ROE", f"{fund.get('ROE (%)', 'N/A')}%" if fund.get('ROE (%)') != 'N/A' else "N/A")

                    # Mini Sparkline chart
                    df_mini = get_stock_data(symbol, period="3mo")
                    if not df_mini.empty:
                        fig_mini = go.Figure()
                        color = "#00E676" if chg >= 0 else "#FF5252"
                        fig_mini.add_trace(go.Scatter(
                            x=df_mini.index, y=df_mini['Close'],
                            mode='lines', line=dict(color=color, width=2),
                            hovertemplate="%{x|%b %d}: ₹%{y:.2f}"
                        ))
                        fig_mini.update_layout(
                            height=120, margin=dict(l=0, r=0, t=10, b=0),
                            xaxis=dict(visible=False), yaxis=dict(visible=False),
                            template="plotly_dark", showlegend=False,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_mini, use_container_width=True, key=f"spark_{symbol}")

def render_dashboard_page():
    apply_custom_theme()

    st.markdown("<div class='glowing-header'>📊 Live Market Dashboard & Watchlist</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>24x7 Real-Time NSE/BSE Stock Monitoring Engine — CDSL, NSDL & Custom Watchlist</div>", unsafe_allow_html=True)

    watchlist = st.session_state.get("watchlist", ["CDSL.NS", "NSDL.BO"])

    # Render 5s Auto-Refreshing Fragment
    render_live_stock_cards_fragment(watchlist)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Demat Additions Weekly Run-Rate & Impact Widget (CDSL vs NSDL)
    render_demat_analytics_widget()

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Fundamentals Comparison Table
    st.markdown("### 📋 Fundamental Comparison Matrix")
    fundamentals_list = []
    for s in watchlist:
        f = get_stock_fundamentals(s)
        fundamentals_list.append({
            "Symbol": s,
            "Name": f.get("Name"),
            "Price (₹)": f.get("Current Price"),
            "1D Change %": f.get("Day Change %"),
            "Market Cap (Cr ₹)": f.get("Market Cap (Cr ₹)"),
            "P/E Ratio": f.get("Trailing P/E"),
            "P/B Ratio": f.get("Price to Book"),
            "ROE (%)": f.get("ROE (%)"),
            "Div Yield (%)": f.get("Dividend Yield (%)"),
            "52W High": f.get("52W High"),
            "52W Low": f.get("52W Low")
        })

    if fundamentals_list:
        df_fund = pd.DataFrame(fundamentals_list)
        st.dataframe(df_fund, use_container_width=True, hide_index=True)

if __name__ == "__main__" or True:
    render_dashboard_page()
