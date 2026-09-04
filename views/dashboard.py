import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.data_loader import get_stock_fundamentals, get_stock_data
from utils.nifty_correlation import analyze_nifty_impact
from config import BENCHMARK_TICKER, STOCK_NAME_MAP

def render_dashboard_page():
    st.title("📊 Live Market Dashboard & Watchlist")
    st.caption("24x7 Stock Monitoring Engine — Tracking CDSL, NSDL & Watchlist")

    watchlist = st.session_state.get("watchlist", ["CDSL.NS", "NSDL.NS"])

    # 1. Benchmark Index Banner (Nifty 50)
    st.subheader("🏛️ Benchmark Market Index (Nifty 50)")
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
        # Nifty impact quick check
        if watchlist:
            nifty_impact = analyze_nifty_impact(watchlist[0])
            if nifty_impact.get("status") == "success":
                st.metric(
                    "Nifty Market Trend", 
                    f"{nifty_impact['nifty_badge']} {nifty_impact['nifty_trend']}",
                    f"Beta: {nifty_impact['beta']}"
                )

    st.markdown("---")

    # 2. Watchlist Live Stock Cards
    st.subheader("📌 Monitored Watchlist Stocks")

    if not watchlist:
        st.info("Your watchlist is empty. Add stocks from the sidebar!")
        return

    # Render metric cards in rows of 2 or 3
    for i in range(0, len(watchlist), 2):
        cols = st.columns(2)
        batch = watchlist[i:i+2]
        
        for idx, symbol in enumerate(batch):
            with cols[idx]:
                with st.container(border=True):
                    fund = get_stock_fundamentals(symbol)
                    name = fund.get("Name", symbol)
                    price = fund.get("Current Price", "N/A")
                    chg = fund.get("Day Change", 0.0)
                    chg_pct = fund.get("Day Change %", 0.0)

                    st.markdown(f"### **{name}** (`{symbol}`)")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("LTP (₹)", f"₹{price}", f"{chg} ({chg_pct}%)")
                    m2.metric("Market Cap", f"₹{fund.get('Market Cap (Cr ₹)', 'N/A')} Cr")
                    m3.metric("P/E Ratio", f"{fund.get('Trailing P/E', 'N/A')}")

                    m4, m5, m6 = st.columns(3)
                    m4.metric("52W High", f"₹{fund.get('52W High', 'N/A')}")
                    m5.metric("52W Low", f"₹{fund.get('52W Low', 'N/A')}")
                    m6.metric("ROE", f"{fund.get('ROE (%)', 'N/A')}%" if fund.get('ROE (%)') != 'N/A' else "N/A")

                    # Mini Sparkline chart
                    df_mini = get_stock_data(symbol, period="3mo")
                    if not df_mini.empty:
                        fig_mini = go.Figure()
                        color = "#26a69a" if isinstance(chg, (int, float)) and chg >= 0 else "#ef5350"
                        fig_mini.add_trace(go.Scatter(
                            x=df_mini.index, y=df_mini['Close'],
                            mode='lines', line=dict(color=color, width=2),
                            hovertemplate="%{x|%b %d}: ₹%{y:.2f}"
                        ))
                        fig_mini.update_layout(
                            height=120, margin=dict(l=0, r=0, t=10, b=0),
                            xaxis=dict(visible=False), yaxis=dict(visible=False),
                            template="plotly_dark", showlegend=False
                        )
                        st.plotly_chart(fig_mini, use_container_width=True, key=f"spark_{symbol}")

    st.markdown("---")

    # 3. Fundamentals Comparison Table
    st.subheader("📋 Fundamental Comparison")
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

