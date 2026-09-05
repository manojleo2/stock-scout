import streamlit as st
from config import STOCK_NAME_MAP, COMMON_ALIASES
from utils.data_loader import load_saved_watchlist, save_persistent_watchlist, validate_ticker
from utils.ui_theme import apply_custom_theme

# 1. Page Configuration
st.set_page_config(
    page_title="Stock Scout — NSE Monitoring & Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Global Dark Glassmorphic Theme
apply_custom_theme()

# 2. Persistent Watchlist Initialization
if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = load_saved_watchlist()

# 3. Sidebar Watchlist Management Controls
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 10px 0;'>
            <h1 style='font-size: 2rem; margin:0; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                ⚡ Stock Scout
            </h1>
            <p style='color: #94a3b8; font-size: 0.8rem; margin-top: 4px;'>24x7 NSE Stock Monitoring & AI Signals</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("📌 Watchlist Manager")
    
    st.caption("💡 Search ticker (e.g. SBIN.NS, BSE.NS, RELIANCE.NS) or shortcuts (SBI, TCS, INFY):")
    new_stock = st.text_input("Add Stock Symbol", placeholder="e.g. SBIN.NS or SBI", label_visibility="collapsed").strip().upper()
    
    if st.button("➕ Add to Watchlist", use_container_width=True):
        if new_stock:
            resolved_stock = COMMON_ALIASES.get(new_stock, new_stock)
            
            if not resolved_stock.endswith(".NS") and not resolved_stock.endswith(".BO") and not resolved_stock.startswith("^"):
                resolved_stock = f"{resolved_stock}.NS"

            if resolved_stock in st.session_state["watchlist"]:
                st.info(f"ℹ️ `{resolved_stock}` is already in your watchlist.")
            else:
                with st.spinner(f"Verifying `{resolved_stock}`..."):
                    is_valid = validate_ticker(resolved_stock)
                
                if is_valid:
                    st.session_state["watchlist"].append(resolved_stock)
                    save_persistent_watchlist(st.session_state["watchlist"])
                    st.success(f"✅ Added `{resolved_stock}`!")
                    st.rerun()
                else:
                    if "SBI" in new_stock:
                        st.error(f"❌ Symbol `{resolved_stock}` not found. For State Bank of India, use `SBIN.NS`!")
                    else:
                        st.error(f"❌ Symbol `{resolved_stock}` not found on NSE/BSE.")

    st.markdown("#### Monitored Portfolio:")
    to_remove = None
    for symbol in st.session_state["watchlist"]:
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{symbol}**")
        if c2.button("❌", key=f"del_{symbol}"):
            to_remove = symbol

    if to_remove:
        st.session_state["watchlist"].remove(to_remove)
        save_persistent_watchlist(st.session_state["watchlist"])
        st.rerun()

    st.markdown("---")
    st.caption("⚡ Live Data Engine: yfinance + Scikit-Learn Ensemble")
    st.caption("🔒 Watchlist automatically saved to disk")

# 4. Multi-Page Navigation Setup
pages = {
    "Market Monitoring": [
        st.Page("views/dashboard.py", title="Live Dashboard", icon="📊", default=True),
        st.Page("views/portfolio.py", title="Portfolio & P&L Tracker", icon="💼"),
    ],
    "Analytics & AI Signals": [
        st.Page("views/technical_analysis.py", title="Technical Analysis", icon="📈"),
        st.Page("views/ml_prediction.py", title="AI Up/Down Forecast", icon="🤖"),
        st.Page("views/news_sentiment.py", title="News & Sentiment", icon="📰"),
    ]
}

pg = st.navigation(pages)
pg.run()
