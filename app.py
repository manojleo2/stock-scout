import streamlit as st
from config import STOCK_NAME_MAP, COMMON_ALIASES
from utils.data_loader import load_saved_watchlist, save_persistent_watchlist, validate_ticker

# 1. Page Configuration
st.set_page_config(
    page_title="Stock Scout — NSE Monitoring & Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Persistent Watchlist Initialization
if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = load_saved_watchlist()

# 3. Sidebar Watchlist Management Controls
with st.sidebar:
    st.image("https://img.icons8.com/color/96/line-chart.png", width=64)
    st.title("Stock Scout 🚀")
    st.caption("24x7 NSE Stock Monitoring & Prediction")
    st.markdown("---")

    st.subheader("📌 Watchlist Manager")
    
    # Quick Add Helpers
    st.caption("💡 Type ticker (e.g. `SBIN.NS`, `BSE.NS`, `RELIANCE.NS`) or common names like `SBI`, `TCS`, `INFY`:")
    new_stock = st.text_input("Add Stock Symbol", placeholder="e.g. SBIN.NS or SBI").strip().upper()
    
    if st.button("➕ Add to Watchlist", use_container_width=True):
        if new_stock:
            # Check for common shortcut aliases (e.g. SBI -> SBIN.NS)
            resolved_stock = COMMON_ALIASES.get(new_stock, new_stock)
            
            # Append .NS default if no exchange suffix
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
                    st.success(f"✅ Added `{resolved_stock}` to Watchlist!")
                    st.rerun()
                else:
                    if "SBI" in new_stock:
                        st.error(f"❌ Symbol `{resolved_stock}` not found on exchange. For State Bank of India, use `SBIN.NS`!")
                    else:
                        st.error(f"❌ Symbol `{resolved_stock}` not found on NSE/BSE. Please check the official ticker symbol.")

    st.markdown("#### Active Watchlist:")
    to_remove = None
    for symbol in st.session_state["watchlist"]:
        c1, c2 = st.columns([4, 1])
        label = STOCK_NAME_MAP.get(symbol, symbol)
        c1.markdown(f"**{symbol}**")
        if c2.button("❌", key=f"del_{symbol}"):
            to_remove = symbol

    if to_remove:
        st.session_state["watchlist"].remove(to_remove)
        save_persistent_watchlist(st.session_state["watchlist"])
        st.rerun()

    st.markdown("---")
    st.caption("Data source: Yahoo Finance (yfinance) & Google News RSS")
    st.caption("Watchlist automatically saved across sessions")

# 4. Multi-Page Navigation Setup (Streamlit 1.36+)
pages = {
    "Market Monitoring": [
        st.Page("views/dashboard.py", title="Live Dashboard", icon="📊", default=True),
    ],
    "Analytics & AI Signals": [
        st.Page("views/technical_analysis.py", title="Technical Analysis", icon="📈"),
        st.Page("views/ml_prediction.py", title="ML Up/Down Prediction", icon="🤖"),
        st.Page("views/news_sentiment.py", title="News & Sentiment", icon="📰"),
    ]
}

pg = st.navigation(pages)
pg.run()
