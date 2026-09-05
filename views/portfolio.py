import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.portfolio_manager import (
    load_saved_portfolio, save_persistent_portfolio, calculate_portfolio_performance
)
from utils.notifications import get_telegram_credentials, send_test_notification
from utils.ui_theme import apply_custom_theme
from config import STOCK_NAME_MAP, COMMON_ALIASES
from utils.data_loader import validate_ticker

@st.fragment(run_every="5s")
def render_live_portfolio_summary_fragment(portfolio: list):
    """
    Auto-refreshes live portfolio performance and P&L metrics every 5 seconds.
    """
    perf = calculate_portfolio_performance(portfolio)

    if perf["status"] == "empty":
        st.info("Your portfolio is currently empty. Add holdings below!")
        return

    # 1. Top KPI Portfolio Summary Banner
    pnl_rs = perf['total_pnl_rs']
    pnl_pct = perf['total_pnl_pct']
    pnl_prefix = "+" if pnl_rs >= 0 else ""

    today_pnl_rs = perf['today_pnl_rs']
    today_pnl_prefix = "+" if today_pnl_rs >= 0 else ""

    st.markdown("### 💼 Overall Portfolio Performance & P&L")
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Investment", f"₹{perf['total_invested']:,.2f}")
    with k2:
        st.metric("Current Portfolio Value", f"₹{perf['total_current_value']:,.2f}")
    with k3:
        st.metric(
            "Overall P&L (Unrealized)", 
            f"{pnl_prefix}₹{abs(pnl_rs):,.2f}", 
            f"{pnl_prefix}{pnl_pct:.2f}% Total Gain",
            delta_color="normal" if pnl_rs >= 0 else "inverse"
        )
    with k4:
        st.metric(
            "Today's P&L Change", 
            f"{today_pnl_prefix}₹{abs(today_pnl_rs):,.2f}", 
            "Live Session Change",
            delta_color="normal" if today_pnl_rs >= 0 else "inverse"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Holdings Individual Stock Cards
    st.markdown("### 📌 Individual Holdings & Target Tracking (Live 5s)")

    holdings = perf['holdings']
    for h in holdings:
        item_pnl_rs = h['pnl_rs']
        item_pnl_pct = h['pnl_pct']
        badge_class = "badge-up" if item_pnl_rs >= 0 else "badge-down"
        badge_text = f"P&L: +₹{item_pnl_rs:,.2f} (+{item_pnl_pct:.2f}%)" if item_pnl_rs >= 0 else f"P&L: -₹{abs(item_pnl_rs):,.2f} ({item_pnl_pct:.2f}%)"

        with st.container(border=True):
            st.markdown(f"""
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <h3 style='margin:0; font-weight:700;'>{h['name']} <span style='font-size:0.85rem; color:#94a3b8;'>({h['symbol']})</span></h3>
                    <span class='{badge_class}'>{badge_text}</span>
                </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Buy Price", f"₹{h['buy_price']:,.2f}")
            m2.metric("LTP (Current)", f"₹{h['current_price']:,.2f}", f"{h['day_change_pct']}% today")
            m3.metric("Quantity", f"{h['quantity']} shares")
            m4.metric("Current Value", f"₹{h['current_value']:,.2f}")

            # Target Sell Price Progress Bar
            target_p = h['target_price']
            prog_pct = h['target_progress_pct']
            st.progress(int(prog_pct) / 100.0)
            st.caption(f"🎯 **Target Sell Price:** ₹{target_p:,.2f} | **Target Progress:** {prog_pct}% achieved")

def render_portfolio_page():
    apply_custom_theme()

    st.markdown("<div class='glowing-header'>💼 Personal Portfolio & Live P&L Tracker</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>Track Your Investments, Live Unrealized Gains, Day's P&L, and Target Sell Prices</div>", unsafe_allow_html=True)

    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = load_saved_portfolio()

    portfolio = st.session_state["portfolio"]

    # Render Live 5s Summary Fragment
    render_live_portfolio_summary_fragment(portfolio)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Add / Edit Holding Form
    st.markdown("### ⚙️ Manage Portfolio Holdings")
    
    with st.expander("➕ Add / Edit Stock Holding", expanded=False):
        c_in1, c_in2, c_in3, c_in4 = st.columns(4)
        with c_in1:
            sym_input = st.text_input("Stock Symbol / Name", placeholder="e.g. CDSL.NS or SBI").strip().upper()
        with c_in2:
            buy_p_input = st.number_input("Average Buy Price (₹)", min_value=1.0, value=1200.0, step=10.0)
        with c_in3:
            qty_input = st.number_input("Share Quantity", min_value=1, value=50, step=5)
        with c_in4:
            target_p_input = st.number_input("Target Sell Price (₹)", min_value=1.0, value=1500.0, step=10.0)

        notes_input = st.text_input("Notes / Strategy", placeholder="e.g. Core long-term holding")

        if st.button("💾 Save Holding to Portfolio", use_container_width=True):
            if sym_input:
                resolved_sym = COMMON_ALIASES.get(sym_input, sym_input)
                if not resolved_sym.endswith(".NS") and not resolved_sym.endswith(".BO") and not resolved_sym.startswith("^"):
                    resolved_sym = f"{resolved_sym}.NS"

                if validate_ticker(resolved_sym):
                    existing = next((item for item in st.session_state["portfolio"] if item["symbol"] == resolved_sym), None)
                    if existing:
                        existing["buy_price"] = buy_p_input
                        existing["quantity"] = qty_input
                        existing["target_price"] = target_p_input
                        existing["notes"] = notes_input
                        st.success(f"✅ Updated holding for `{resolved_sym}`!")
                    else:
                        st.session_state["portfolio"].append({
                            "symbol": resolved_sym,
                            "buy_price": buy_p_input,
                            "quantity": qty_input,
                            "target_price": target_p_input,
                            "notes": notes_input
                        })
                        st.success(f"✅ Added `{resolved_sym}` to Portfolio!")
                    
                    save_persistent_portfolio(st.session_state["portfolio"])
                    st.rerun()
                else:
                    st.error(f"❌ Symbol `{resolved_sym}` not found on NSE/BSE.")

    # Remove Holding Selector
    if portfolio:
        with st.expander("❌ Remove Holding", expanded=False):
            sym_to_del = st.selectbox("Select Holding to Remove", options=[item["symbol"] for item in portfolio])
            if st.button("🗑️ Remove Holding", type="secondary"):
                st.session_state["portfolio"] = [item for item in st.session_state["portfolio"] if item["symbol"] != sym_to_del]
                save_persistent_portfolio(st.session_state["portfolio"])
                st.success(f"Removed `{sym_to_del}` from Portfolio.")
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Detailed Holdings Summary Matrix Table
    perf = calculate_portfolio_performance(portfolio)
    if perf["status"] == "success" and perf["holdings"]:
        st.markdown("### 📋 Holdings Master Table")
        df_table = pd.DataFrame([
            {
                "Symbol": h["symbol"],
                "Name": h["name"],
                "Buy Price (₹)": f"₹{h['buy_price']:,.2f}",
                "Current Price (₹)": f"₹{h['current_price']:,.2f}",
                "Quantity": h["quantity"],
                "Invested (₹)": f"₹{h['invested']:,.2f}",
                "Current Value (₹)": f"₹{h['current_value']:,.2f}",
                "Total P&L (₹)": f"{'+' if h['pnl_rs']>=0 else ''}₹{h['pnl_rs']:,.2f}",
                "P&L Gain (%)": f"{'+' if h['pnl_pct']>=0 else ''}{h['pnl_pct']:.2f}%",
                "Target Price (₹)": f"₹{h['target_price']:,.2f}",
                "Notes": h["notes"]
            } for h in perf["holdings"]
        ])
        st.dataframe(df_table, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. Telegram Mobile Alert Settings
    st.markdown("### 📱 Mobile Alert Notifications Status")
    bot_token, chat_id = get_telegram_credentials()

    with st.expander("📲 Configure Telegram Mobile Alerts", expanded=False):
        if bot_token and chat_id:
            st.success("✅ **Telegram Bot is Connected & Active!** Your phone will receive live market alerts.")
            if st.button("🧪 Send Test Alert to My Phone", use_container_width=True):
                with st.spinner("Sending test alert..."):
                    success, err_msg = send_test_notification()
                if success:
                    st.success("🎉 Test notification sent! Check your Telegram app on your phone.")
                else:
                    st.error(f"❌ Failed to send alert: {err_msg}")
                    st.info("💡 **Common Fix:** Open your Telegram app, search for your bot name, and click **START** (or send `/start`). Telegram blocks bots from messaging users until you start the chat!")
        else:
            st.warning("⚠️ **Telegram Alerts are not activated yet.** Follow the simple 2-step setup below:")
            st.markdown("""
                1. Open Telegram, search for `@BotFather`, type `/newbot`, and copy your **Bot Token**.
                2. Search for `@userinfobot` on Telegram to get your numeric **Chat ID**.
                3. Go to your **Streamlit Cloud Dashboard** -> click **`< Manage app`** (bottom right) -> **App Settings** -> **Secrets**.
                4. Paste the following 2 lines into Secrets (replace with your real values):
                   ```toml
                   TELEGRAM_BOT_TOKEN = "your_bot_token_here"
                   TELEGRAM_CHAT_ID = "your_chat_id_here"
                   ```
            """)

if __name__ == "__main__" or True:
    render_portfolio_page()
