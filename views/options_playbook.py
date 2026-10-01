import streamlit as st
import pandas as pd
import datetime as dt

from utils.data_loader import get_stock_fundamentals
from utils.ui_theme import apply_custom_theme
from utils.market_calendar import NSE_HOLIDAYS_2026, is_trading_holiday
from config import STOCK_NAME_MAP

def get_2026_monthly_expiries() -> list:
    """Compute the exact NSE monthly options expiry date for every month in 2026."""
    expiries = []
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m in range(1, 13):
        # find last day of month
        if m == 12:
            last_d = dt.date(2026, 12, 31)
        else:
            last_d = dt.date(2026, m + 1, 1) - dt.timedelta(days=1)
        
        # backtrack to last Thursday
        d = last_d
        while d.weekday() != 3:
            d -= dt.timedelta(days=1)
            
        # if the last Thursday is a market holiday, expiry prepends to Wednesday
        is_holiday, holiday_name = is_trading_holiday(d)
        note = "Standard Last Thursday"
        if is_holiday:
            d -= dt.timedelta(days=1)
            note = f"Prepended to Wednesday ({holiday_name} Holiday on Thu)"
            
        expiries.append({
            "Month": f"{month_names[m-1]} 2026",
            "Expiry Date": d.strftime("%d %b %Y (%a)"),
            "Date_Obj": d,
            "Type": "Monthly F&O Expiry",
            "Settlement Note": note
        })
    return expiries

def render_options_playbook_page():
    apply_custom_theme()

    st.markdown("<div class='glowing-header'>🎯 CDSL Intraday Options Trading Playbook</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>Complete Execution Guide for CDSL Intraday Options (V2 FSM), Real-Time Premium P&L Simulator & NSE Expiry Rules</div>", unsafe_allow_html=True)

    # 4 Main Tabs
    tab_sim, tab_groww, tab_expiry, tab_risk = st.tabs([
        "🧮 Live Intraday Premium & P&L Simulator",
        "📱 Step-by-Step Groww App Execution Guide",
        "⏳ NSE Monthly Expiry & Settlement Rules",
        "🛡️ Capital Sizing, Pre-Trade Gates & Win Rules"
    ])

    watchlist = st.session_state.get("watchlist", ["CDSL.NS"])

    # ==========================================
    # TAB 1: LIVE INTRADAY P&L SIMULATOR
    # ==========================================
    with tab_sim:
        st.subheader("🧮 Interactive Intraday Options Premium & P&L Calculator")
        st.caption("All targets, stop-losses, and returns are calculated strictly on OPTIONS PREMIUM PRICE (₹).")

        c_s1, c_s2, c_s3 = st.columns(3)
        with c_s1:
            sim_stock = st.selectbox(
                "Select Stock",
                options=watchlist,
                format_func=lambda s: f"{STOCK_NAME_MAP.get(s, s)} ({s})"
            )
            fund = get_stock_fundamentals(sim_stock)
            live_price = fund.get("Current Price")
            default_price = float(live_price) if isinstance(live_price, (int, float)) else 1360.0

        with c_s2:
            sim_trade = st.radio(
                "Trade Direction",
                options=["🔴 BUY PUT (PE) — Model DOWN", "🟢 BUY CALL (CE) — Model UP"],
                index=0
            )
            is_put = "PUT" in sim_trade

        with c_s3:
            sizing_mode = st.radio(
                "Position Sizing",
                options=["2 Lots (950 Qty / Standard)", "1 Lot (475 Qty / Conservative)"],
                index=0
            )
            lot_qty = 950 if "2 Lots" in sizing_mode else 475

        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            stock_spot = st.number_input("CDSL Entry Spot Price (₹)", min_value=100.0, max_value=10000.0, value=default_price, step=5.0)
        with c_p2:
            strike_price = st.number_input("ATM Strike Price (₹20 Interval)", min_value=100.0, max_value=10000.0, value=round(stock_spot / 20) * 20.0, step=20.0)
        with c_p3:
            premium_paid = st.number_input("Option Entry Premium Paid (₹)", min_value=1.0, max_value=500.0, value=30.0, step=0.5)

        total_capital_invested = lot_qty * premium_paid

        st.markdown("---")
        st.markdown("#### ⚡ Simulate Intraday Spot Move & Option Premium Change")

        col_slider1, col_slider2 = st.columns(2)
        with col_slider1:
            spot_move = st.slider(
                "CDSL Spot Price Move from Entry (₹)",
                min_value=-40.0,
                max_value=40.0,
                value=-14.0 if is_put else 14.0,
                step=1.0,
                help="Positive (+) means stock rises. Negative (-) means stock falls."
            )
        with col_slider2:
            # ATM Delta ~ 0.50
            delta_val = -0.50 if is_put else 0.50
            implied_prem_change = round(spot_move * delta_val, 2)
            premium_slider = st.slider(
                "Direct Option Premium Change (₹/share)",
                min_value=-15.0,
                max_value=25.0,
                value=float(implied_prem_change),
                step=0.25,
                help="Simulate the actual rupee change in option premium."
            )

        new_premium = max(round(premium_paid + premium_slider, 2), 0.05)
        new_position_value = round(new_premium * lot_qty, 2)
        net_pl_rs = round(new_position_value - total_capital_invested - 100.0, 2)  # flat ₹100 brokerage/taxes
        net_pl_pct = round((net_pl_rs / total_capital_invested) * 100.0, 1)

        # Rulebook Milestones Comparison
        tgt_premium = premium_paid + 10.00
        prec_sl_premium = max(premium_paid - 5.00, 0.05)
        emerg_sl_premium = max(premium_paid - 8.00, 0.05)

        is_profit = net_pl_rs >= 0
        card_color = "#00E676" if is_profit else "#FF5252"
        badge_text = "🎉 PROFITABLE INTRADAY MOVE" if is_profit else "⚠️ POSITION ADVERSE MOVE"

        with st.container(border=True):
            st.markdown(
                f"<div style='border-left: 5px solid {card_color}; padding: 10px 14px; background: rgba(30,41,59,0.3); border-radius: 6px; margin-bottom: 12px;'>"
                f"<h3 style='margin:0; color:{card_color};'>{badge_text}: {net_pl_rs:+,.2f} ₹ ({net_pl_pct:+.1f}%)</h3>"
                f"<p style='margin:4px 0 0 0; color:#94a3b8; font-size:0.9rem;'>Option: <strong>CDSL ₹{int(strike_price)} {'PE' if is_put else 'CE'}</strong> | Entry: ₹{premium_paid:.2f} ➔ Now: <strong>₹{new_premium:.2f}</strong> ({premium_slider:+.2f} ₹/sh)</p>"
                f"</div>",
                unsafe_allow_html=True
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Capital Deployed", f"₹{total_capital_invested:,.2f}", f"₹{premium_paid:.2f} × {lot_qty} Qty")
            m2.metric("Target Premium (+₹10)", f"₹{tgt_premium:.2f}", f"+₹{10.0 * lot_qty:,.0f} Gross Gain")
            m3.metric("Precision SL (-₹5)", f"₹{prec_sl_premium:.2f}", f"-₹{5.0 * lot_qty:,.0f} Max Loss")
            m4.metric("Simulated Net P&L", f"{net_pl_rs:+,.2f} ₹", f"{net_pl_pct:+.1f}% Net", delta_color="normal")

            st.markdown("##### 🎯 Winning Rulebook Trailing Thresholds:")
            b1, b2, b3 = st.columns(3)
            b1.info(f"**At +₹7.00 Gain (₹{premium_paid + 7:.2f}):** Trail SL immediately to Cost/Entry (₹{premium_paid:.2f}) $\\rightarrow$ Zero Risk.")
            b2.info(f"**At +₹8.50 Gain (₹{premium_paid + 8.5:.2f}):** Trail SL immediately to +₹5.00 Gain (₹{premium_paid + 5:.2f}) $\\rightarrow$ Lock +₹{5.0 * lot_qty:,.0f}.")
            b3.info(f"**Target (+₹10.00 @ ₹{tgt_premium:.2f}):** Book Lot 1 (+₹4,750). Trail Lot 2 runner with 5-min 20 EMA.")

    # ==========================================
    # TAB 2: STEP-BY-STEP GROWW APP GUIDE
    # ==========================================
    with tab_groww:
        st.subheader("📱 How to Execute CDSL Intraday Options on Groww (Step-by-Step)")
        st.caption("Follow these exact steps between 09:18 AM and 09:20 AM IST. Exit strictly before 03:05 PM.")

        step1, step2 = st.columns(2)
        with step1:
            with st.container(border=True):
                st.markdown("#### 1️⃣ Open Groww & Search CDSL (09:18 AM)")
                st.markdown("""
                1. Open the **Groww** app on your phone at **09:18 AM**.
                2. In the search bar, type **`CDSL`** and tap on **Central Depository Services Ltd**.
                3. On the CDSL stock overview page, tap the **"Option Chain"** button (next to the chart icon).
                """)

        with step2:
            with st.container(border=True):
                st.markdown("#### 2️⃣ Choose Side Based on Model Direction")
                st.markdown("""
                - **LEFT Side = CALLS (CE):** Tap here if Model predicts **UP** (spot above VWAP).
                - **RIGHT Side = PUTS (PE):** Tap here if Model predicts **DOWN** (spot below VWAP).
                - **MIDDLE Column = Strike Prices** (CDSL trades in ₹20 intervals, e.g. ₹1340, ₹1360, ₹1380).
                """)

        step3, step4 = st.columns(2)
        with step3:
            with st.container(border=True):
                st.markdown("#### 3️⃣ Select the ATM Strike (At 09:20 AM)")
                st.markdown("""
                1. Identify the **At-The-Money (ATM)** strike closest to CDSL current spot price (e.g. ₹1360).
                2. Tap on the premium price (e.g. **₹30.00**) on the Put or Call side.
                3. Tap the green **"BUY"** button at the bottom.
                """)

        with step4:
            with st.container(border=True):
                st.markdown("#### 4️⃣ ⚠️ Order Settings & Position Sizing")
                st.markdown("""
                - **Order Type:** Select **Intraday (MIS)** or **Delivery (Normal)**.
                - **Quantity:** 
                  - **2 Lots (950 shares)** if account capital > ₹30,000.
                  - **1 Lot (475 shares)** if account capital ≤ ₹30,000.
                - **Order Price:** Market (or Limit at current Ask). Tap **BUY**.
                """)

        with st.container(border=True):
            st.markdown("#### 5️⃣ Profit Management & 03:05 PM Hard Cutoff")
            st.markdown("""
            1. **At 09:25 AM:** Arm strict Precision Stop-Loss at **Entry Premium - ₹5.00** (-₹4,850 on 2 Lots).
            2. **At +₹7.00 Gain:** Modify SL order to **Cost / Entry Price (Zero Risk)**.
            3. **At +₹8.50 Gain:** Modify SL order to **Entry + ₹5.00 (+₹4,750 Locked Profit)**.
            4. **At +₹10.00 Gain:** Sell **Lot 1 (475 Qty)** to lock profit. Hold Lot 2 as runner trailing 5-min 20 EMA.
            5. **03:05 PM Cutoff:** If still holding, exit all positions immediately at market. **Never hold overnight!**
            """)

    # ==========================================
    # TAB 3: NSE MONTHLY EXPIRY & SETTLEMENT RULES
    # ==========================================
    with tab_expiry:
        st.subheader("⏳ NSE Monthly Options Expiry & SEBI Physical Settlement Guide")
        st.caption("Stock options physical delivery rules and why all intraday positions are squared off by 03:05 PM.")

        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            with st.container(border=True):
                st.markdown("### ⚠️ The Physical Settlement Mandate")
                st.error("""
                **NSE stock options are NOT cash-settled at expiry!**
                Under SEBI guidelines, if you hold an In-The-Money (ITM) stock option past 3:30 PM on expiry day:
                - **If you hold a Call (CE):** You are legally obligated to **buy 475 physical shares** of CDSL (requiring ~₹6.5 Lakhs in cash!).
                - **If you hold a Put (PE):** You are legally obligated to **deliver 475 physical shares** of CDSL from your Demat account!
                """)
                st.success("""
                🛡️ **How We Stay 100% Safe:**
                - **We trade STRICTLY INTRADAY.**
                - All positions are closed at **03:05 PM every single day**.
                - Zero physical delivery risk, zero overnight gap risk.
                """)

        with exp_c2:
            with st.container(border=True):
                st.markdown("### 📉 The 3 Effects of Expiry Week")
                st.markdown("""
                1. **Accelerated Theta (Time Decay):** In expiry week, Out-of-the-Money options lose 50-80% of value very quickly.
                2. **The Zero Rule:** At 3:30 PM on expiry Thursday, **any option that is Out-of-the-Money becomes exactly ₹0.00**.
                3. **When to Roll Over:** On the last Wednesday or Thursday of the month, do NOT buy current month options. Select the **Next Month expiry** in Groww.
                """)

        st.markdown("### 📅 Official 2026 Monthly F&O Expiry Calendar (NSE/BSE)")
        expiries_2026 = get_2026_monthly_expiries()
        today = dt.date.today()

        df_exp = pd.DataFrame([
            {
                "Month": e["Month"],
                "Expiry Date": e["Expiry Date"],
                "Contract Type": e["Type"],
                "Status": "✅ Completed" if e["Date_Obj"] < today else ("🟢 ACTIVE EXPIRY" if e["Date_Obj"].month == today.month and e["Date_Obj"].year == today.year else "⏳ Upcoming"),
                "Exchange Settlement Schedule": e["SettlementNote" if "SettlementNote" in e else "Settlement Note"]
            } for e in expiries_2026
        ])
        st.dataframe(df_exp, use_container_width=True, hide_index=True)

    # ==========================================
    # TAB 4: CAPITAL SIZING & WIN RULES
    # ==========================================
    with tab_risk:
        st.subheader("🛡️ Professional Capital Allocation & Intraday Win Discipline")
        st.caption("Rules governing lot sizing, pre-trade gates, and daily stop limits.")

        r1, r2 = st.columns(2)
        with r1:
            with st.container(border=True):
                st.markdown("### 💰 Capital Sizing Thresholds")
                st.markdown("""
                - **Capital > ₹30,000:** Deploy **2 Lots (950 Qty)** (~₹24,000–₹28,000 deployment).
                - **Capital ≤ ₹30,000:** Deploy **1 Lot ONLY (475 Qty)** (~₹12,000–₹14,000 deployment).
                - **Capital < ₹20,000:** **STOP TRADING.** Alert immediately.
                - **Daily 2R Loss Limit:** If 2 trades hit SL in one day (**-₹9,500 total loss**) $\\rightarrow$ **PERMANENT LOCKOUT FOR THE DAY**. No revenge trading.
                """)

        with r2:
            with st.container(border=True):
                st.markdown("### 🎯 09:20 AM Pre-Trade Gatekeeper Checklist")
                st.markdown(r"""
                Before placing any order at 09:20 AM, verify all 4 gates pass:
                1. **Conviction Gate:** Calibrated probability $\ge 65\%$ (Mon–Thu) or $\ge 70\%$ (Friday).
                2. **VWAP Gate:** Spot must be below VWAP for PE, or above VWAP for CE.
                3. **Spread Check:** Bid-Ask spread $\le 1.0\%$ of midpoint or $\le ₹0.40$.
                4. **Anti-Chase Cap:** Option premium must NOT have gained $> +₹6.50$ from 09:15 open.
                """)

if __name__ == "__main__" or True:
    render_options_playbook_page()
