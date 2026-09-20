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

    st.markdown("<div class='glowing-header'>🎯 Options Trading Playbook & Expiry Guide</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>Complete Execution Guide for 3:05 PM Put/Call Trades on Groww, Real-Time P&L Simulator & NSE Expiry Rules</div>", unsafe_allow_html=True)

    # 4 Main Tabs
    tab_sim, tab_groww, tab_expiry, tab_risk = st.tabs([
        "🧮 Live Gap & P&L Simulator",
        "📱 Step-by-Step Groww App Guide",
        "⏳ NSE Monthly Expiry & Settlement Rules",
        "🛡️ Capital Sizing & Win Rules"
    ])

    watchlist = st.session_state.get("watchlist", ["CDSL.NS", "HDFCBANK.NS"])

    # ==========================================
    # TAB 1: LIVE GAP & P&L SIMULATOR
    # ==========================================
    with tab_sim:
        st.subheader("🧮 Interactive Overnight Gap & P&L Calculator")
        st.caption("Simulate exact rupee returns, Delta movement, and overnight Theta time decay for any overnight Put/Call trade.")

        c_s1, c_s2, c_s3 = st.columns(3)
        with c_s1:
            sim_stock = st.selectbox(
                "Select Stock",
                options=watchlist,
                format_func=lambda s: f"{STOCK_NAME_MAP.get(s, s)} ({s})"
            )
            # Fetch live current price
            fund = get_stock_fundamentals(sim_stock)
            live_price = fund.get("Current Price")
            default_price = float(live_price) if isinstance(live_price, (int, float)) else 1360.0

        with c_s2:
            sim_trade = st.radio(
                "Trade Direction",
                options=["🔴 BUY PUT (PE) — Bearish", "🟢 BUY CALL (CE) — Bullish"],
                index=0
            )
            is_put = "PUT" in sim_trade

        with c_s3:
            lot_size = st.number_input("NSE Lot Size (Shares per Lot)", min_value=50, max_value=2000, value=350, step=50, help="For CDSL, typical lot size is 350 or 400 shares.")

        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            stock_spot = st.number_input("Stock Entry Price (₹)", min_value=100.0, max_value=10000.0, value=default_price, step=5.0)
        with c_p2:
            strike_price = st.number_input("Option Strike Price (₹)", min_value=100.0, max_value=10000.0, value=round(stock_spot / 20) * 20.0, step=20.0)
        with c_p3:
            premium_paid = st.number_input("Premium Paid per Share (₹)", min_value=1.0, max_value=500.0, value=30.0, step=1.0)

        total_capital_invested = lot_size * premium_paid

        st.markdown("---")
        st.markdown("#### ⚡ Simulate Tomorrow Morning's 9:15 AM Opening Gap")

        gap_val = st.slider(
            "Tomorrow's 9:15 AM Opening Price Gap (₹)",
            min_value=-50.0,
            max_value=50.0,
            value=10.0 if is_put else 10.0,
            step=1.0,
            help="Positive (+) means stock gaps UP. Negative (-) means stock gaps DOWN."
        )

        # Mathematical calculations
        # ATM Delta ~ 0.50 (positive for CE, negative for PE)
        delta = -0.50 if is_put else +0.50
        # Overnight Theta Decay ~ 6% to 8% of premium for 1 night
        theta_decay = round(premium_paid * 0.07, 2)
        
        # Price change impact
        delta_impact = gap_val * delta
        new_premium = max(round(premium_paid + delta_impact - theta_decay, 2), 0.05)
        new_position_value = round(new_premium * lot_size, 2)
        net_pl_rs = round(new_position_value - total_capital_invested, 2)
        net_pl_pct = round((net_pl_rs / total_capital_invested) * 100.0, 1)

        # Display Outcome Card
        is_profit = net_pl_rs >= 0
        card_color = "#00E676" if is_profit else "#FF5252"
        badge_text = "🎉 PROFITABLE TRADE" if is_profit else "⚠️ POSITION IN LOSS"

        with st.container(border=True):
            st.markdown(
                f"<div style='border-left: 5px solid {card_color}; padding: 10px 14px; background: rgba(30,41,59,0.3); border-radius: 6px; margin-bottom: 12px;'>"
                f"<h3 style='margin:0; color:{card_color};'>{badge_text}: {net_pl_rs:+,.2f} ₹ ({net_pl_pct:+.1f}%)</h3>"
                f"<p style='margin:4px 0 0 0; color:#94a3b8; font-size:0.9rem;'>CDSL at ₹{stock_spot:,.2f} gapping to <strong>₹{stock_spot + gap_val:,.2f}</strong> ({gap_val:+,.2f} ₹ move)</p>"
                f"</div>",
                unsafe_allow_html=True
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Initial Capital Paid", f"₹{total_capital_invested:,.2f}", f"₹{premium_paid:.2f} × {lot_size}")
            m2.metric("New Premium at 9:15 AM", f"₹{new_premium:,.2f}", f"{new_premium - premium_paid:+.2f} ₹/share")
            m3.metric("New Position Value", f"₹{new_position_value:,.2f}")
            m4.metric("Net Return on Trade", f"{net_pl_pct:+.1f}%", f"{net_pl_rs:+,.2f} ₹", delta_color="normal")

            st.markdown("##### 🔬 Mathematical Breakdown:")
            b1, b2, b3 = st.columns(3)
            b1.info(f"**1. Delta Impact:** `{gap_val:+} ₹` move × `{delta}` Delta = **`{delta_impact:+.2f} ₹`**")
            b2.info(f"**2. Overnight Theta Decay:** One night time loss = **`-₹{theta_decay:.2f}`**")
            b3.info(f"**3. Exit Timing:** Premium is estimated at **9:15–9:20 AM**. Daytime chop after 9:30 AM will increase Theta decay.")

    # ==========================================
    # TAB 2: STEP-BY-STEP GROWW APP GUIDE
    # ==========================================
    with tab_groww:
        st.subheader("📱 How to Execute Put / Call Trades on Groww (Step-by-Step)")
        st.caption("Follow these exact steps on your phone between 3:10 PM and 3:20 PM IST.")

        step1, step2 = st.columns(2)
        with step1:
            with st.container(border=True):
                st.markdown("#### 1️⃣ Search & Open Option Chain")
                st.markdown("""
                1. Open the **Groww** app on your phone.
                2. In the top search bar, type **`CDSL`** and tap on **Central Depository Services Ltd**.
                3. On the CDSL stock overview page, tap the **"Option Chain"** button (next to the chart icon).
                """)

        with step2:
            with st.container(border=True):
                st.markdown("#### 2️⃣ Choose Calls (CE) or Puts (PE)")
                st.markdown("""
                - **LEFT Side = CALLS (CE):** Tap here if Stock Scout says `🟢 BUY CALL` (Expecting Gap UP).
                - **RIGHT Side = PUTS (PE):** Tap here if Stock Scout says `🔴 BUY PUT` (Expecting Gap DOWN).
                - **MIDDLE Column = Strike Prices** (e.g. ₹1340, ₹1360, ₹1380).
                """)

        step3, step4 = st.columns(2)
        with step3:
            with st.container(border=True):
                st.markdown("#### 3️⃣ Select the Suggested Strike (At 3:10 PM)")
                st.markdown("""
                1. Look at the At-The-Money (ATM) strike closest to the stock's current price (e.g. **₹1360**).
                2. Tap on the premium price (e.g. **₹30.00**) next to that strike on the Put or Call side.
                3. Tap the green **"BUY"** button at the bottom.
                """)

        with step4:
            with st.container(border=True):
                st.markdown("#### 4️⃣ ⚠️ Critical Order Settings (Delivery vs Intraday)")
                st.markdown("""
                - **Order Type:** Select **Delivery** (Normal / NRML).
                  > 🚨 **NEVER select Intraday (MIS)!** If you choose Intraday, Groww's automated RMS risk system will force-sell your option at 3:20 PM today, and you won't get tomorrow's gap!
                - **Quantity:** Enter **1 Lot** (e.g. 350 shares).
                - **Price:** Choose **Market** (instant buy) or Limit. Tap **BUY**.
                """)

        with st.container(border=True):
            st.markdown("#### 5️⃣ Next Morning: How to Exit at 9:16 AM")
            st.markdown("""
            1. Next morning at **9:15 AM**, open Groww and tap the **"Positions"** tab.
            2. Tap on your active CDSL option position.
            3. Tap **"EXIT"** $\rightarrow$ select **Market Order** $\rightarrow$ tap **Confirm Sell**.
            4. Your profit (or controlled loss) is immediately realized and cash returns to your balance!
            """)

    # ==========================================
    # TAB 3: NSE MONTHLY EXPIRY & SETTLEMENT RULES
    # ==========================================
    with tab_expiry:
        st.subheader("⏳ NSE Monthly Options Expiry & SEBI Physical Settlement Guide")
        st.caption("Why stock options behave differently from index options, and how to avoid severe penalties.")

        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            with st.container(border=True):
                st.markdown("### ⚠️ The Physical Settlement Mandate")
                st.error("""
                **NSE stock options are NOT cash-settled at expiry!**
                Under SEBI guidelines, if you hold an In-The-Money (ITM) stock option past 3:30 PM on expiry day:
                - **If you hold a Call (CE):** You are legally obligated to **buy 350 physical shares** of CDSL (requiring ~₹4.8 Lakhs in cash!).
                - **If you hold a Put (PE):** You are legally obligated to **deliver 350 physical shares** of CDSL from your Demat account!
                """)
                st.success("""
                🛡️ **How to Stay 100% Safe:**
                - **Always EXIT before 3:00 PM on Expiry Thursday.**
                - Never carry an option past 3:00 PM on the last Thursday of the month.
                - Groww auto-squares off retail positions around 3:00–3:15 PM, but it is best practice to exit manually.
                """)

        with exp_c2:
            with st.container(border=True):
                st.markdown("### 📉 The 3 Effects of Expiry Week")
                st.markdown("""
                1. **Accelerated Theta (Time Decay):** During the first 3 weeks, theta decay is mild (~₹1-₹2/day). In expiry week, Out-of-the-Money options lose 50-80% of value very quickly.
                2. **The Zero Rule:** At 3:30 PM on expiry Thursday, **any option that is Out-of-the-Money becomes exactly ₹0.00**.
                3. **When to Roll Over:** On the last Wednesday or Thursday of the month, do NOT buy current month options. Select the **Next Month tab** in Groww.
                """)

        st.markdown("### 📅 Official 2026 Monthly F&O Expiry Calendar (NSE/BSE)")
        expiries_2026 = get_2026_monthly_expiries()
        today = dt.date.today()

        # Format dataframe
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
        st.subheader("🛡️ Professional Capital Allocation & Win Discipline")
        st.caption("How to manage ₹50,000 starting capital without risking account wipeouts.")

        r1, r2 = st.columns(2)
        with r1:
            with st.container(border=True):
                st.markdown("### 💰 The 20-30% Capital Allocation Rule")
                st.markdown("""
                If your total trading capital is **₹50,000**:
                - **Trade Size:** Deploy at most 1 to 2 lots (~₹8,000 to ₹16,000 per trade).
                - **Safety Buffer:** Always keep at least 65% cash (₹32,000+) uncommitted in your account.
                - **Why?** Even with an 80% AI hit rate, unexpected global overnight swings happen. By risking only 1 lot, a loss is easily absorbed and recovered on the next win!
                """)

        with r2:
            with st.container(border=True):
                st.markdown("### 🎯 Trade Execution Filter Checklist")
                st.markdown(r"""
                Before placing an order at 3:10 PM, ask these 4 questions:
                1. **Is the AI Probability High?** (Must be $\ge 65\%$ for CE, or $\le 35\%$ for PE).
                2. **Is the order type Delivery (NRML)?** (Never MIS).
                3. **Am I trading ATM strike?** (At-The-Money gives the best balance of delta vs theta).
                4. **Is my alarm set for 9:15 AM tomorrow?** (Exit in the first 5 minutes).
                """)

if __name__ == "__main__" or True:
    render_options_playbook_page()
