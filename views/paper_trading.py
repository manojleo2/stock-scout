import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.paper_trading import (
    load_paper_trades,
    evaluate_simulated_gap_exit,
    DEFAULT_STARTING_BANKROLL
)
from utils.ui_theme import apply_custom_theme

def render_paper_trading_page():
    apply_custom_theme()

    st.markdown("<div class='glowing-header'>💼 Paper Trading & Forward P&L Tracker</div>", unsafe_allow_html=True)
    
    col_sub, col_reload = st.columns([3, 1])
    with col_sub:
        st.markdown("<div class='sub-glow'>Forward Testing Simulator for 3:05 PM Gap Predictor & AI Up/Down +₹4 Scalp Strategies (Real Market Prices)</div>", unsafe_allow_html=True)
    with col_reload:
        if st.button("🔄 Force Clear Cache & Reload", use_container_width=True, help="Clears memory cache and re-computes all forward testing values"):
            st.cache_data.clear()
            st.rerun()

    # Dynamic Bankroll Selector (Defaults to ₹50,000 / 2 Lots)
    bankroll_option = st.radio(
        "💼 Select Starting Account Bankroll Sizing:",
        options=[
            "₹50,000 Starting Bankroll (2 Lots / 700 shares sizing — Standard Growth)",
            "₹20,000 Starting Bankroll (1 Lot / 350 shares sizing — Conservative Base)"
        ],
        index=0,
        horizontal=True
    )
    is_50k = "50,000" in bankroll_option
    active_starting_bankroll = 50000.0 if is_50k else 20000.0
    active_lot_mult = 1.0 if is_50k else 0.5  # Base ledger is calibrated at 2 lots (₹50k). 0.5 scales to 1 lot (₹20k).

    # Automatically evaluate completed sessions
    evaluate_simulated_gap_exit()
    raw_trades = load_paper_trades()

    # Scale trades dynamically based on chosen bankroll mode
    trades = []
    for t in raw_trades:
        t_copy = dict(t)
        base_lot = int(t.get("lot_size", 700))
        effective_lot = int(base_lot * active_lot_mult)
        t_copy["display_lot_size"] = effective_lot
        t_copy["display_capital"] = round(float(t.get("entry_premium", 0.0)) * effective_lot, 2)
        
        if t.get("net_pnl") is not None:
            is_zero_trade = t.get("lot_size", 0) == 0
            gross = round(float(t.get("gross_pnl", 0.0)) * active_lot_mult, 2) if not is_zero_trade else 0.0
            brok = 100.0 if not is_zero_trade else 0.0
            net = round(gross - brok, 2) if not is_zero_trade else 0.0
            t_copy["display_gross"] = gross
            t_copy["display_net"] = net
            t_copy["display_ret"] = round((net / t_copy["display_capital"]) * 100.0, 1) if t_copy["display_capital"] > 0 else 0.0
        else:
            t_copy["display_gross"] = None
            t_copy["display_net"] = None
            t_copy["display_ret"] = None
        trades.append(t_copy)

    # Compute summary with active starting bankroll
    completed = [t for t in trades if t.get("status") in ("✅ WIN", "❌ LOSS")]
    total_net_pnl = sum(float(t["display_net"]) for t in completed)
    total_gross_pnl = sum(float(t["display_gross"]) for t in completed)
    total_brokerage = len(completed) * 100.0
    current_capital = active_starting_bankroll + total_net_pnl
    wins = sum(1 for t in completed if t["display_net"] > 0)
    losses = len(completed) - wins
    win_rate_pct = round((wins / len(completed) * 100.0), 1) if completed else 0.0
    gross_wins = sum(float(t["display_net"]) for t in completed if t["display_net"] > 0)
    gross_losses = abs(sum(float(t["display_net"]) for t in completed if t["display_net"] <= 0))
    profit_factor = round((gross_wins / gross_losses), 2) if gross_losses > 0 else 1.0
    net_return_pct = round((total_net_pnl / active_starting_bankroll) * 100.0, 1)

    # Build dynamic equity curve
    running_cap = active_starting_bankroll
    running_pnl = 0.0
    equity_curve = [{"Date": "Initial", "Bankroll": active_starting_bankroll, "Net P&L": 0.0}]
    for t in completed:
        running_cap += t["display_net"]
        running_pnl += t["display_net"]
        d_lbl = t.get("exit_date") or t.get("entry_date") or "Trade"
        equity_curve.append({
            "Date": d_lbl,
            "Bankroll": round(running_cap, 2),
            "Net P&L": round(running_pnl, 2)
        })

    # 1. Virtual Bankroll Performance Cards
    st.markdown("### 🏦 Virtual Bankroll & Performance KPIs")
    lot_desc = "2 Lots (700 shares / ~₹20,000 deployment per trade)" if is_50k else "1 Lot (350 shares / ~₹10,000 deployment per trade)"
    st.caption(f"Active Mode: **₹{active_starting_bankroll:,.0f} Starting Capital** | Standard Sizing: **{lot_desc}** | Flat ₹100 round-trip fee.")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Starting Bankroll", f"₹{active_starting_bankroll:,.2f}")
    
    pnl_delta = f"{total_net_pnl:+,.2f} ₹ ({net_return_pct:+.1f}%)"
    k2.metric("Current Bankroll", f"₹{current_capital:,.2f}", delta=pnl_delta, delta_color="normal")
    
    k3.metric("Overall Win Rate", f"{win_rate_pct}%", f"{wins}W / {losses}L")
    k4.metric("Net Realized P&L", f"₹{total_net_pnl:+,.2f}", f"After -₹{total_brokerage:.0f} fees")
    k5.metric("Profit Factor", f"{profit_factor}", "Gross Win / Gross Loss")

    st.markdown("---")

    # 2. Cumulative Equity Curve Chart
    st.subheader("📈 Cumulative Virtual Bankroll Growth Curve")
    st.caption(f"Visualizes account equity progression starting from ₹{active_starting_bankroll:,.0f} baseline.")

    if equity_curve and len(equity_curve) > 1:
        df_curve = pd.DataFrame(equity_curve)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[f"#{i}: {r['Date']}" for i, r in enumerate(equity_curve)],
            y=df_curve["Bankroll"],
            mode="lines+markers",
            name="Bankroll (₹)",
            line=dict(color="#00E676", width=3),
            marker=dict(size=8, color="#38bdf8"),
            fill="tozeroy",
            fillcolor="rgba(0, 230, 118, 0.08)"
        ))

        # Add benchmark starting capital line
        fig.add_hline(
            y=active_starting_bankroll,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text=f"Starting Capital (₹{active_starting_bankroll:,.0f})",
            annotation_position="bottom right"
        )

        fig.update_layout(
            height=300,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Account Balance (₹)",
            xaxis_title="Trade Timeline"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 3. Strategy Filters & Performance Breakdown
    st.subheader("📋 Forward Trade Log & Prediction Audit")
    st.caption(f"Review exactly what the AI predicted (expected) versus what happened in real market trading, with exact rupee returns based on {lot_desc}.")

    strat_filter = st.radio(
        "Filter by Strategy",
        options=["All Strategies", "3:05 PM Gap Overnight", "AI Intraday +₹4 Scalp"],
        horizontal=True
    )

    filtered_trades = trades
    if strat_filter != "All Strategies":
        filtered_trades = [t for t in trades if t.get("strategy") == strat_filter]

    # Strategy breakdown metrics
    gap_trades = [t for t in trades if t.get("strategy") == "3:05 PM Gap Overnight" and t.get("status") in ("✅ WIN", "❌ LOSS")]
    scalp_trades = [t for t in trades if t.get("strategy") == "AI Intraday +₹4 Scalp" and t.get("status") in ("✅ WIN", "❌ LOSS")]

    sb1, sb2 = st.columns(2)
    with sb1:
        gap_pnl = sum(float(t["display_net"]) for t in gap_trades)
        gap_wins = sum(1 for t in gap_trades if t.get("status") == "✅ WIN")
        gap_win_rate = round((gap_wins / len(gap_trades) * 100.0), 1) if gap_trades else 0.0
        st.info(f"🌙 **3:05 PM Gap Overnight Strategy:** {gap_wins}/{len(gap_trades)} Wins ({gap_win_rate}%) | Net P&L: **₹{gap_pnl:+,.2f}**")

    with sb2:
        scalp_pnl = sum(float(t["display_net"]) for t in scalp_trades)
        scalp_wins = sum(1 for t in scalp_trades if t.get("status") == "✅ WIN")
        scalp_win_rate = round((scalp_wins / len(scalp_trades) * 100.0), 1) if scalp_trades else 0.0
        st.info(f"⚡ **AI Intraday +₹4 Scalp Strategy:** {scalp_wins}/{len(scalp_trades)} Wins ({scalp_win_rate}%) | Net P&L: **₹{scalp_pnl:+,.2f}**")

    # 4. Detailed Trade Log Table
    if filtered_trades:
        df_log = pd.DataFrame([
            {
                "Date": f"{t.get('exit_date', t.get('entry_date'))}",
                "Strategy": t.get("strategy"),
                "Stock": t.get("symbol"),
                "Strike & Action": f"{t.get('action')} ({t.get('strike')})",
                "Lots & Qty": "0 Lots (Cash)" if t.get('display_lot_size', 0) == 0 else f"{t.get('display_lot_size') // 350 if 'CDSL' in t.get('symbol', '') else t.get('display_lot_size') // 500} Lot(s) ({t.get('display_lot_size')} sh)",
                "Capital Deployed": "₹0.00 (100% Cash)" if t.get('display_lot_size', 0) == 0 else f"₹{t.get('display_capital', 0):,.2f}",
                "Entry ₹": "N/A" if t.get('display_lot_size', 0) == 0 else f"₹{t.get('entry_premium', 0):.2f}",
                "Exit ₹": "N/A (Cash)" if t.get('display_lot_size', 0) == 0 else (f"₹{t.get('exit_premium', 0):.2f}" if t.get('exit_premium') is not None else "Pending..."),
                "Net P&L (₹)": "₹0.00" if t.get('display_lot_size', 0) == 0 else (f"₹{t.get('display_net', 0):+,.2f}" if t.get('display_net') is not None else "Pending..."),
                "Return %": "0.0%" if t.get('display_lot_size', 0) == 0 else (f"{t.get('display_ret', 0):+.1f}%" if t.get('display_ret') is not None else "Pending..."),
                "Result": t.get("status"),
                "What Was Expected": t.get("what_was_expected", "N/A"),
                "What Had Happened": t.get("what_had_happened", "N/A")
            } for t in reversed(filtered_trades)
        ])

        st.dataframe(
            df_log,
            use_container_width=True,
            hide_index=True,
            column_config={
                "What Was Expected": st.column_config.TextColumn("🎯 What Was Expected", width="medium"),
                "What Had Happened": st.column_config.TextColumn("⚡ What Had Happened", width="large"),
                "Capital Deployed": st.column_config.TextColumn("Capital Deployed", width="small"),
                "Lots & Qty": st.column_config.TextColumn("Position", width="small"),
                "Net P&L (₹)": st.column_config.TextColumn("Net P&L (₹)", width="small"),
                "Result": st.column_config.TextColumn("Result", width="small")
            }
        )
    else:
        st.info("No paper trades recorded yet for this strategy.")

    st.markdown("---")

    # 5. Operational Guidelines
    with st.expander("ℹ️ How This Forward Testing Tracker Protects Your Money", expanded=False):
        st.markdown(f"""
        - **Why Paper Trade First?** 
          A model may have a high win rate on paper, but if you don't know the exact rupee payoff (e.g. $+₹3,050$ on wins vs $-₹2,340$ on losses), you cannot trade with confidence.
        - **Realism Built In:** 
          Every simulated trade deducts **₹100 flat** for broker commissions (Groww ~₹40 round trip) and STT/turnover taxes.
        - **Capital Sizing for ₹50,000 Account:**
          Deploying **2 lots (~₹18,000–₹22,000)** commits ~40% of capital, leaving a safe **60% cash buffer (₹30,000+)** in reserve.
        - **Execution Timers:**
          - **3:05 PM Gap Overnight:** Entry at 3:10 PM, exit strictly at **9:18 AM** the next morning.
          - **AI Intraday Scalp:** Entry at morning breakout (9:20 AM), exit when option premium gains **+₹4.00** (+₹2,800 to +₹3,150 on 2 lots), or stop-loss hits.
        """)

if __name__ == "__main__" or True:
    render_paper_trading_page()
