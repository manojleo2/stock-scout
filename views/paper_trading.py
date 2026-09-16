import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.paper_trading import (
    load_paper_trades,
    get_portfolio_performance_summary,
    evaluate_simulated_gap_exit,
    DEFAULT_STARTING_BANKROLL
)
from utils.ui_theme import apply_custom_theme

def render_paper_trading_page():
    apply_custom_theme()

    st.markdown("<div class='glowing-header'>💼 Paper Trading & Forward P&L Tracker</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>Forward Testing Simulator for 3:05 PM Gap Predictor & AI Up/Down +₹4 Scalp Strategies (Real Market Prices)</div>", unsafe_allow_html=True)

    # Automatically evaluate completed sessions
    evaluate_simulated_gap_exit()
    trades = load_paper_trades()

    summary = get_portfolio_performance_summary(trades)

    # 1. Virtual Bankroll Performance Cards
    st.markdown("### 🏦 Virtual Bankroll & Performance KPIs")
    st.caption("Tracks forward-tested trades using ₹50,000 starting virtual capital and realistic ₹100 round-trip brokerage deduction.")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Starting Bankroll", f"₹{summary['starting_capital']:,.2f}")
    
    pnl_delta = f"{summary['total_net_pnl']:+,.2f} ₹ ({summary['net_return_pct']:+.1f}%)"
    k2.metric("Current Bankroll", f"₹{summary['current_capital']:,.2f}", delta=pnl_delta, delta_color="normal")
    
    k3.metric("Overall Win Rate", f"{summary['win_rate_pct']}%", f"{summary['wins']}W / {summary['losses']}L")
    k4.metric("Net Realized P&L", f"₹{summary['total_net_pnl']:+,.2f}", f"After -₹{summary['total_brokerage']:.0f} fees")
    k5.metric("Profit Factor", f"{summary['profit_factor']}", "Gross Win / Gross Loss")

    st.markdown("---")

    # 2. Cumulative Equity Curve Chart
    st.subheader("📈 Cumulative Virtual Bankroll Growth Curve")
    st.caption("Visualizes account equity progression trade-by-trade.")

    curve_data = summary.get("equity_curve", [])
    if curve_data and len(curve_data) > 1:
        df_curve = pd.DataFrame(curve_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[f"#{i}: {r['Date']}" for i, r in enumerate(curve_data)],
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
            y=DEFAULT_STARTING_BANKROLL,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text=f"Starting Capital (₹{DEFAULT_STARTING_BANKROLL:,.0f})",
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
    st.caption("Review exactly what the AI predicted (expected) versus what happened in real market trading, with exact rupee returns.")

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
        gap_pnl = sum(float(t.get("net_pnl", 0)) for t in gap_trades)
        gap_wins = sum(1 for t in gap_trades if t.get("status") == "✅ WIN")
        gap_win_rate = round((gap_wins / len(gap_trades) * 100.0), 1) if gap_trades else 0.0
        st.info(f"🌙 **3:05 PM Gap Overnight Strategy:** {gap_wins}/{len(gap_trades)} Wins ({gap_win_rate}%) | Net P&L: **₹{gap_pnl:+,.2f}**")

    with sb2:
        scalp_pnl = sum(float(t.get("net_pnl", 0)) for t in scalp_trades)
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
                "What Was Expected": t.get("what_was_expected", "N/A"),
                "What Had Happened": t.get("what_had_happened", "N/A"),
                "Entry ₹": f"₹{t.get('entry_premium', 0):.2f}",
                "Exit ₹": f"₹{t.get('exit_premium', 0):.2f}" if t.get('exit_premium') else "Pending...",
                "Net P&L (₹)": f"₹{t.get('net_pnl', 0):+,.2f}" if t.get('net_pnl') is not None else "Pending...",
                "Return %": f"{t.get('return_pct', 0):+.1f}%" if t.get('return_pct') is not None else "Pending...",
                "Result": t.get("status")
            } for t in reversed(filtered_trades)
        ])

        st.dataframe(
            df_log,
            use_container_width=True,
            hide_index=True,
            column_config={
                "What Was Expected": st.column_config.TextColumn("🎯 What Was Expected", width="medium"),
                "What Had Happened": st.column_config.TextColumn("⚡ What Had Happened", width="large"),
                "Net P&L (₹)": st.column_config.TextColumn("Net P&L (₹)", width="small"),
                "Result": st.column_config.TextColumn("Result", width="small")
            }
        )
    else:
        st.info("No paper trades recorded yet for this strategy.")

    st.markdown("---")

    # 5. Operational Guidelines
    with st.expander("ℹ️ How This Forward Testing Tracker Protects Your Money", expanded=False):
        st.markdown("""
        - **Why Paper Trade First?** 
          A model may have a high win rate on paper, but if you don't know the exact rupee payoff (e.g. $+₹2,100$ on wins vs $-₹1,600$ on losses), you cannot trade with confidence.
        - **Realism Built In:** 
          Every simulated trade deducts **₹100 flat** for broker commissions (Groww ~₹40 round trip) and STT/turnover taxes.
        - **Execution Timers:**
          - **3:05 PM Gap Overnight:** Entry at 3:10 PM, exit strictly at **9:18 AM** the next morning.
          - **AI Intraday Scalp:** Entry at morning breakout, exit when option premium gains **+₹4.00** (₹1,400 per lot), or stop-loss hits.
        """)

if __name__ == "__main__" or True:
    render_paper_trading_page()
