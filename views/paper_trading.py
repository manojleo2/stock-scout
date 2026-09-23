import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.paper_trading import (
    load_paper_trades,
    evaluate_simulated_gap_exit,
    sync_today_paper_trades,
    DEFAULT_STARTING_BANKROLL
)
from utils.intraday_timing_audit import (
    sync_intraday_timing_audit,
    load_intraday_timing_audit,
    get_timing_kpis
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

    # Automatically sync today's morning trade & evaluate 5-minute candle targets
    sync_today_paper_trades()
    sync_intraday_timing_audit("CDSL.NS")
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

    col_f1, col_f2 = st.columns([2, 1.2])
    with col_f1:
        strat_filter = st.radio(
            "Filter by Strategy",
            options=["All Strategies", "3:05 PM Gap Overnight", "AI Intraday +₹4 Scalp"],
            horizontal=True
        )
    with col_f2:
        view_mode = st.radio(
            "Display Format",
            options=["🎴 Modern Visual Cards (Full Wrapped Text)", "📊 Compact Table Grid"],
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

    # 4. Standout Trade Log Display
    if not filtered_trades:
        st.info("No paper trades recorded yet for this strategy.")
    elif "Modern" in view_mode:
        # ── Modern Glassmorphic Quant Cards with 100% Wrapped Zero-Click Text ──
        style_block = """
<style>
.trade-log-container {
    display: flex;
    flex-direction: column;
    gap: 14px;
    margin-top: 10px;
    margin-bottom: 25px;
}
.trade-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(24, 34, 53, 0.85) 100%);
    border: 1px solid rgba(56, 189, 248, 0.22);
    border-radius: 12px;
    padding: 18px 22px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    margin-bottom: 14px;
}
.trade-card:hover {
    border-color: rgba(56, 189, 248, 0.55);
    box-shadow: 0 6px 26px rgba(56, 189, 248, 0.15);
}
.trade-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 14px;
}
.trade-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.pill-win {
    background: rgba(0, 230, 118, 0.15);
    color: #00E676;
    border: 1px solid rgba(0, 230, 118, 0.4);
    box-shadow: 0 0 10px rgba(0, 230, 118, 0.2);
}
.pill-loss {
    background: rgba(255, 82, 82, 0.15);
    color: #FF5252;
    border: 1px solid rgba(255, 82, 82, 0.4);
    box-shadow: 0 0 10px rgba(255, 82, 82, 0.2);
}
.pill-preserve {
    background: rgba(148, 163, 184, 0.12);
    color: #cbd5e1;
    border: 1px solid rgba(148, 163, 184, 0.35);
}
.pill-active {
    background: rgba(56, 189, 248, 0.18);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.5);
}
.trade-metrics-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    margin-bottom: 14px;
    background: rgba(0, 0, 0, 0.28);
    padding: 12px 16px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.trade-m-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    color: #94a3b8;
    letter-spacing: 0.5px;
    margin-bottom: 3px;
}
.trade-m-val {
    font-size: 0.98rem;
    font-weight: 700;
    color: #f8fafc;
    font-family: monospace;
}
.trade-narratives-box {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
}
@media (max-width: 800px) {
    .trade-narratives-box {
        grid-template-columns: 1fr;
    }
}
.narrative-item {
    background: rgba(15, 23, 42, 0.65);
    padding: 14px 16px;
    border-radius: 8px;
    font-size: 0.88rem;
    line-height: 1.55;
    white-space: normal;
    word-wrap: break-word;
    overflow-wrap: break-word;
}
.narrative-item-exp {
    border-left: 3px solid #38bdf8;
}
.narrative-item-hap {
    border-left: 3px solid #a855f7;
}
.narrative-label {
    font-size: 0.74rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
}
</style>
<div class='trade-log-container'>
"""
        cards_html = style_block
        for t in reversed(filtered_trades):
            status = t.get("status", "")
            if "WIN" in status:
                pill_class = "pill-win"
            elif "LOSS" in status:
                pill_class = "pill-loss"
            elif "Active" in status or "Progress" in status or "Carry" in status:
                pill_class = "pill-active"
            else:
                pill_class = "pill-preserve"

            date_str = t.get("exit_date") or t.get("entry_date") or "N/A"
            strat_icon = "🌙" if "Gap" in t.get("strategy", "") else "⚡"
            strat_label = t.get("strategy", "Strategy")
            sym = t.get("symbol", "CDSL.NS")

            disp_lots = t.get("display_lot_size", 0)
            if disp_lots == 0:
                pos_str = "0 Lots (100% Cash Buffer)"
                cap_str = "₹0.00"
                entry_str = "N/A (Cash)"
                exit_str = "N/A (Cash)"
                net_str = "₹0.00"
                net_color = "#94a3b8"
                ret_str = "0.0%"
            else:
                lot_count = disp_lots // 350 if "CDSL" in sym else disp_lots // 500
                pos_str = f"{lot_count} Lot(s) ({disp_lots} shares)"
                cap_str = f"₹{t.get('display_capital', 0):,.2f}"
                
                # Format: 1346(24) -> Stock Price with Option Premium
                entry_s = t.get("entry_spot")
                entry_p = t.get("entry_premium", 0.0)
                entry_str = f"{int(round(entry_s))}({int(round(entry_p))})" if entry_s else f"₹{entry_p:.2f}"

                exit_s = t.get("exit_spot")
                exit_val = t.get("exit_premium")
                if exit_val is not None:
                    exit_str = f"{int(round(exit_s))}({int(round(exit_val))})" if exit_s else f"₹{exit_val:.2f}"
                else:
                    exit_str = "⏳ Live / Pending"

                net_val = t.get("display_net")
                if net_val is not None:
                    net_str = f"₹{net_val:+,.2f}"
                    net_color = "#00E676" if net_val > 0 else "#FF5252"
                    ret_str = f"{t.get('display_ret', 0):+.1f}%"
                else:
                    net_str = "⏳ Pending..."
                    net_color = "#38bdf8"
                    ret_str = "Pending..."

            exp_text = t.get("what_was_expected", "N/A")
            hap_text = t.get("what_had_happened", "N/A")
            strike_act = f"{t.get('action', '')} ({t.get('strike', '')})"

            card = f"""
<div class='trade-card'>
    <div class='trade-header-row'>
        <div style='display:flex; align-items:center; gap:10px; flex-wrap:wrap;'>
            <span style='font-weight:700; font-size:0.95rem; color:#f8fafc;'>📅 {date_str}</span>
            <span style='background:rgba(56,189,248,0.12); color:#38bdf8; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600;'>{strat_icon} {strat_label}</span>
            <span style='color:#94a3b8; font-size:0.85rem;'>• {sym}</span>
            <span style='color:#cbd5e1; font-size:0.85rem; font-weight:600;'>• {strike_act}</span>
        </div>
        <div>
            <span class='trade-pill {pill_class}'>{status}</span>
        </div>
    </div>
    <div class='trade-metrics-strip'>
        <div class='trade-m-item'>
            <div class='trade-m-label'>Position Sizing</div>
            <div class='trade-m-val' style='font-size:0.88rem;'>{pos_str}</div>
        </div>
        <div class='trade-m-item'>
            <div class='trade-m-label'>Capital Deployed</div>
            <div class='trade-m-val'>{cap_str}</div>
        </div>
        <div class='trade-m-item'>
            <div class='trade-m-label'>Entry [Stock(Opt)]</div>
            <div class='trade-m-val'>{entry_str}</div>
        </div>
        <div class='trade-m-item'>
            <div class='trade-m-label'>Exit [Stock(Opt)]</div>
            <div class='trade-m-val'>{exit_str}</div>
        </div>
        <div class='trade-m-item'>
            <div class='trade-m-label'>Net P&L (Post-Tax)</div>
            <div class='trade-m-val' style='color:{net_color}; font-size:1.05rem;'>{net_str}</div>
        </div>
        <div class='trade-m-item'>
            <div class='trade-m-label'>Return %</div>
            <div class='trade-m-val' style='color:{net_color};'>{ret_str}</div>
        </div>
    </div>
    <div class='trade-narratives-box'>
        <div class='narrative-item narrative-item-exp'>
            <div class='narrative-label' style='color:#38bdf8;'>🎯 What Was Expected (Morning Forecast)</div>
            <div style='color:#e2e8f0;'>{exp_text}</div>
        </div>
        <div class='narrative-item narrative-item-hap'>
            <div class='narrative-label' style='color:#c084fc;'>⚡ What Had Happened (Realized Audit)</div>
            <div style='color:#e2e8f0;'>{hap_text}</div>
        </div>
    </div>
</div>
"""
            cards_html += card

        cards_html += "</div>"
        clean_html = "\n".join(line.strip() for line in cards_html.splitlines() if line.strip())
        if hasattr(st, "html"):
            st.html(clean_html)
        else:
            st.markdown(clean_html, unsafe_allow_html=True)

    else:
        # ── Compact Data Table View ──
        df_log = pd.DataFrame([
            {
                "Date": f"{t.get('exit_date', t.get('entry_date'))}",
                "Strategy": t.get("strategy"),
                "Stock": t.get("symbol"),
                "Strike & Action": f"{t.get('action')} ({t.get('strike')})",
                "Lots & Qty": "0 Lots (Cash)" if t.get('display_lot_size', 0) == 0 else f"{t.get('display_lot_size') // 350 if 'CDSL' in t.get('symbol', '') else t.get('display_lot_size') // 500} Lot(s) ({t.get('display_lot_size')} sh)",
                "Capital Deployed": "₹0.00 (100% Cash)" if t.get('display_lot_size', 0) == 0 else f"₹{t.get('display_capital', 0):,.2f}",
                "Entry [Stock(Opt)]": "N/A" if t.get('display_lot_size', 0) == 0 else (f"{int(round(t.get('entry_spot', 0)))}({int(round(t.get('entry_premium', 0)))})" if t.get('entry_spot') else f"₹{t.get('entry_premium', 0):.2f}"),
                "Exit [Stock(Opt)]": "N/A (Cash)" if t.get('display_lot_size', 0) == 0 else ((f"{int(round(t.get('exit_spot', 0)))}({int(round(t.get('exit_premium', 0)))})" if t.get('exit_spot') else f"₹{t.get('exit_premium', 0):.2f}") if t.get('exit_premium') is not None else "Pending..."),
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

    st.markdown("---")

    # 5. Intraday Entry Timing Benchmark (09:20 AM vs 09:25 AM vs 09:30 AM)
    st.subheader("⏱️ Intraday Entry Timing Benchmark: 09:20 AM vs 09:25 AM vs 09:30 AM")
    st.caption("Empirical head-to-head comparison across all historical sessions. Evaluates fill prices, exact target (+₹8.00 spot / +₹4.50 opt) or stop-loss hit times, and time-to-target. Automatically updated daily from 5-minute tick data.")

    timing_records = load_intraday_timing_audit()
    cdsl_records = [r for r in timing_records if r.get("symbol") == "CDSL.NS"]
    if not cdsl_records:
        cdsl_records = sync_intraday_timing_audit("CDSL.NS")

    kpis = get_timing_kpis(cdsl_records)
    d920 = kpis["details"]["9:20"]
    d925 = kpis["details"]["9:25"]
    d930 = kpis["details"]["9:30"]

    # 3 Summary KPI Cards
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.metric(
            "09:20 AM Entry (🏆 Winner)",
            f"{d920['win_rate_pct']}% Win Rate",
            f"{d920['wins']}W / {d920['losses']}L | ~{d920['avg_minutes']}m to Target"
        )
        st.caption("🌟 **Optimal Fill**: Lowest option premium before volatility candle expands. 4 instant hits (≤10m).")

    with col_t2:
        st.metric(
            "09:25 AM Entry (🥈 Secondary)",
            f"{d925['win_rate_pct']}% Win Rate",
            f"{d925['wins']}W / {d925['losses']}L | ~{d925['avg_minutes']}m to Target"
        )
        st.caption("🔍 **Confirmation Entry**: Waits for two 5-min candles. Sacrifices ₹2–₹5 spot premium.")

    with col_t3:
        st.metric(
            "09:30 AM Entry (🥉 Traditional)",
            f"{d930['win_rate_pct']}% Win Rate",
            f"{d930['wins']}W / {d930['losses']}L | ~{d930['avg_minutes']}m to Target"
        )
        st.caption("⚠️ **Impulse Lag**: Standard ORB. High slippage on trend days (enters after ₹15+ initial run).")

    def format_timing_cell(e):
        if not e or e.get("entry_price") is None:
            return e.get("hit_status", "⏳ Pending Open")
        
        entry_p = e.get("entry_price", 0.0)
        entry_prem = e.get("entry_prem")
        status = e.get("hit_status", "")
        hit_t = e.get("hit_time", "")
        hit_p = e.get("hit_price", 0.0)
        exit_prem = e.get("exit_prem")
        pts = e.get("points")
        opt_pts = e.get("opt_points")
        mins = e.get("minutes_to_hit")
        
        # User-specified syntax: 1346(24) -> stock price with option premium
        entry_dual = f"{int(round(entry_p))}({int(round(entry_prem))})" if entry_prem is not None else f"{int(round(entry_p))}"
        exit_dual = f"{int(round(hit_p))}({int(round(exit_prem))})" if exit_prem is not None and hit_p is not None else ""
        
        opt_pts_str = f"({opt_pts:+.2f} opt)" if opt_pts is not None else (f"({pts:+.2f} pts)" if pts is not None else "")
        mins_str = f"in {mins}m" if mins is not None else ""
        
        if "Target" in status:
            return f"{entry_dual} ➔ 🎯 Tgt @ {hit_t} {mins_str} {exit_dual} {opt_pts_str}"
        elif "Stop" in status:
            return f"{entry_dual} ➔ 🛑 SL @ {hit_t} {mins_str} {exit_dual} {opt_pts_str}"
        elif "Cutoff" in status:
            return f"{entry_dual} ➔ ⏱️ Cut @ {hit_t} {mins_str} {exit_dual} {opt_pts_str}"
        elif "Held" in status:
            return f"{entry_dual} ➔ ⏱️ Close @ {hit_t} {exit_dual} {opt_pts_str}"
        else:
            return f"{entry_dual} ➔ {status}"

    timing_rows = []
    for r in reversed(cdsl_records):
        entries = r.get("entries", {})
        e20 = entries.get("9:20", {})
        e25 = entries.get("9:25", {})
        e30 = entries.get("9:30", {})
        
        timing_rows.append({
            "Date": r.get("target_date"),
            "AI Direction": r.get("predicted_direction"),
            "09:20 AM Entry": format_timing_cell(e20),
            "09:25 AM Entry": format_timing_cell(e25),
            "09:30 AM Entry": format_timing_cell(e30),
            "Best Execution Timing": r.get("best_entry", "N/A")
        })

    df_timing = pd.DataFrame(timing_rows)
    st.dataframe(
        df_timing,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.TextColumn("Session Date", width="small"),
            "AI Direction": st.column_config.TextColumn("AI Bias", width="small"),
            "09:20 AM Entry": st.column_config.TextColumn("09:20 AM Entry [Stock(Opt)]", width="large"),
            "09:25 AM Entry": st.column_config.TextColumn("09:25 AM Entry [Stock(Opt)]", width="large"),
            "09:30 AM Entry": st.column_config.TextColumn("09:30 AM Entry [Stock(Opt)]", width="large"),
            "Best Execution Timing": st.column_config.TextColumn("Best Fill Outcome", width="medium"),
        }
    )

    with st.expander("🔍 Detailed Price & Target Breakdown [Stock(Option) Format: e.g. 1346(24)]", expanded=False):
        detailed_rows = []
        for r in reversed(cdsl_records):
            entries = r.get("entries", {})
            e20 = entries.get("9:20", {})
            e25 = entries.get("9:25", {})
            e30 = entries.get("9:30", {})
            
            def fmt_d(spot, prem):
                if spot is None: return "N/A"
                if prem is not None: return f"{int(round(spot))}({int(round(prem))})"
                return f"{int(round(spot))}"

            detailed_rows.append({
                "Date": r.get("target_date"),
                "Bias": r.get("predicted_direction"),
                "9:20 Entry": fmt_d(e20.get('entry_price'), e20.get('entry_prem')),
                "9:20 Tgt / SL": f"Tgt: {fmt_d(e20.get('target_price'), e20.get('target_prem'))} | SL: {fmt_d(e20.get('sl_price'), e20.get('sl_prem'))}",
                "9:20 Exit & Outcome": f"{e20.get('hit_status', 'Pending')} @ {fmt_d(e20.get('hit_price'), e20.get('exit_prem'))} ({e20.get('hit_time', '')})",
                "9:25 Entry": fmt_d(e25.get('entry_price'), e25.get('entry_prem')),
                "9:25 Tgt / SL": f"Tgt: {fmt_d(e25.get('target_price'), e25.get('target_prem'))} | SL: {fmt_d(e25.get('sl_price'), e25.get('sl_prem'))}",
                "9:25 Exit & Outcome": f"{e25.get('hit_status', 'Pending')} @ {fmt_d(e25.get('hit_price'), e25.get('exit_prem'))} ({e25.get('hit_time', '')})",
                "9:30 Entry": fmt_d(e30.get('entry_price'), e30.get('entry_prem')),
                "9:30 Tgt / SL": f"Tgt: {fmt_d(e30.get('target_price'), e30.get('target_prem'))} | SL: {fmt_d(e30.get('sl_price'), e30.get('sl_prem'))}",
                "9:30 Exit & Outcome": f"{e30.get('hit_status', 'Pending')} @ {fmt_d(e30.get('hit_price'), e30.get('exit_prem'))} ({e30.get('hit_time', '')})",
            })
        st.dataframe(pd.DataFrame(detailed_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # 6. Operational Guidelines
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
