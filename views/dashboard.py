import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.data_loader import get_stock_fundamentals, get_stock_data
from utils.nifty_correlation import analyze_nifty_impact
from utils.demat_analytics import render_demat_analytics_widget
from utils.market_calendar import is_trading_holiday
from utils.ui_theme import apply_custom_theme
from utils.manual_trades import (
    load_manual_sub60_trades,
    add_or_update_manual_trade,
    delete_manual_trade_by_index
)
from config import BENCHMARK_TICKER, STOCK_NAME_MAP


@st.fragment(run_every="20s")
def render_live_stock_cards_fragment(watchlist: list):
    """
    Auto-refreshes live stock cards and Nifty index every 20 seconds during market hours.
    """
    import datetime as dt
    now_ist = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5, minutes=30)
    current_time_str = now_ist.strftime("%I:%M %p IST")
    weekday = now_ist.weekday()
    is_holiday, holiday_name = is_trading_holiday(now_ist.date())

    market_open = (weekday < 5) and (not is_holiday) and (dt.time(9, 15) <= now_ist.time() <= dt.time(15, 30))
    settling = (weekday < 5) and (not is_holiday) and (dt.time(15, 30) < now_ist.time() <= dt.time(15, 45))

    if is_holiday:
        status_badge = f"🏖️ MARKET CLOSED (HOLIDAY: {holiday_name.upper()})"
        status_color = "#38bdf8"
        status_sub = f"NSE & BSE closed for {holiday_name} • Trading resumes next session"
    elif market_open:
        status_badge = "🟢 MARKET OPEN (Live Trading)"
        status_color = "#00E676"
        status_sub = "Live quotes auto-refreshing every 5 seconds"
    elif settling:
        status_badge = "⏳ POST-MARKET RECONCILIATION"
        status_color = "#FFB300"
        status_sub = "Exchange computing 30-min VWAP closing settlement (3:30 - 3:45 PM IST)"
    else:
        status_badge = "🔴 MARKET CLOSED"
        status_color = "#94a3b8"
        status_sub = "Session ended • Official closing prices finalized"

    st.markdown(f"""
        <div style='display:flex; justify-content:space-between; align-items:center; background:rgba(30, 41, 59, 0.4); padding:8px 16px; border-radius:8px; margin-bottom:14px; border:1px solid rgba(255,255,255,0.06);'>
            <div style='display:flex; align-items:center; gap:10px;'>
                <span style='color:{status_color}; font-weight:700; font-size:0.95rem;'>{status_badge}</span>
                <span style='color:#64748b;'>|</span>
                <span style='color:#94a3b8; font-size:0.85rem;'>{status_sub}</span>
            </div>
            <span style='color:#94a3b8; font-size:0.85rem; font-family:monospace;'>🕒 {current_time_str}</span>
        </div>
    """, unsafe_allow_html=True)

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
    st.markdown("<div class='sub-glow'>24x7 Real-Time NSE/BSE Stock Monitoring Engine — CDSL, HDFCBANK & Custom Watchlist</div>", unsafe_allow_html=True)

    watchlist = st.session_state.get("watchlist", ["CDSL.NS", "HDFCBANK.NS"])

    # Render 5s Auto-Refreshing Fragment
    render_live_stock_cards_fragment(watchlist)

    st.markdown("<br>", unsafe_allow_html=True)

def render_sub60_manual_tracker():
    """
    Renders the Sub-60% Probability Discretionary Trade Tracker table & interactive logging form.
    Permits manual logging and auditing of CDSL options trades on low-conviction days (<60%)
    to verify target (+₹10) and stop-loss (-₹5) hit rates.
    """
    st.markdown("### 📝 Sub-60% Probability Discretionary Trade Tracker")
    st.caption("Manual forward audit ledger for low-conviction sessions (<60% AI probability) where automated execution is sidelined. Log and track option entry, target (+₹10), and stop-loss (-₹5) outcomes in real market price action.")

    trades = load_manual_sub60_trades()

    # KPI summary cards
    total_trades = len(trades)
    wins = sum(1 for t in trades if "WIN" in str(t.get("Result", "")))
    losses = sum(1 for t in trades if "LOSS" in str(t.get("Result", "")))
    in_progress = sum(1 for t in trades if any(k in str(t.get("Result", "")) for k in ["Progress", "⏳", "Awaiting", "Pending"]))
    completed = wins + losses
    win_rate = (wins / completed * 100.0) if completed > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Sessions Logged", f"{total_trades}")
    k2.metric("Discretionary Win Rate", f"{win_rate:.1f}%" if completed > 0 else "N/A", f"{wins}W / {losses}L" if completed > 0 else "0 Completed")
    k3.metric("Pending / In Progress", f"{in_progress} Sessions")
    k4.metric("Rulebook Protocol", "T: +₹10.00 | SL: -₹5.00", "03:05 PM Cutoff")

    # Render Colorful Glassmorphic Table
    if trades:
        html = """
<div style='overflow-x: auto; border-radius: 14px; border: 1px solid rgba(56, 189, 248, 0.35); box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5); margin: 16px 0 24px 0; background: linear-gradient(180deg, #090e17 0%, #0d1527 100%);'>
<table style='width: 100%; border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 0.88rem; text-align: left;'>
<thead>
<tr style='background: linear-gradient(90deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%); border-bottom: 2px solid rgba(56, 189, 248, 0.45);'>
<th style='padding: 14px 16px; color: #38bdf8; font-weight: 700; text-transform: uppercase; font-size: 0.78rem; letter-spacing: 0.5px; white-space: nowrap;'>Date</th>
<th style='padding: 14px 16px; color: #38bdf8; font-weight: 700; text-transform: uppercase; font-size: 0.78rem; letter-spacing: 0.5px; white-space: nowrap;'>Contract</th>
<th style='padding: 14px 16px; color: #38bdf8; font-weight: 700; text-transform: uppercase; font-size: 0.78rem; letter-spacing: 0.5px; white-space: nowrap;'>Stock</th>
<th style='padding: 14px 16px; color: #c084fc; font-weight: 700; text-transform: uppercase; font-size: 0.78rem; letter-spacing: 0.5px; white-space: nowrap;'>AI Probability</th>
<th style='padding: 14px 16px; color: #38bdf8; font-weight: 700; text-transform: uppercase; font-size: 0.78rem; letter-spacing: 0.5px; white-space: nowrap;'>Entry</th>
<th style='padding: 14px 16px; color: #f43f5e; font-weight: 700; text-transform: uppercase; font-size: 0.78rem; letter-spacing: 0.5px; white-space: nowrap;'>AI Prediction</th>
<th style='padding: 14px 16px; color: #4ade80; font-weight: 700; text-transform: uppercase; font-size: 0.78rem; letter-spacing: 0.5px; white-space: nowrap;'>T / SL</th>
<th style='padding: 14px 16px; color: #e2e8f0; font-weight: 700; text-transform: uppercase; font-size: 0.78rem; letter-spacing: 0.5px;'>What Happened</th>
<th style='padding: 14px 16px; color: #fbbf24; font-weight: 700; text-transform: uppercase; font-size: 0.78rem; letter-spacing: 0.5px; white-space: nowrap;'>Exit</th>
<th style='padding: 14px 16px; color: #a5b4fc; font-weight: 700; text-transform: uppercase; font-size: 0.78rem; letter-spacing: 0.5px; white-space: nowrap;'>Cutoff</th>
<th style='padding: 14px 16px; color: #4ade80; font-weight: 700; text-transform: uppercase; font-size: 0.78rem; letter-spacing: 0.5px; white-space: nowrap;'>Result</th>
</tr>
</thead>
<tbody>
"""
        for r in reversed(trades):
            pred = str(r.get("AI Prediction", ""))
            if "PUT" in pred or "PE" in pred or "DOWN" in pred:
                pred_pill = f"<span style='background: rgba(244, 63, 94, 0.2); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.5); padding: 4px 10px; border-radius: 20px; font-weight: 700; font-size: 0.80rem;'>🔻 {pred}</span>"
            elif "CALL" in pred or "CE" in pred or "UP" in pred:
                pred_pill = f"<span style='background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.5); padding: 4px 10px; border-radius: 20px; font-weight: 700; font-size: 0.80rem;'>🔺 {pred}</span>"
            else:
                pred_pill = f"<span style='background: rgba(148, 163, 184, 0.2); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.4); padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.80rem;'>⚪ {pred}</span>"

            res = str(r.get("Result", ""))
            if "WIN" in res:
                res_pill = f"<span style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.3) 0%, rgba(16, 185, 129, 0.2) 100%); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.7); box-shadow: 0 0 12px rgba(34, 197, 94, 0.25); padding: 5px 12px; border-radius: 20px; font-weight: 800; font-size: 0.82rem;'>{res}</span>"
            elif "LOSS" in res:
                res_pill = f"<span style='background: linear-gradient(135deg, rgba(239, 68, 68, 0.3) 0%, rgba(244, 63, 94, 0.2) 100%); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.7); box-shadow: 0 0 12px rgba(239, 68, 68, 0.25); padding: 5px 12px; border-radius: 20px; font-weight: 800; font-size: 0.82rem;'>{res}</span>"
            elif "Progress" in res or "⏳" in res or "Awaiting" in res or "Pending" in res:
                res_pill = f"<span style='background: linear-gradient(135deg, rgba(245, 158, 11, 0.25) 0%, rgba(234, 179, 8, 0.15) 100%); color: #fde047; border: 1px solid rgba(245, 158, 11, 0.7); box-shadow: 0 0 12px rgba(245, 158, 11, 0.25); padding: 5px 12px; border-radius: 20px; font-weight: 800; font-size: 0.82rem;'>{res}</span>"
            elif "LOCK" in res or "PROFIT" in res:
                res_pill = f"<span style='background: rgba(56, 189, 248, 0.25); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.6); padding: 5px 12px; border-radius: 20px; font-weight: 800; font-size: 0.82rem;'>{res}</span>"
            else:
                res_pill = f"<span style='background: rgba(148, 163, 184, 0.2); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.4); padding: 5px 12px; border-radius: 20px; font-weight: 600; font-size: 0.82rem;'>{res}</span>"

            tsl = str(r.get("T / SL", ""))
            if "|" in tsl:
                parts = tsl.split("|")
                tsl_html = f"<span style='background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.12); padding: 4px 8px; border-radius: 6px; white-space: nowrap;'><span style='color:#4ade80; font-weight:700;'>🎯 {parts[0].strip()}</span> &nbsp;|&nbsp; <span style='color:#f87171; font-weight:700;'>🛑 {parts[1].strip()}</span></span>"
            else:
                tsl_html = f"<span style='color:#94a3b8;'>{tsl}</span>"

            html += f"""
<tr style='border-bottom: 1px solid rgba(255, 255, 255, 0.07);'>
<td style='padding: 14px 16px; white-space: nowrap;'><span style='background: rgba(148, 163, 184, 0.15); color: #f1f5f9; border: 1px solid rgba(148, 163, 184, 0.3); padding: 4px 10px; border-radius: 16px; font-weight: 700; font-size: 0.80rem;'>📅 {r.get('Date')}</span></td>
<td style='padding: 14px 16px; white-space: nowrap;'><span style='background: rgba(56, 189, 248, 0.18); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.45); padding: 4px 10px; border-radius: 16px; font-weight: 800; font-size: 0.82rem;'>🎫 {r.get('Contract')}</span></td>
<td style='padding: 14px 16px; white-space: nowrap;'><span style='background: rgba(255, 255, 255, 0.08); color: #f8fafc; border: 1px solid rgba(255, 255, 255, 0.18); padding: 4px 10px; border-radius: 16px; font-weight: 700; font-size: 0.80rem;'>🏷️ {r.get('Stock')}</span></td>
<td style='padding: 14px 16px; white-space: nowrap;'><span style='background: linear-gradient(135deg, rgba(168, 85, 247, 0.25) 0%, rgba(139, 92, 246, 0.18) 100%); color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.5); padding: 4px 10px; border-radius: 16px; font-weight: 800; font-size: 0.82rem;'>🔮 {r.get('AI Probability')}</span></td>
<td style='padding: 14px 16px; color: #22d3ee; font-weight: 600; white-space: nowrap;'>⏱️ {r.get('Entry')}</td>
<td style='padding: 14px 16px; white-space: nowrap;'>{pred_pill}</td>
<td style='padding: 14px 16px;'>{tsl_html}</td>
<td style='padding: 14px 16px; color: #cbd5e1; font-size: 0.84rem; line-height: 1.45; min-width: 260px;'>{r.get('What Happened')}</td>
<td style='padding: 14px 16px; color: #fbbf24; font-weight: 700; white-space: nowrap;'>🏁 {r.get('Exit')}</td>
<td style='padding: 14px 16px; white-space: nowrap;'><span style='background: rgba(99, 102, 241, 0.18); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.4); padding: 4px 10px; border-radius: 16px; font-weight: 600; font-size: 0.80rem;'>⏰ {r.get('Cutoff')}</span></td>
<td style='padding: 14px 16px; white-space: nowrap;'>{res_pill}</td>
</tr>
"""
        html += """
</tbody>
</table>
</div>
"""
        if hasattr(st, "html"):
            st.html(html)
        else:
            st.markdown(html, unsafe_allow_html=True)
    else:
        st.info("No sub-60% discretionary trades logged yet. Use the form below to log a new session.")

    # Interactive Trade Form
    with st.expander("➕ Log / Update Sub-60% Discretionary Trade", expanded=False):
        import datetime as dt
        today_default = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5, minutes=30)).strftime("%d %b %Y")

        mode = st.radio("Action Mode", ["➕ Log New Session", "✏️ Edit Existing Record"], horizontal=True, key="sub60_mode")

        selected_idx = None
        current_data = {}
        if mode == "✏️ Edit Existing Record" and trades:
            trade_options = [f"#{i+1}: {t.get('Date', '')} | {t.get('Contract', '')} | {t.get('Result', '')}" for i, t in enumerate(trades)]
            selected_trade_label = st.selectbox("Select Trade to Edit", trade_options, key="sub60_edit_select")
            selected_idx = trade_options.index(selected_trade_label)
            current_data = trades[selected_idx]

        with st.form("manual_sub60_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                f_date = st.text_input("Date", value=current_data.get("Date", today_default))
                f_stock = st.selectbox("Stock", ["CDSL.NS", "HDFCBANK.NS"], index=0 if current_data.get("Stock", "CDSL.NS") == "CDSL.NS" else 1)
            with c2:
                f_contract = st.text_input("Contract", value=current_data.get("Contract", "CDSL 1260 PE"))
                f_prob = st.text_input("AI Probability", value=current_data.get("AI Probability", "50.9% DOWN"))
            with c3:
                pred_options = ["BUY PUT (PE)", "BUY CALL (CE)", "NO TRADE / SIDELINED"]
                default_pred_idx = pred_options.index(current_data.get("AI Prediction", "BUY PUT (PE)")) if current_data.get("AI Prediction") in pred_options else 0
                f_pred = st.selectbox("AI Prediction", pred_options, index=default_pred_idx)
                f_entry = st.text_input("Entry", value=current_data.get("Entry", "09:20 AM @ ₹39.00 (Spot ₹1,250)"))

            c4, c5 = st.columns(2)
            with c4:
                f_tsl = st.text_input("T / SL", value=current_data.get("T / SL", "T: ₹43.00 (+₹4) | SL: ₹33.00 (-₹6)"))
                f_exit = st.text_input("Exit", value=current_data.get("Exit", "11:58 AM @ ₹43.00"))
            with c5:
                f_cutoff = st.text_input("Cutoff", value=current_data.get("Cutoff", "Not Reached (Hit Target)"))
                f_result = st.text_input("Result", value=current_data.get("Result", "✅ WIN (+₹1,900 / +10.3%)"))

            f_happened = st.text_area("What Happened", value=current_data.get("What Happened", "Spot dropped from ₹1,250 towards ₹1,245. Premium rallied +₹4.00 to hit ₹43.00 target at 11:58 AM (+₹1,900 on 1 Lot / +₹3,800 on 2 Lots)."))


            submit = st.form_submit_button("💾 Save Trade Record", use_container_width=True)
            if submit:
                new_entry = {
                    "Date": f_date.strip(),
                    "Contract": f_contract.strip(),
                    "Stock": f_stock.strip(),
                    "AI Probability": f_prob.strip(),
                    "Entry": f_entry.strip(),
                    "AI Prediction": f_pred.strip(),
                    "T / SL": f_tsl.strip(),
                    "What Happened": f_happened.strip(),
                    "Exit": f_exit.strip(),
                    "Cutoff": f_cutoff.strip(),
                    "Result": f_result.strip()
                }
                add_or_update_manual_trade(new_entry, orig_index=selected_idx if mode == "✏️ Edit Existing Record" else None)
                st.success(f"✅ Trade record for {f_date} saved successfully!")
                st.rerun()

    # Delete Record Expander
    if trades:
        with st.expander("🗑️ Delete a Logged Trade", expanded=False):
            del_options = [f"#{i+1}: {t.get('Date', '')} | {t.get('Contract', '')} | {t.get('Result', '')}" for i, t in enumerate(trades)]
            del_selection = st.selectbox("Select Record to Delete", del_options, key="sub60_delete_select")
            del_idx = del_options.index(del_selection)
            if st.button("❌ Confirm Delete", type="primary", key="sub60_confirm_delete"):
                delete_manual_trade_by_index(del_idx)
                st.warning(f"Record #{del_idx+1} deleted.")
                st.rerun()

def render_dashboard_page():
    apply_custom_theme()

    st.markdown("<div class='glowing-header'>📊 Live Market Dashboard & Watchlist</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>24x7 Real-Time NSE/BSE Stock Monitoring Engine — CDSL, HDFCBANK & Custom Watchlist</div>", unsafe_allow_html=True)

    watchlist = st.session_state.get("watchlist", ["CDSL.NS", "HDFCBANK.NS"])

    # Render 5s Auto-Refreshing Fragment
    render_live_stock_cards_fragment(watchlist)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Demat Additions Weekly Run-Rate & CDSL Growth Engine
    render_demat_analytics_widget()

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Sub-60% Probability Discretionary Trade Tracker
    render_sub60_manual_tracker()

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. Fundamentals Comparison Table
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


if __name__ == "__main__":
    render_dashboard_page()

