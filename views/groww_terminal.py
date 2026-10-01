"""
views/groww_terminal.py
───────────────────────
Real-Time Groww Trading Terminal Dashboard.
Direct, zero-delay pipe into user's live Groww Demat & Trading Account.

Features:
- Live Account & UCC Profile
- Available Margin & Option Buying Cash
- Real-Time F&O & Equity Positions with Realized P&L
- Real Demat Holdings with Live Value & Return %
- Live Orders & Trades Audit
- CDSL Monday Rulebook Order Sizer (Auto-calibrated to Groww Cash)
"""

import streamlit as st
import pandas as pd
import datetime as dt
import logging
import yfinance as yf

from utils.options_feed import get_groww_client, get_live_option_quote
from utils.ui_theme import apply_custom_theme

logging.basicConfig(level=logging.INFO)

# Apply UI Theme
apply_custom_theme()


def fetch_groww_data():
    """Fetch complete live dataset from Groww API with error isolation."""
    client = get_groww_client()
    if not client:
        return {"status": "unauthenticated", "error": "Groww API client not initialized. Check .env token."}

    data = {"status": "connected", "timestamp": dt.datetime.now().strftime("%I:%M:%S %p")}
    
    # 1. Profile
    try:
        data["profile"] = client.get_user_profile()
    except Exception as e:
        data["profile"] = {}
        data["profile_err"] = str(e)

    # 2. Margins
    try:
        data["margin"] = client.get_available_margin_details()
    except Exception as e:
        data["margin"] = {}
        data["margin_err"] = str(e)

    # 3. Positions
    try:
        data["positions"] = client.get_positions_for_user()
    except Exception as e:
        data["positions"] = {"positions": []}
        data["positions_err"] = str(e)

    # 4. Holdings
    try:
        data["holdings"] = client.get_holdings_for_user()
    except Exception as e:
        data["holdings"] = {"holdings": []}
        data["holdings_err"] = str(e)

    # 5. Orders
    try:
        data["orders"] = client.get_order_list()
    except Exception as e:
        data["orders"] = {"order_list": []}
        data["orders_err"] = str(e)

    return data


# ── Page Header ───────────────────────────────────────────────────────────────
c_head, c_btn = st.columns([4, 1])
with c_head:
    st.markdown("""
        <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
            <h1 style='margin:0; font-size: 2.2rem; background: linear-gradient(90deg, #00D09C, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                🟢 Groww Live Terminal
            </h1>
            <span style='background: rgba(0, 208, 156, 0.15); color: #00D09C; border: 1px solid #00D09C; padding: 4px 12px; border-radius: 16px; font-weight: 600; font-size: 0.85rem;'>
                ⚡ ZERO-DELAY DIRECT PIPE
            </span>
        </div>
        <p style='color: #94a3b8; margin: 0 0 16px 0; font-size: 0.9rem;'>
            Live streaming connection to Groww Demat & Trading Account • Real Cash, Positions, Orders & Holdings
        </p>
    """, unsafe_allow_html=True)

with c_btn:
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Sync Live Now", use_container_width=True, type="primary"):
        st.rerun()

# ── Fetch Live Data ───────────────────────────────────────────────────────────
groww_res = fetch_groww_data()

if groww_res.get("status") == "unauthenticated":
    st.error(f"❌ {groww_res.get('error')}")
    st.info("💡 Paste your fresh Groww Auth Token into `.env` under `GROWW_AUTH_TOKEN`.")
    st.stop()

profile   = groww_res.get("profile", {})
margin    = groww_res.get("margin", {})
positions = groww_res.get("positions", {}).get("positions", [])
holdings  = groww_res.get("holdings", {}).get("holdings", [])
orders    = groww_res.get("orders", {}).get("order_list", [])

# Margin Metrics
clear_cash       = float(margin.get("clear_cash") or 0.0)
net_margin_used  = float(margin.get("net_margin_used") or 0.0)
fno_details      = margin.get("fno_margin_details", {}) or {}
opt_buy_power    = float(fno_details.get("option_buy_balance_available") or clear_cash)

# Calculate Total Realized P&L from positions
total_realized_pnl = sum(float(p.get("realised_pnl") or 0.0) for p in positions)
pnl_sign = "+" if total_realized_pnl >= 0 else ""
pnl_color = "#00E676" if total_realized_pnl >= 0 else "#FF5252"

# ── Top Level Account Metric Bar ──────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("💰 Clear Available Cash", f"₹{clear_cash:,.2f}", "Unencumbered Balance")
with m2:
    st.metric("⚡ Option Buying Power", f"₹{opt_buy_power:,.2f}", "F&O Margin")
with m3:
    st.metric("🛡️ Margin Used Today", f"₹{net_margin_used:,.2f}", "0% Utilized" if net_margin_used == 0 else "Active Margin")
with m4:
    st.metric("📊 Realized Session P&L", f"{pnl_sign}₹{abs(total_realized_pnl):,.2f}", f"{len(positions)} Tracked Positions",
              delta_color="normal" if total_realized_pnl >= 0 else "inverse")
with m5:
    ucc = profile.get("ucc", "—")
    st.metric("👤 Groww Account UCC", f"{ucc}", f"Sync: {groww_res['timestamp']}")

st.markdown("---")

# ── Navigation Sub-Tabs ───────────────────────────────────────────────────────
tab_pos, tab_hold, tab_ord, tab_sizer, tab_diag = st.tabs([
    f"⚡ Live Positions ({len(positions)})",
    f"💼 Demat Holdings ({len(holdings)})",
    f"📋 Orders & Trades ({len(orders)})",
    "🎯 CDSL Live Sizer (Monday)",
    "🏦 Margin & Account Diagnostic"
])

# ── TAB 1: Live Positions ─────────────────────────────────────────────────────
with tab_pos:
    st.markdown("### ⚡ Live Intraday & F&O Positions (From Groww)")
    if not positions:
        st.info("No active open positions on Groww right now.")
    else:
        pos_rows = []
        for p in positions:
            sym         = p.get("trading_symbol", "—")
            seg         = p.get("segment", "—")
            prod        = p.get("product", "—")
            qty         = int(p.get("quantity") or 0)
            c_qty       = int(p.get("credit_quantity") or 0)
            d_qty       = int(p.get("debit_quantity") or 0)
            buy_avg     = float(p.get("debit_price") or 0.0)
            sell_avg    = float(p.get("credit_price") or 0.0)
            real_pnl    = float(p.get("realised_pnl") or 0.0)
            
            status = "CLOSED" if qty == 0 else ("LONG" if qty > 0 else "SHORT")
            
            pos_rows.append({
                "Contract / Symbol": sym,
                "Segment": seg,
                "Product": prod,
                "Net Qty": qty,
                "Buy Avg (₹)": f"₹{buy_avg:.2f}" if buy_avg > 0 else "—",
                "Sell Avg (₹)": f"₹{sell_avg:.2f}" if sell_avg > 0 else "—",
                "Total Traded Qty": max(c_qty, d_qty),
                "Realized P&L (₹)": real_pnl,
                "Position Status": status
            })

        df_pos = pd.DataFrame(pos_rows)
        
        # Color formatted display
        st.dataframe(
            df_pos.style.format({
                "Realized P&L (₹)": lambda x: f"{'+' if x >= 0 else ''}₹{x:,.2f}"
            }).applymap(
                lambda v: "color: #00E676; font-weight: bold;" if isinstance(v, (int, float)) and v > 0 
                else ("color: #FF5252; font-weight: bold;" if isinstance(v, (int, float)) and v < 0 else ""),
                subset=["Realized P&L (₹)"]
            ),
            use_container_width=True,
            hide_index=True
        )

from concurrent.futures import ThreadPoolExecutor

@st.cache_data(ttl=60, show_spinner=False)
def fetch_holdings_cmp_batch(symbols: tuple) -> dict:
    """Fetch live CMPs in parallel using ThreadPoolExecutor for lightning-fast zero-delay rendering."""
    results = {}
    def _fetch_one(s):
        ticker_sym = f"{s}.NS" if not s.endswith(".BO") and not s.endswith(".NS") else s
        try:
            t = yf.Ticker(ticker_sym)
            p = float(t.fast_info.last_price or 0.0)
            return s, p
        except Exception:
            return s, 0.0

    with ThreadPoolExecutor(max_workers=min(len(symbols), 8) or 1) as executor:
        for s, p in executor.map(_fetch_one, symbols):
            if p > 0:
                results[s] = p
    return results

# ── TAB 2: Demat Holdings ─────────────────────────────────────────────────────
with tab_hold:
    st.markdown("### 💼 Real Demat Holdings (From Groww Depository)")
    if not holdings:
        st.info("No long-term holdings found in this account.")
    else:
        # Fetch live prices for holdings to calculate live P&L in parallel
        sym_list = tuple(h.get("trading_symbol", "") for h in holdings if h.get("trading_symbol"))
        live_prices = fetch_holdings_cmp_batch(sym_list)

        hold_rows = []
        total_invested_holdings = 0.0
        total_current_holdings = 0.0

        for h in holdings:
            sym_raw = h.get("trading_symbol", "")
            qty = float(h.get("quantity") or 0.0)
            avg_p = float(h.get("average_price") or 0.0)
            inv_val = qty * avg_p
            total_invested_holdings += inv_val

            live_p = live_prices.get(sym_raw, avg_p)
            cur_val = qty * live_p
            total_current_holdings += cur_val
            pnl = cur_val - inv_val
            pnl_pct = (pnl / inv_val * 100.0) if inv_val > 0 else 0.0

            hold_rows.append({
                "Stock": sym_raw,
                "ISIN": h.get("isin", "—"),
                "Quantity": int(qty),
                "Average Price (₹)": avg_p,
                "Live CMP (₹)": live_p,
                "Invested (₹)": inv_val,
                "Current Value (₹)": cur_val,
                "Unrealized P&L (₹)": pnl,
                "Return (%)": pnl_pct
            })

        hk1, hk2, hk3 = st.columns(3)
        hold_pnl = total_current_holdings - total_invested_holdings
        hold_pnl_pct = (hold_pnl / total_invested_holdings * 100.0) if total_invested_holdings > 0 else 0.0
        with hk1:
            st.metric("Total Demat Invested", f"₹{total_invested_holdings:,.2f}")
        with hk2:
            st.metric("Total Demat Portfolio Value", f"₹{total_current_holdings:,.2f}")
        with hk3:
            st.metric("Overall Unrealized Return", f"{'+' if hold_pnl>=0 else ''}₹{hold_pnl:,.2f}",
                      f"{'+' if hold_pnl>=0 else ''}{hold_pnl_pct:.2f}%",
                      delta_color="normal" if hold_pnl>=0 else "inverse")

        df_hold = pd.DataFrame(hold_rows)
        st.dataframe(
            df_hold.style.format({
                "Average Price (₹)": "₹{:,.2f}",
                "Live CMP (₹)": "₹{:,.2f}",
                "Invested (₹)": "₹{:,.2f}",
                "Current Value (₹)": "₹{:,.2f}",
                "Unrealized P&L (₹)": lambda x: f"{'+' if x >= 0 else ''}₹{x:,.2f}",
                "Return (%)": lambda x: f"{'+' if x >= 0 else ''}{x:.2f}%"
            }).applymap(
                lambda v: "color: #00E676; font-weight: bold;" if isinstance(v, (int, float)) and v > 0 
                else ("color: #FF5252; font-weight: bold;" if isinstance(v, (int, float)) and v < 0 else ""),
                subset=["Unrealized P&L (₹)", "Return (%)"]
            ),
            use_container_width=True,
            hide_index=True
        )

# ── TAB 3: Orders & Trades ────────────────────────────────────────────────────
with tab_ord:
    st.markdown("### 📋 Today's Orders & Trades (From Groww)")
    if not orders:
        st.info("No orders placed yet in today's session.")
    else:
        st.dataframe(pd.DataFrame(orders), use_container_width=True, hide_index=True)

# ── TAB 4: CDSL Live Sizer (Monday Rulebook) ──────────────────────────────────
with tab_sizer:
    st.markdown("### 🎯 Live CDSL Position Sizer (Synchronized to Real Groww Balance)")
    
    # Capital safety check against live Groww cash
    lots_allowed = 2 if clear_cash > 30000 else (1 if clear_cash >= 20000 else 0)
    
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Live Groww Cash Balance", f"₹{clear_cash:,.2f}", "Direct Broker Balance")
    with sc2:
        lot_badge = "🟢 2 LOTS PERMITTED" if lots_allowed == 2 else ("🟡 1 LOT ONLY" if lots_allowed == 1 else "🔴 STOP TRADING")
        st.metric("Rulebook Allowed Lots", f"{lots_allowed} Lot ({lots_allowed * 700} Qty)", lot_badge)
    with sc3:
        # CDSL live quote
        try:
            cdsl_spot = float(yf.Ticker("CDSL.NS").fast_info.last_price or 1335.0)
        except Exception:
            cdsl_spot = 1335.0
        atm_strike = round(cdsl_spot / 20) * 20
        st.metric("CDSL ATM Strike", f"₹{atm_strike:.0f}", f"Spot: ₹{cdsl_spot:,.2f}")

    st.markdown("---")
    st.markdown("#### 🛡️ Monday Winning Rulebook Parameters for CDSL")
    
    c_pe, c_ce = st.columns(2)
    with c_pe:
        st.markdown(f"""
            <div style='background: rgba(255, 82, 82, 0.08); border: 1px solid rgba(255, 82, 82, 0.3); border-radius: 12px; padding: 16px;'>
                <h4 style='color: #FF5252; margin: 0 0 10px 0;'>📉 IF MODEL IS DOWN (PUT)</h4>
                <p><b>Target Contract:</b> CDSL ₹{atm_strike:.0f} PE</p>
                <p><b>VWAP Gatekeeper:</b> CDSL Spot MUST trade <b>BELOW</b> 09:15-09:20 VWAP</p>
                <p><b>Initial SL:</b> -₹5.00 Option Premium (-₹{5.0 * 700 * lots_allowed:,.0f} Max Risk)</p>
                <p><b>Milestone 1 (+₹7.00):</b> Trail SL to Cost (₹0 Risk)</p>
                <p><b>Milestone 2 (+₹8.50):</b> Trail SL to +₹5.00 (+₹{5.0 * 700 * lots_allowed - 100:,.0f} Locked)</p>
                <p><b>Target:</b> +₹10.00 Option Premium (+₹{10.0 * 700 * lots_allowed - 100:,.0f} Net Gain)</p>
            </div>
        """, unsafe_allow_html=True)

    with c_ce:
        st.markdown(f"""
            <div style='background: rgba(0, 230, 118, 0.08); border: 1px solid rgba(0, 230, 118, 0.3); border-radius: 12px; padding: 16px;'>
                <h4 style='color: #00E676; margin: 0 0 10px 0;'>📈 IF MODEL IS UP (CALL)</h4>
                <p><b>Target Contract:</b> CDSL ₹{atm_strike:.0f} CE</p>
                <p><b>VWAP Gatekeeper:</b> CDSL Spot MUST trade <b>ABOVE</b> 09:15-09:20 VWAP</p>
                <p><b>Initial SL:</b> -₹5.00 Option Premium (-₹{5.0 * 700 * lots_allowed:,.0f} Max Risk)</p>
                <p><b>Milestone 1 (+₹7.00):</b> Trail SL to Cost (₹0 Risk)</p>
                <p><b>Milestone 2 (+₹8.50):</b> Trail SL to +₹5.00 (+₹{5.0 * 700 * lots_allowed - 100:,.0f} Locked)</p>
                <p><b>Target:</b> +₹10.00 Option Premium (+₹{10.0 * 700 * lots_allowed - 100:,.0f} Net Gain)</p>
            </div>
        """, unsafe_allow_html=True)

# ── TAB 5: Margin & Account Diagnostics ───────────────────────────────────────
with tab_diag:
    st.markdown("### 🏦 Full Groww Margin & Authorization Breakdown")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("#### User & Exchange Clearance:")
        st.json({
            "User UCC": profile.get("ucc"),
            "NSE Enabled": profile.get("nse_enabled"),
            "BSE Enabled": profile.get("bse_enabled"),
            "DDPI Enabled": profile.get("ddpi_enabled"),
            "Active Segments": profile.get("active_segments"),
            "Groww Vendor": "Official Trade API"
        })
    with d2:
        st.markdown("#### Real Margin Data:")
        st.json(margin)
