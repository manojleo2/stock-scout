import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass

WEEKS_PER_MONTH = 4.3333
TRADING_DAYS_PER_MONTH = 21.5

# Standard Industry Benchmarks (SEBI & Depository Disclosures)
DEFAULT_CDSL_SHARE = 0.77  # ~77% cumulative account market share
DEFAULT_NSDL_SHARE = 0.23  # ~23% cumulative account market share

@dataclass(frozen=True)
class DematRunRateResult:
    monthly_additions_m: float        # e.g. 3.20 Million
    monthly_additions_lakh: float     # e.g. 32.0 Lakhs
    weekly_total_k: float             # e.g. 738.5 k/week
    weekly_cdsl_k: float              # e.g. 568.6 k/week
    weekly_nsdl_k: float              # e.g. 169.8 k/week
    daily_total_k: float              # e.g. 148.8 k/day
    cdsl_share_pct: float
    nsdl_share_pct: float
    tier_name: str                    # "High Impact", "Moderate Impact", "Low Impact"
    tier_badge: str                   # "🟢 High Impact (Bullish Trigger)", etc.
    stock_impact_cdsl: str            # CDSL thesis
    stock_impact_nsdl: str            # NSDL thesis
    cvl_kyc_impact: str               # CDSL Ventures KYC impact

def compute_demat_run_rates(
    monthly_additions_m: float = 3.20,
    cdsl_share: float = DEFAULT_CDSL_SHARE,
    nsdl_share: float = DEFAULT_NSDL_SHARE
) -> DematRunRateResult:
    """
    Convert monthly demat additions (in Millions) into weekly run-rates
    for both CDSL and NSDL with threshold impact scores.
    """
    total_weekly_k = (monthly_additions_m * 1000.0) / WEEKS_PER_MONTH
    cdsl_weekly_k = total_weekly_k * cdsl_share
    nsdl_weekly_k = total_weekly_k * nsdl_share

    total_daily_k = (monthly_additions_m * 1000.0) / TRADING_DAYS_PER_MONTH

    if monthly_additions_m > 3.5:
        tier_name = "High Impact"
        tier_badge = "🟢 High Impact (Bullish Trigger)"
        stock_impact_cdsl = (
            "Strong fundamental catalyst for CDSL. Rapid account expansion fuels recurring annual folio charges, "
            "boosts CVL KYC onboarding fees, and elevates trading velocity."
        )
        stock_impact_nsdl = (
            "Robust institutional and banking-broker account inflows, expanding custody holdings."
        )
        cvl_kyc_impact = "Surging (>2.0M+ fresh KYC hits/month). Significant margin uplift for CVL."
    elif 2.5 <= monthly_additions_m <= 3.5:
        tier_name = "Moderate Impact"
        tier_badge = "🟡 Moderate Impact (Steady Growth)"
        stock_impact_cdsl = (
            "Healthy baseline compounding. Matches India's normalized post-FY24 run-rate. "
            "Maintains steady sequential top-line growth and solid return ratios for CDSL."
        )
        stock_impact_nsdl = (
            "Stable banking-broker channel additions, supported by corporate actions and debt IPOs."
        )
        cvl_kyc_impact = "Healthy baseline (1.4M - 2.0M KYC inquiries/month)."
    else:
        tier_name = "Low Impact"
        tier_badge = "🔴 Low Impact (Cooling / Saturation)"
        stock_impact_cdsl = (
            "Retail participation fatigue. Short-term headwinds for fresh KYC revenue. "
            "Earnings depend primarily on existing portfolio delivery turnover rather than new customer influx."
        )
        stock_impact_nsdl = (
            "Subdued retail influx; NSDL performance insulated by institutional AUC custodian fee stickiness."
        )
        cvl_kyc_impact = "Muted (<1.4M KYC hits/month)."

    return DematRunRateResult(
        monthly_additions_m=monthly_additions_m,
        monthly_additions_lakh=monthly_additions_m * 10.0,
        weekly_total_k=round(total_weekly_k, 1),
        weekly_cdsl_k=round(cdsl_weekly_k, 1),
        weekly_nsdl_k=round(nsdl_weekly_k, 1),
        daily_total_k=round(total_daily_k, 1),
        cdsl_share_pct=round(cdsl_share * 100, 1),
        nsdl_share_pct=round(nsdl_share * 100, 1),
        tier_name=tier_name,
        tier_badge=tier_badge,
        stock_impact_cdsl=stock_impact_cdsl,
        stock_impact_nsdl=stock_impact_nsdl,
        cvl_kyc_impact=cvl_kyc_impact
    )

def render_demat_analytics_widget():
    """
    Renders an interactive Demat Additions weekly tracking section for CDSL and NSDL.
    """
    st.markdown("### 📈 Demat Account Additions: Weekly Run-Rate & Impact Engine")
    st.caption("Weekly run-rate breakdown and stock impact analysis specifically tailored for CDSL.NS & NSDL.BO")

    # Current simulated baseline (3.2M monthly = ~738k/week)
    result = compute_demat_run_rates(monthly_additions_m=3.20)

    # 1. Weekly Run-Rate Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Weekly Industry Additions", 
            f"{result.weekly_total_k:,.0f}k / week", 
            f"{result.monthly_additions_lakh:.1f} Lakhs/mo"
        )
    with c2:
        st.metric(
            "CDSL Weekly Run-Rate (77%)", 
            f"{result.weekly_cdsl_k:,.0f}k / week", 
            f"~{result.weekly_cdsl_k/5:.0f}k / trading day"
        )
    with c3:
        st.metric(
            "NSDL Weekly Run-Rate (23%)", 
            f"{result.weekly_nsdl_k:,.0f}k / week", 
            f"~{result.weekly_nsdl_k/5:.0f}k / trading day"
        )
    with c4:
        st.metric(
            "Weekly Impact Rating", 
            result['tier_name'] if isinstance(result, dict) else result.tier_name,
            result['tier_badge'] if isinstance(result, dict) else result.tier_badge
        )

    # 2. Interactive Weekly Simulation Slider
    with st.expander("🎛️ Adjust Weekly Run-Rate Simulation", expanded=False):
        sim_monthly = st.slider(
            "Monthly Additions Horizon (Millions)",
            min_value=1.0, max_value=5.0, value=3.2, step=0.1,
            help="Recent historical run-rate range: 2.0M (cooling) to 4.6M (bull peak)"
        )
        sim_cdsl_pct = st.slider(
            "CDSL Share of Additions (%)",
            min_value=60, max_value=90, value=77, step=1,
            help="CDSL accounts for ~77% cumulative share and ~82-86% incremental monthly share"
        ) / 100.0

        sim_result = compute_demat_run_rates(monthly_additions_m=sim_monthly, cdsl_share=sim_cdsl_pct, nsdl_share=1.0 - sim_cdsl_pct)

    # 3. Weekly Summary Analysis Cards for CDSL and NSDL
    st.markdown("#### 📝 Weekly Impact Summary for CDSL & NSDL")
    
    col_cdsl, col_nsdl = st.columns(2)
    with col_cdsl:
        with st.container(border=True):
            st.markdown(f"""
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <h4 style='margin:0; color:#00E676;'>CDSL (`CDSL.NS`) Impact</h4>
                    <span class='badge-up'>Weekly Run-Rate: {sim_result.weekly_cdsl_k:,.0f}k</span>
                </div>
                <p style='color:#cbd5e1; font-size:0.9rem; margin-top:10px;'>
                    {sim_result.stock_impact_cdsl}
                </p>
                <div style='color:#94a3b8; font-size:0.8rem;'>
                    💡 <b>CVL KYC Impact:</b> {sim_result.cvl_kyc_impact}
                </div>
            """, unsafe_allow_html=True)

    with col_nsdl:
        with st.container(border=True):
            st.markdown(f"""
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <h4 style='margin:0; color:#818cf8;'>NSDL (`NSDL.BO`) Impact</h4>
                    <span class='badge-neutral'>Weekly Run-Rate: {sim_result.weekly_nsdl_k:,.0f}k</span>
                </div>
                <p style='color:#cbd5e1; font-size:0.9rem; margin-top:10px;'>
                    {sim_result.stock_impact_nsdl}
                </p>
                <div style='color:#94a3b8; font-size:0.8rem;'>
                    💡 <b>AUC Stability:</b> NSDL holds >70% market share in institutional Assets Under Custody.
                </div>
            """, unsafe_allow_html=True)
