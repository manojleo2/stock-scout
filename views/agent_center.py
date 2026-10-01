import streamlit as st
import pandas as pd
import datetime as dt
import plotly.graph_objects as go

from utils.ui_theme import apply_custom_theme
from utils.data_loader import load_saved_watchlist
from config import STOCK_NAME_MAP
from agent.model_agent import StockScoutModelAgent

def render_agent_center_page():
    apply_custom_theme()

    # Initialize Agent
    if "model_agent" not in st.session_state:
        st.session_state["model_agent"] = StockScoutModelAgent()
    agent: StockScoutModelAgent = st.session_state["model_agent"]

    st.markdown("""
        <div style='margin-bottom: 20px;'>
            <h1 style='font-size: 2.2rem; font-weight: 800; margin:0; background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                🤖 AI Model Monitoring & Training Agent
            </h1>
            <p style='color: #94a3b8; font-size: 0.95rem; margin-top: 6px;'>
                Autonomous ML Watchdog • Performance Drift Detection • One-Click & Scheduled Retraining Engine
            </p>
        </div>
    """, unsafe_allow_html=True)

    watchlist = load_saved_watchlist()
    state = agent.get_state()
    monitored_models = state.get("monitored_models", {})

    # Top Configuration & Control Strip
    with st.expander("⚙️ Agent Drift Detection & Sensitivity Settings", expanded=False):
        c_set1, c_set2 = st.columns(2)
        with c_set1:
            threshold_slider = st.slider(
                "Retraining Trigger Threshold (Recent 10-Trade Win Rate %)",
                min_value=40.0,
                max_value=75.0,
                value=55.0,
                step=1.0,
                help="If a model's rolling 10-trade audit accuracy slips below this threshold, the agent flags it for retraining."
            )
        with c_set2:
            min_samples_slider = st.slider(
                "Minimum Completed Audit Samples Before Drift Alert",
                min_value=3,
                max_value=15,
                value=5,
                step=1,
                help="Prevents false-alarm retraining on new symbols with insufficient trade records."
            )

    # Action Controls Header
    col_act1, col_act2, col_act3 = st.columns([1.2, 1.2, 1.2])
    
    with col_act1:
        if st.button("🔍 Run Health & Drift Audit", use_container_width=True, type="secondary"):
            with st.spinner("Agent auditing real-world prediction accuracy across watchlist..."):
                res = agent.monitor_models(threshold_pct=threshold_slider, min_eval_samples=min_samples_slider)
                if res.success:
                    st.toast(f"✅ {res.message}", icon="🛡️")
                else:
                    st.error(res.message)
                st.rerun()

    with col_act2:
        if st.button("⚡ Auto-Retrain Drifted Models", use_container_width=True, type="primary"):
            with st.spinner("Agent inspecting drift and executing automatic retraining..."):
                res = agent.auto_evaluate_and_retrain(threshold_pct=threshold_slider)
                if res.success:
                    st.toast(f"🚀 {res.message}", icon="🔄")
                else:
                    st.error(res.message)
                st.rerun()

    with col_act3:
        if st.button("🔄 Force Retrain All Models", use_container_width=True):
            with st.spinner("Executing full ensemble retraining across all watchlist stocks..."):
                res = agent.train_models()
                if res.success:
                    st.toast(f"✅ {res.message}", icon="🎯")
                else:
                    st.error(res.message)
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI Top Bar
    total_monitored = len(watchlist)
    healthy_count = sum(1 for m in monitored_models.values() if "Healthy" in m.get("status_badge", ""))
    drift_count = sum(1 for m in monitored_models.values() if "Retrain" in m.get("status_badge", "") or "Drift" in m.get("status_badge", ""))
    last_run_display = state.get("last_run", "Not yet run")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
            <div style='background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 15px;'>
                <div style='color: #94a3b8; font-size: 0.8rem; font-weight: 600;'>WATCHLIST MODELS</div>
                <div style='font-size: 1.8rem; font-weight: 800; color: #38bdf8;'>{total_monitored}</div>
                <div style='color: #64748b; font-size: 0.75rem;'>Core Daily Ensembles</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
            <div style='background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(0, 230, 118, 0.3); border-radius: 12px; padding: 15px;'>
                <div style='color: #94a3b8; font-size: 0.8rem; font-weight: 600;'>HEALTHY MODELS</div>
                <div style='font-size: 1.8rem; font-weight: 800; color: #00E676;'>{healthy_count}</div>
                <div style='color: #64748b; font-size: 0.75rem;'>Win Rate ≥ {threshold_slider}%</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi3:
        drift_color = "#FF5252" if drift_count > 0 else "#94a3b8"
        st.markdown(f"""
            <div style='background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 82, 82, 0.3); border-radius: 12px; padding: 15px;'>
                <div style='color: #94a3b8; font-size: 0.8rem; font-weight: 600;'>RETRAIN NEEDED</div>
                <div style='font-size: 1.8rem; font-weight: 800; color: {drift_color};'>{drift_count}</div>
                <div style='color: #64748b; font-size: 0.75rem;'>Accuracy Drift Flagged</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
            <div style='background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(192, 132, 252, 0.3); border-radius: 12px; padding: 15px;'>
                <div style='color: #94a3b8; font-size: 0.8rem; font-weight: 600;'>LAST AGENT RUN</div>
                <div style='font-size: 1.2rem; font-weight: 700; color: #c084fc; margin-top: 6px;'>{last_run_display}</div>
                <div style='color: #64748b; font-size: 0.75rem;'>Autonomous Engine Active</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Tabs
    tab_overview, tab_history, tab_extensibility = st.tabs([
        "📊 Model Health & Retraining Grid",
        "📜 Agent History & Audit Log",
        "🧩 Agent Extensibility Hub"
    ])

    with tab_overview:
        st.markdown("### 🎯 Watchlist Models Monitoring Grid")
        st.caption("Real-world audit win rate tracks actual trading performance post-session close.")

        # Ensure we have fresh monitoring data if empty
        if not monitored_models:
            with st.spinner("Initializing first-time agent audit..."):
                agent.monitor_models(threshold_pct=threshold_slider)
                state = agent.get_state()
                monitored_models = state.get("monitored_models", {})

        for sym in watchlist:
            name = STOCK_NAME_MAP.get(sym, sym)
            m_info = monitored_models.get(sym, {})
            status_badge = m_info.get("status_badge", "🟡 Unchecked")
            overall_acc = m_info.get("overall_accuracy_pct")
            roll10_acc = m_info.get("rolling_10_acc_pct")
            completed_samples = m_info.get("completed_samples", 0)
            test_acc = m_info.get("test_accuracy_pct")
            last_trained = m_info.get("last_trained", "Not trained this session")
            last_bias = m_info.get("last_direction", "N/A")
            last_conf = m_info.get("last_confidence", "N/A")

            # Border color based on health
            border_color = "rgba(56, 189, 248, 0.3)"
            if "Healthy" in status_badge:
                border_color = "rgba(0, 230, 118, 0.4)"
            elif "Retrain" in status_badge or "Drift" in status_badge:
                border_color = "rgba(255, 82, 82, 0.5)"
            elif "Dip" in status_badge:
                border_color = "rgba(255, 179, 0, 0.5)"

            with st.container():
                st.markdown(f"""
                    <div style='background: rgba(15, 23, 42, 0.75); border: 1px solid {border_color}; border-radius: 12px; padding: 18px; margin-bottom: 15px;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <span style='font-size: 1.3rem; font-weight: 700; color: #f8fafc;'>{name}</span>
                                <span style='color: #94a3b8; font-size: 0.9rem; margin-left: 8px;'>({sym})</span>
                            </div>
                            <span style='background: rgba(30, 41, 59, 0.9); padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;'>
                                {status_badge}
                            </span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns([1.1, 1.1, 1.1, 1.4, 1.1])
                
                with col_m1:
                    acc_display = f"{overall_acc}%" if overall_acc is not None else "N/A"
                    st.metric("Overall Audit Accuracy", acc_display, help="Historical win rate across all completed sessions in audit file.")
                
                with col_m2:
                    roll_display = f"{roll10_acc}%" if roll10_acc is not None else "N/A"
                    delta_acc = None
                    if roll10_acc is not None and overall_acc is not None:
                        delta_acc = f"{round(roll10_acc - overall_acc, 1)}% vs Avg"
                    st.metric("Rolling 10-Trade Win Rate", roll_display, delta=delta_acc, help="Win rate in the most recent 10 audited market sessions.")

                with col_m3:
                    test_display = f"{test_acc}%" if test_acc is not None else "N/A"
                    st.metric("Test Set Validation", test_display, help="Out-of-sample accuracy calculated during training split.")

                with col_m4:
                    st.markdown(f"**Directional Bias:** `{last_bias}`")
                    st.caption(f"Conviction: {last_conf} • Trained: {last_trained}")

                with col_m5:
                    if st.button(f"🎯 Retrain {sym}", key=f"retrain_btn_{sym}", use_container_width=True):
                        with st.spinner(f"Retraining {sym} ensemble model..."):
                            train_res = agent.train_models(symbols=[sym])
                            if train_res.success:
                                st.toast(f"✅ Retrained {sym}!", icon="🎯")
                            else:
                                st.error(f"Retraining failed: {train_res.message}")
                            st.rerun()

                # Expandable details
                with st.expander(f"🔎 Detailed Diagnostics & Feature Drivers for {sym}", expanded=False):
                    diag_c1, diag_c2 = st.columns(2)
                    with diag_c1:
                        st.markdown("#### 🎯 Audited Session Metrics")
                        st.write(f"- **Completed Trade Audits:** `{completed_samples}`")
                        st.write(f"- **Last Audited Date:** `{m_info.get('last_prediction_date', 'N/A')}`")
                        st.write(f"- **Last Agent Check:** `{m_info.get('last_monitored', 'N/A')}`")
                    
                    with diag_c2:
                        st.markdown("#### ⚡ Top Predictive Features")
                        top_f = m_info.get("top_features", [])
                        if top_f:
                            for fname, fval in top_f:
                                st.write(f"- **{fname}**: `{round(fval * 100, 1)}%` importance")
                        else:
                            st.caption("Retrain model to inspect top predictive feature weights.")

                st.markdown("---")

    with tab_history:
        st.markdown("### 📜 Autonomous Agent Activity Log")
        st.caption("Every check, drift detection event, and retraining cycle is persistently recorded.")

        history = agent.get_history(limit=50)
        if not history:
            st.info("No activity recorded yet. Run a health audit or training cycle above!")
        else:
            table_rows = []
            for item in history:
                details = item.get("details", {})
                table_rows.append({
                    "Timestamp": item.get("timestamp"),
                    "Task / Event": item.get("event_type", "").upper(),
                    "Status": "✅ Success" if details.get("success") else "❌ Error",
                    "Message": details.get("message", ""),
                    "Recommendations": "; ".join(details.get("recommendations", [])) or "None"
                })
            df_hist = pd.DataFrame(table_rows)
            st.dataframe(df_hist, use_container_width=True, hide_index=True)

    with tab_extensibility:
        st.markdown("### 🧩 Enhancing the Agent with New Works")
        st.markdown("""
            This agent is built on a clean **Pluggable Task Architecture**. You can easily enhance it with new works,
            such as:
            - **Custom Intraday Breakout Audits**
            - **Hyperparameter Grid Sweeps**
            - **Automated Telegram / Discord Notifications**
            - **Custom Feature Engineering Pipelines**
            - **Execution & Sizing Logic**
        """)

        st.markdown("#### 📝 How to Register a Custom Task (Code Recipe):")
        st.code("""
from agent.base import BaseTask, TaskResult
from agent.model_agent import StockScoutModelAgent

class CustomRegimeTask(BaseTask):
    def __init__(self):
        super().__init__(
            name="volatility_regime_checker",
            description="Checks if India VIX or market regime requires switching model hyperparameters."
        )

    def execute(self, context: dict) -> TaskResult:
        symbols = context.get("symbols", [])
        # 1. Your custom logic here (e.g. check VIX or macro data)
        # 2. Return standard TaskResult
        return TaskResult(
            task_name=self.name,
            success=True,
            message="Checked market regime: Normal volatility regime.",
            data={"regime": "LOW_VOLATILITY"},
            recommendations=["Proceed with standard swing position sizing."]
        )

# Register with agent:
agent = StockScoutModelAgent()
agent.register_task("check_regime", CustomRegimeTask())

# Execute on-demand or in automation:
result = agent.run_task("check_regime", {"symbols": ["CDSL.NS"]})
print(result.message)
        """, language="python")

        st.markdown("#### ⚙️ Background / Headless Automation CLI:")
        st.markdown("""
            You can also run this agent in a headless terminal, crontab, or Windows Task Scheduler:
            ```bash
            # 1. Monitor watchlist and detect accuracy drift:
            python -m agent.runner --monitor

            # 2. Automatically audit & retrain if drift is detected:
            python -m agent.runner --auto --threshold 55.0

            # 3. Retrain specific stocks:
            python -m agent.runner --train --symbols CDSL.NS,HDFCBANK.NS

            # 4. Check status in JSON:
            python -m agent.runner --status --json
            ```
        """)

if __name__ == "__main__" or True:
    render_agent_center_page()
