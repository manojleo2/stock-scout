import streamlit as st

def apply_custom_theme():
    """
    Inject modern glassmorphism dark theme CSS styling into the Streamlit app.
    Forces dark root background and high-contrast metric values across all devices.
    """
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600;800&display=swap');

        /* Force Root App Dark Background */
        .stApp, div[data-testid="stAppViewContainer"], [data-testid="stHeader"], .main {
            background-color: #0b0f19 !important;
            background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0d111a 100%) !important;
            color: #f8fafc !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !alignment;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0f172a !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        }

        /* Force high contrast text in Sidebar */
        section[data-testid="stSidebar"] *, 
        section[data-testid="stSidebar"] p, 
        section[data-testid="stSidebar"] span, 
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #f8fafc !important;
        }

        section[data-testid="stSidebar"] .stCaption, 
        section[data-testid="stSidebar"] .stCaption p {
            color: #94a3b8 !important;
        }

        /* Metric Value Display - Bright High Contrast Cyan */
        div[data-testid="stMetricValue"], 
        div[data-testid="stMetricValue"] *, 
        [data-testid="stMetricValue"] > div {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 1.7rem !important;
            font-weight: 800 !important;
            color: #38bdf8 !important;
            text-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
        }

        /* Metric Label Color - Soft Slate */
        div[data-testid="stMetricLabel"], 
        div[data-testid="stMetricLabel"] *, 
        [data-testid="stMetricLabel"] > label {
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.8px !important;
            color: #cbd5e1 !important;
        }

        /* Container & Card Border Styling */
        div[data-testid="stVerticalBlock"] > div[data-testid="stBlock"] {
            border-radius: 12px;
        }

        /* Custom Cards */
        .glass-card {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
        }

        /* Status Badges */
        .badge-up {
            background: rgba(16, 185, 129, 0.2);
            color: #34d399 !important;
            border: 1px solid rgba(16, 185, 129, 0.4);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 700;
            display: inline-block;
        }

        .badge-down {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171 !important;
            border: 1px solid rgba(239, 68, 68, 0.4);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 700;
            display: inline-block;
        }

        /* Glowing Header Text */
        .glowing-header {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }

        .sub-glow {
            color: #cbd5e1;
            font-size: 0.95rem;
            margin-bottom: 25px;
        }

        /* Buttons Styling */
        .stButton>button {
            background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 10px !important;
        }

        /* Inputs & Selectboxes Contrast */
        .stSelectbox label, .stTextInput label, .stSlider label {
            color: #f8fafc !important;
            font-weight: 600 !important;
        }

        /* Hide Streamlit default header/footer branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)
