import streamlit as st

def apply_custom_theme():
    """
    Inject compact, high-contrast dark theme CSS styling into the Streamlit app.
    Fixes input box text visibility with dark input background (#1e293b) and bright white typed text (#f8fafc).
    """
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Force Root App Dark Background */
        .stApp, div[data-testid="stAppViewContainer"], [data-testid="stHeader"], .main {
            background-color: #0b0f19 !important;
            background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0d111a 100%) !important;
            color: #f8fafc !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        /* Sidebar Styling & High Contrast */
        section[data-testid="stSidebar"] {
            background-color: #0f172a !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        }

        section[data-testid="stSidebar"] p, 
        section[data-testid="stSidebar"] span, 
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #f8fafc !important;
        }

        section[data-testid="stSidebar"] .stCaption, 
        section[data-testid="stSidebar"] .stCaption p {
            color: #94a3b8 !important;
        }

        /* INPUT BOX TEXT VISIBILITY FIX */
        .stTextInput input, 
        div[data-baseweb="input"] input,
        div[data-baseweb="base-input"] input {
            background-color: #1e293b !important;
            color: #ffffff !important;
            border: 1px solid rgba(56, 189, 248, 0.4) !important;
            border-radius: 8px !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            padding: 8px 12px !important;
        }

        .stTextInput input::placeholder {
            color: #94a3b8 !important;
            opacity: 1 !important;
        }

        /* Selectboxes & Dropdowns */
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #1e293b !important;
            color: #ffffff !important;
            border: 1px solid rgba(56, 189, 248, 0.4) !important;
            border-radius: 8px !important;
        }

        ul[role="listbox"], div[data-baseweb="popover"], div[data-baseweb="menu"] {
            background-color: #0f172a !important;
            color: #ffffff !important;
        }

        div[data-baseweb="menu"] * {
            color: #ffffff !important;
        }

        /* Code Tag Styling */
        code, .stCode, pre {
            background-color: rgba(30, 41, 59, 0.8) !important;
            color: #38bdf8 !important;
            border: 1px solid rgba(56, 189, 248, 0.3) !important;
            border-radius: 4px !important;
            padding: 2px 6px !important;
        }

        /* Compact Metric Value Display - Bright High Contrast Cyan, No Truncation */
        div[data-testid="stMetricValue"], 
        div[data-testid="stMetricValue"] *, 
        [data-testid="stMetricValue"] > div {
            font-family: 'Inter', sans-serif !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            color: #38bdf8 !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            word-wrap: break-word !important;
            line-height: 1.3 !important;
        }

        /* Crisp Bright Metric Label Color */
        div[data-testid="stMetricLabel"], 
        div[data-testid="stMetricLabel"] *, 
        [data-testid="stMetricLabel"] > label,
        [data-testid="stMetricLabel"] p {
            font-size: 0.8rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.6px !important;
            color: #e2e8f0 !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }

        /* Cards & Container Styling */
        div[data-testid="stVerticalBlock"] > div[data-testid="stBlock"] {
            border-radius: 10px;
        }

        .glass-card {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
        }

        /* Status Badges */
        .badge-up {
            background: rgba(16, 185, 129, 0.2);
            color: #34d399 !important;
            border: 1px solid rgba(16, 185, 129, 0.4);
            padding: 3px 10px;
            border-radius: 16px;
            font-size: 0.8rem;
            font-weight: 700;
            display: inline-block;
        }

        .badge-down {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171 !important;
            border: 1px solid rgba(239, 68, 68, 0.4);
            padding: 3px 10px;
            border-radius: 16px;
            font-size: 0.8rem;
            font-weight: 700;
            display: inline-block;
        }

        /* Glowing Header Text */
        .glowing-header {
            font-size: 1.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        }

        .sub-glow {
            color: #cbd5e1;
            font-size: 0.9rem;
            margin-bottom: 20px;
        }

        /* Buttons Styling */
        .stButton>button {
            background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 8px !important;
            font-size: 0.85rem !important;
        }

        /* Inputs & Selectboxes Contrast */
        .stSelectbox label, .stTextInput label, .stSlider label {
            color: #f8fafc !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
        }

        /* Hide Streamlit default header/footer branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)
