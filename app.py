"""
app.py — Nifty & Sensex Intraday Options Predictor
Standalone Streamlit application entry point.
Run: streamlit run app.py
"""

import streamlit as st
from views.nifty_predictor_view import render_nifty_predictor_tab

st.set_page_config(
    page_title="Nifty/Sensex Intraday Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

render_nifty_predictor_tab()
