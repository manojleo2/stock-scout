import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.news_sentiment import fetch_stock_news, aggregate_news_sentiment
from utils.ui_theme import apply_custom_theme
from config import STOCK_NAME_MAP

def render_news_sentiment_page():
    apply_custom_theme()

    st.markdown("<div class='glowing-header'>📰 Financial News & Sentiment Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>Live Headlines & VADER NLP Sentiment Analyzer Tuned for Indian Equity Markets</div>", unsafe_allow_html=True)

    watchlist = st.session_state.get("watchlist", ["CDSL.NS", "NSDL.BO"])

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        selected_symbol = st.selectbox(
            "Select Stock for News",
            options=watchlist,
            format_func=lambda s: f"{STOCK_NAME_MAP.get(s, s)} ({s})"
        )
    with col_s2:
        max_news = st.slider("Max Headlines", min_value=5, max_value=25, value=12)

    if not selected_symbol:
        st.warning("Please select a stock.")
        return

    stock_display_name = STOCK_NAME_MAP.get(selected_symbol, selected_symbol.replace(".NS", "").replace(".BO", ""))

    with st.spinner(f"Fetching financial headlines for {stock_display_name}..."):
        news_items = fetch_stock_news(stock_display_name, max_items=max_news)
        summary = aggregate_news_sentiment(news_items)

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Composite Sentiment Card
    st.markdown(f"### 📊 Sentiment Overview: {stock_display_name}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Overall Sentiment", f"{summary['badge']} {summary['overall_sentiment']}")
    with c2:
        st.metric("Bullish Headlines", f"{summary['bullish_pct']}%")
    with c3:
        st.metric("Bearish Headlines", f"{summary['bearish_pct']}%")
    with c4:
        st.metric("Neutral Headlines", f"{summary['neutral_pct']}%")

    # Sentiment Distribution Donut Chart
    if news_items:
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Bullish', 'Bearish', 'Neutral'],
            values=[summary['bullish_pct'], summary['bearish_pct'], summary['neutral_pct']],
            hole=.55,
            marker_colors=['#00E676', '#FF5252', '#9E9E9E']
        )])
        fig_donut.update_layout(
            height=250, template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20),
            title_text="Sentiment Distribution Breakdown"
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Live News Headlines Feed
    st.markdown("### 📰 Live Headlines Feed")

    if not news_items:
        st.info(f"No recent news headlines retrieved for {stock_display_name}.")
        return

    for item in news_items:
        badge_color = "#00E676" if item['sentiment'] == "Bullish" else ("#FF5252" if item['sentiment'] == "Bearish" else "#9E9E9E")
        
        with st.container(border=True):
            st.markdown(f"""
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;'>
                    <a href='{item['link']}' target='_blank' style='font-size: 1.05rem; font-weight: 700; color: #38bdf8; text-decoration: none;'>
                        {item['badge']} {item['title']}
                    </a>
                    <span style='background: rgba(255,255,255,0.05); color: {badge_color}; border: 1px solid {badge_color}; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;'>
                        {item['sentiment']} ({item['compound_score']})
                    </span>
                </div>
                <div style='color: #94a3b8; font-size: 0.8rem;'>
                    Source: <b>{item['source']}</b> | Published: {item['published']}
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__" or True:
    render_news_sentiment_page()
