import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.news_sentiment import fetch_stock_news, aggregate_news_sentiment
from config import STOCK_NAME_MAP

def render_news_sentiment_page():
    st.title("📰 Financial News & Sentiment Analysis")
    st.caption("Live headlines and sentiment scoring tuned for Indian stock market terminology")

    watchlist = st.session_state.get("watchlist", ["CDSL.NS", "NSDL.NS"])

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        selected_symbol = st.selectbox(
            "Select Stock for News & Sentiment",
            options=watchlist,
            format_func=lambda s: f"{STOCK_NAME_MAP.get(s, s)} ({s})"
        )
    with col_s2:
        max_news = st.slider("Max Headlines", min_value=5, max_value=25, value=12)

    if not selected_symbol:
        st.warning("Please select a stock.")
        return

    stock_display_name = STOCK_NAME_MAP.get(selected_symbol, selected_symbol.replace(".NS", ""))

    with st.spinner(f"Fetching latest financial headlines for {stock_display_name}..."):
        news_items = fetch_stock_news(stock_display_name, max_items=max_news)
        summary = aggregate_news_sentiment(news_items)

    st.markdown("---")

    # 1. Composite Sentiment Card
    st.subheader(f"📊 Sentiment Overview: {stock_display_name}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Overall Sentiment", f"{summary['badge']} {summary['overall_sentiment']}")
    with c2:
        st.metric("Bullish Headlines", f"{summary['bullish_pct']}%")
    with c3:
        st.metric("Bearish Headlines", f"{summary['bearish_pct']}%")
    with c4:
        st.metric("Neutral Headlines", f"{summary['neutral_pct']}%")

    # Sentiment Breakdown Bar
    if news_items:
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Bullish', 'Bearish', 'Neutral'],
            values=[summary['bullish_pct'], summary['bearish_pct'], summary['neutral_pct']],
            hole=.5,
            marker_colors=['#26a69a', '#ef5350', '#9e9e9e']
        )])
        fig_donut.update_layout(
            height=250, template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            title_text="Headlines Sentiment Distribution"
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")

    # 2. Live News Headlines Feed
    st.subheader("📰 Recent Headlines Feed")

    if not news_items:
        st.info(f"No recent news headlines retrieved for {stock_display_name}.")
        return

    for item in news_items:
        with st.container(border=True):
            cols = st.columns([1, 8, 3])
            with cols[0]:
                st.markdown(f"### {item['badge']}")
            with cols[1]:
                st.markdown(f"**[{item['title']}]({item['link']})**")
                st.caption(f"Source: {item['source']} | Published: {item['published']}")
            with cols[2]:
                st.markdown(f"**{item['sentiment']}**")
                st.caption(f"VADER Score: `{item['compound_score']}`")

if __name__ == "__main__" or True:
    render_news_sentiment_page()

