import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import streamlit as st
import logging
import urllib.parse
from config import CACHE_TTL_SECONDS

logging.basicConfig(level=logging.INFO)

# Download VADER lexicon if not already available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Custom Financial Lexicon Adjustments
FINANCIAL_LEXICON = {
    'bullish': 2.0,
    'bearish': -2.0,
    'outperform': 1.5,
    'underperform': -1.5,
    'surge': 1.8,
    'plunge': -1.8,
    'rally': 1.5,
    'slump': -1.5,
    'profit': 1.2,
    'loss': -1.2,
    'growth': 1.0,
    'decline': -1.0,
    'breakout': 1.6,
    'breakdown': -1.6,
    'upgrade': 1.5,
    'downgrade': -1.5,
    'demat': 0.8,
    'allotment': 0.8
}

def get_vader_analyzer():
    sid = SentimentIntensityAnalyzer()
    sid.lexicon.update(FINANCIAL_LEXICON)
    return sid

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_stock_news(stock_name: str, max_items: int = 15) -> list:
    """
    Fetch latest news headlines for a stock via Google News RSS feed.
    """
    query = f"{stock_name} stock share price India"
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    try:
        feed = feedparser.parse(rss_url)
        sid = get_vader_analyzer()
        
        news_items = []
        for entry in feed.entries[:max_items]:
            title = entry.title
            link = entry.link
            published = getattr(entry, 'published', 'Recent')
            source = entry.source.title if hasattr(entry, 'source') else 'Google News'
            
            # VADER Sentiment Analysis
            scores = sid.polarity_scores(title)
            compound = scores['compound']
            
            if compound >= 0.05:
                sentiment = "Bullish"
                badge = "🟢"
            elif compound <= -0.05:
                sentiment = "Bearish"
                badge = "🔴"
            else:
                sentiment = "Neutral"
                badge = "⚪"
                
            news_items.append({
                "title": title,
                "link": link,
                "published": published,
                "source": source,
                "compound_score": round(compound, 3),
                "sentiment": sentiment,
                "badge": badge
            })
            
        return news_items
    except Exception as e:
        logging.error(f"Error fetching news for {stock_name}: {e}")
        return []

def aggregate_news_sentiment(news_items: list) -> dict:
    """
    Aggregate individual news sentiments into an overall score.
    """
    if not news_items:
        return {
            "overall_sentiment": "Neutral",
            "badge": "⚪",
            "avg_score": 0.0,
            "bullish_pct": 0,
            "bearish_pct": 0,
            "neutral_pct": 100,
            "total_articles": 0
        }
        
    total = len(news_items)
    bullish = sum(1 for x in news_items if x['sentiment'] == "Bullish")
    bearish = sum(1 for x in news_items if x['sentiment'] == "Bearish")
    neutral = sum(1 for x in news_items if x['sentiment'] == "Neutral")
    
    avg_score = sum(x['compound_score'] for x in news_items) / total
    
    if avg_score >= 0.05:
        overall = "Bullish"
        badge = "🟢"
    elif avg_score <= -0.05:
        overall = "Bearish"
        badge = "🔴"
    else:
        overall = "Neutral"
        badge = "⚪"
        
    return {
        "overall_sentiment": overall,
        "badge": badge,
        "avg_score": round(avg_score, 3),
        "bullish_pct": round((bullish / total) * 100),
        "bearish_pct": round((bearish / total) * 100),
        "neutral_pct": round((neutral / total) * 100),
        "total_articles": total
    }

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_stock_news_sentiment_score(stock_name: str) -> dict:
    """
    Fetch latest news headlines for stock and return a normalized sentiment score (-1.0 to +1.0).
    """
    items = fetch_stock_news(stock_name, max_items=10)
    summary = aggregate_news_sentiment(items)
    
    if not items or summary.get("total_articles", 0) == 0:
        return {"score": 0.0, "sentiment": "Neutral", "badge": "⚪ Neutral", "count": 0}
    
    total_compound = sum(item.get("compound_score", 0.0) for item in items)
    avg_score = total_compound / len(items)
    
    return {
        "score": round(avg_score, 3),
        "sentiment": summary.get("overall_sentiment", "Neutral"),
        "badge": summary.get("badge", "⚪ Neutral"),
        "count": len(items)
    }
