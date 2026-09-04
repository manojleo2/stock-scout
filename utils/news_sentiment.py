import feedparser
import urllib.parse
import logging
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure vader_lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    try:
        nltk.download('vader_lexicon', quiet=True)
    except Exception as e:
        logging.warning(f"Could not download NLTK vader_lexicon: {e}")

def get_financial_vader_analyzer():
    """
    Initialize VADER analyzer updated with Indian financial market vocabulary.
    """
    sia = SentimentIntensityAnalyzer()
    financial_lexicon = {
        'bullish': 2.0,
        'bearish': -2.0,
        'rally': 1.8,
        'slump': -1.8,
        'surge': 1.8,
        'plunge': -2.0,
        'multibagger': 2.5,
        'outperform': 1.5,
        'underperform': -1.5,
        'upper circuit': 2.2,
        'lower circuit': -2.2,
        'dividend': 1.0,
        'bonus issue': 1.2,
        'headwinds': -1.2,
        'tailwinds': 1.2,
        'strong q3': 1.5,
        'weak q3': -1.5,
        'profit rises': 1.8,
        'profit falls': -1.8,
        'record high': 2.0,
        'all time high': 2.0,
        'downgrade': -1.8,
        'upgrade': 1.8
    }
    sia.lexicon.update(financial_lexicon)
    return sia

def fetch_stock_news(stock_name: str, max_items: int = 12) -> list:
    """
    Fetch financial news headlines from Google News RSS for Indian stock keywords.
    """
    try:
        query_str = f"{stock_name} share NSE OR {stock_name} stock price"
        encoded_query = urllib.parse.quote(query_str)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

        feed = feedparser.parse(rss_url)
        sia = get_financial_vader_analyzer()

        news_results = []
        for entry in feed.entries[:max_items]:
            title = entry.title
            
            # Remove trailing source tag e.g. " - Moneycontrol"
            clean_title = title.rsplit(" - ", 1)[0] if " - " in title else title
            source = title.rsplit(" - ", 1)[1] if " - " in title else "Financial Media"

            scores = sia.polarity_scores(clean_title)
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

            news_results.append({
                "title": clean_title,
                "source": source,
                "link": entry.link,
                "published": entry.get("published", "Recent"),
                "sentiment": sentiment,
                "badge": badge,
                "compound_score": round(compound, 3)
            })

        return news_results
    except Exception as e:
        logging.error(f"Error fetching news for {stock_name}: {e}")
        return []

def aggregate_news_sentiment(news_list: list) -> dict:
    """
    Summarize composite sentiment from a list of news headline items.
    """
    if not news_list:
        return {
            "overall_sentiment": "Neutral",
            "badge": "⚪",
            "avg_score": 0.0,
            "bullish_pct": 0,
            "bearish_pct": 0,
            "neutral_pct": 100
        }

    scores = [n['compound_score'] for n in news_list]
    avg_score = sum(scores) / len(scores)

    bullish_cnt = sum(1 for n in news_list if n['sentiment'] == 'Bullish')
    bearish_cnt = sum(1 for n in news_list if n['sentiment'] == 'Bearish')
    neutral_cnt = sum(1 for n in news_list if n['sentiment'] == 'Neutral')

    total = len(news_list)

    if avg_score >= 0.1:
        overall = "Bullish / Positive"
        badge = "🟢"
    elif avg_score <= -0.1:
        overall = "Bearish / Negative"
        badge = "🔴"
    else:
        overall = "Neutral / Mixed"
        badge = "🟡"

    return {
        "overall_sentiment": overall,
        "badge": badge,
        "avg_score": round(avg_score, 3),
        "bullish_pct": round((bullish_cnt / total) * 100),
        "bearish_pct": round((bearish_cnt / total) * 100),
        "neutral_pct": round((neutral_cnt / total) * 100),
        "total_news_items": total
    }
