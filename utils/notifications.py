import urllib.request
import urllib.parse
import urllib.error
import json
import logging
import streamlit as st

logging.basicConfig(level=logging.INFO)

def get_telegram_credentials() -> tuple:
    """
    Safely fetch Telegram Bot Token & Chat ID from encrypted st.secrets.
    """
    bot_token = None
    chat_id = None

    try:
        if hasattr(st, "secrets") and "TELEGRAM_BOT_TOKEN" in st.secrets:
            bot_token = str(st.secrets["TELEGRAM_BOT_TOKEN"]).strip()
        if hasattr(st, "secrets") and "TELEGRAM_CHAT_ID" in st.secrets:
            chat_id = str(st.secrets["TELEGRAM_CHAT_ID"]).strip()
    except Exception as e:
        logging.warning(f"st.secrets not configured for Telegram: {e}")

    return bot_token, chat_id

def send_telegram_alert(message: str) -> tuple:
    """
    Sends an instant push notification message to the user's phone via Telegram Bot.
    Returns (success: bool, error_details: str).
    """
    bot_token, chat_id = get_telegram_credentials()

    if not bot_token or not chat_id:
        return False, "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing in st.secrets."

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                logging.info("Telegram alert sent successfully.")
                return True, "Success"
    except urllib.error.HTTPError as e:
        err_body = e.read().decode('utf-8')
        logging.error(f"Telegram HTTP Error {e.code}: {err_body}")
        try:
            err_json = json.loads(err_body)
            desc = err_json.get("description", err_body)
            return False, f"Telegram API Error {e.code}: {desc}"
        except Exception:
            return False, f"Telegram API Error {e.code}: {err_body}"
    except Exception as e:
        logging.error(f"Error sending Telegram alert: {e}")
        return False, str(e)

    return False, "Unknown network error."

def send_test_notification() -> tuple:
    """
    Sends a test ping to verify Telegram Bot configuration.
    """
    msg = (
        "🚀 *Stock Scout Mobile Alerts Activated!*\n\n"
        "✅ Your Telegram Bot is successfully connected.\n"
        "📱 You will now receive instant phone alerts for:\n"
        "• Intraday 9:30 AM Gap Continuation Signals\n"
        "• Nifty 50 Volatility Movements (>1.0%)\n"
        "• Portfolio Target Price Hits & Stop Loss Warnings"
    )
    return send_telegram_alert(msg)

def send_gap_alert_notification(symbol: str, name: str, signal: str, price: float, vwap: float, target: float) -> tuple:
    """
    Sends an intraday gap continuation or profit booking alert.
    """
    badge = "🟢 UP" if "CONTINUATION" in signal else "🔴 DOWN"
    msg = (
        f"⚡ *Stock Scout Intraday Alert ({badge})*\n\n"
        f"📌 *Stock:* {name} (`{symbol}`)\n"
        f"📊 *Current Price (LTP):* ₹{price:,.2f}\n"
        f"🌊 *VWAP:* ₹{vwap:,.2f}\n"
        f"🎯 *Signal:* {signal}\n"
        f"📈 *Target Level:* ₹{target:,.2f}\n\n"
        f"👉 Check live charts: https://stock-scout-mn.streamlit.app"
    )
    return send_telegram_alert(msg)

def send_target_hit_alert(symbol: str, name: str, current_price: float, target_price: float) -> tuple:
    """
    Sends an instant phone alert when ANY portfolio stock reaches its Target Sell Price.
    """
    msg = (
        f"🎯 *TARGET PRICE HIT ALERT!* 🚀\n\n"
        f"📌 *Stock:* {name} (`{symbol}`)\n"
        f"📊 *Current Price (LTP):* ₹{current_price:,.2f}\n"
        f"🎯 *Target Sell Price:* ₹{target_price:,.2f}\n\n"
        f"✅ *Target Progress:* 100% Achieved!\n"
        f"👉 Check live portfolio: https://stock-scout-mn.streamlit.app"
    )
    return send_telegram_alert(msg)

def send_prediction_alert_notification(symbol: str, name: str, direction: str, prob_up: float, confidence: str, news_sentiment: float, next_date: str) -> tuple:
    """
    Sends a Pre-Market / Daily AI Forecast notification to Telegram BEFORE market open.
    """
    badge = "🟢 BULLISH UP" if direction == "UP" else "🔴 BEARISH DOWN"
    msg = (
        f"🔮 *Stock Scout Pre-Market AI Forecast*\n"
        f"📅 *Target Date:* {next_date}\n\n"
        f"📌 *Stock:* {name} (`{symbol}`)\n"
        f"🎯 *Predicted Signal:* {direction} ({badge})\n"
        f"📊 *Bullish Probability:* {prob_up:.1f}%\n"
        f"💪 *Confidence Level:* {confidence}\n"
        f"📰 *News Sentiment Score:* {news_sentiment:+.2f}\n\n"
        f"👉 View full analysis: https://stock-scout-mn.streamlit.app"
    )
    return send_telegram_alert(msg)

def send_stop_loss_alert(symbol: str, name: str, current_price: float, buy_price: float, drop_pct: float) -> tuple:
    """
    Sends a stop-loss warning when a portfolio stock drops significantly.
    """
    msg = (
        f"⚠️ *STOP-LOSS WARNING ALERT!* 📉\n\n"
        f"📌 *Stock:* {name} (`{symbol}`)\n"
        f"📊 *Current Price (LTP):* ₹{current_price:,.2f}\n"
        f"💸 *Buy Price:* ₹{buy_price:,.2f}\n"
        f"🔻 *Unrealized Loss:* {drop_pct:.2f}%\n\n"
        f"👉 Review risk management: https://stock-scout-mn.streamlit.app"
    )
    return send_telegram_alert(msg)

