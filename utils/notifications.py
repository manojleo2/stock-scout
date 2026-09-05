import urllib.request
import urllib.parse
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
            bot_token = st.secrets["TELEGRAM_BOT_TOKEN"]
        if hasattr(st, "secrets") and "TELEGRAM_CHAT_ID" in st.secrets:
            chat_id = st.secrets["TELEGRAM_CHAT_ID"]
    except Exception as e:
        logging.warning(f"st.secrets not configured for Telegram: {e}")

    return bot_token, chat_id

def send_telegram_alert(message: str) -> bool:
    """
    Sends an instant push notification message to the user's phone via Telegram Bot.
    Uses standard library urllib (no external pip dependencies needed).
    """
    bot_token, chat_id = get_telegram_credentials()

    if not bot_token or not chat_id:
        logging.info("Telegram notification skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not configured in st.secrets.")
        return False

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
                return True
    except Exception as e:
        logging.error(f"Error sending Telegram alert: {e}")

    return False

def send_test_notification() -> bool:
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

def send_gap_alert_notification(symbol: str, name: str, signal: str, price: float, vwap: float, target: float) -> bool:
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
