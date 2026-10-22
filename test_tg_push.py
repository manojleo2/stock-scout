import os
import requests
import json

env_path = r"C:\Users\manoj\scout\stock_scout\.env"
env = {}
with open(env_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")

token = env.get("TELEGRAM_BOT_TOKEN", "").strip()
configured_chat_id = env.get("TELEGRAM_CHAT_ID", "").strip()

print(f"Token present: {bool(token)}, Configured Chat ID: {configured_chat_id}")

# 1. Fetch recent updates to find any new chats
r_updates = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=5)
updates_data = r_updates.json()
print("Recent updates count:", len(updates_data.get("result", [])))

chat_ids_to_try = []
if configured_chat_id:
    chat_ids_to_try.append(configured_chat_id)

for upd in updates_data.get("result", []):
    msg = upd.get("message") or upd.get("my_chat_member") or upd.get("channel_post") or {}
    chat = msg.get("chat", {})
    cid = str(chat.get("id"))
    if cid and cid not in chat_ids_to_try:
        chat_ids_to_try.append(cid)
        print(f"Discovered chat from update: ID={cid}, Title/Name={chat.get('title') or chat.get('first_name')}")

print(f"Attempting test send to: {chat_ids_to_try}")

success_ids = []
for cid in chat_ids_to_try:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": cid,
        "text": (
            "🎯 <b>Nifty & Sensex Sentinel Connected!</b>\n\n"
            "⚡ <b>5-Second Options Pipe:</b> ACTIVE\n"
            "📈 <b>Instruments:</b> Nifty 50 & BSE Sensex\n"
            "🛡️ <b>Dynamic Rules:</b> Target +30% | Stop Loss −15% | Trail to Cost @ +18%\n\n"
            "<i>You will receive instant alerts here whenever a high-conviction trade triggers.</i>"
        ),
        "parse_mode": "HTML"
    }
    res = requests.post(url, json=payload, timeout=5)
    print(f"Target {cid} -> Status: {res.status_code}, Response: {res.text}")
    if res.status_code == 200:
        success_ids.append(cid)

print(f"Final Result: Successfully delivered to {success_ids}")
