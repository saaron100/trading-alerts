# alerts.py  (TEST script)
import os, requests, sys

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TOKEN or not CHAT_ID:
    print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variables")
    sys.exit(1)

message = "✅ GitHub Actions -> Telegram test message!"
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
r = requests.get(url, params={"chat_id": CHAT_ID, "text": message})
print("status:", r.status_code)
print("response:", r.text)
