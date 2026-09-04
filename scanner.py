print("🔥 SCANNER.PY IS ACTUALLY RUNNING", flush=True)

import os
import requests

print("✅ Python imports working", flush=True)

token = os.environ.get("BOT_TOKEN")
chat_id = os.environ.get("CHAT_ID")

print("BOT_TOKEN present:", bool(token), flush=True)
print("CHAT_ID present:", bool(chat_id), flush=True)

if not token or not chat_id:
    raise Exception("Telegram secrets are missing")

url = f"https://api.telegram.org/bot{token}/sendMessage"

response = requests.post(
    url,
    data={
        "chat_id": chat_id,
        "text": "🟢 TEST SUCCESS — GitHub Actions can reach Telegram."
    },
    timeout=20
)

print("Telegram HTTP status:", response.status_code, flush=True)
print("Telegram response:", response.text, flush=True)

response.raise_for_status()

print("🎉 TELEGRAM TEST PASSED", flush=True)
