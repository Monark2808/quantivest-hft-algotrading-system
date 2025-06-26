import asyncio
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import websockets

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET")
BASE_URL = "wss://stream.data.alpaca.markets/v2/iex"  # IEX = free tier
SYMBOL = "AAPL"
PRICE_FILE = "utils/latest_price.json"  # shared file for Streamlit

async def stream_to_file():
    async with websockets.connect(BASE_URL) as ws:
        # Step 1: Authenticate
        await ws.send(json.dumps({
            "action": "auth",
            "key": API_KEY,
            "secret": API_SECRET
        }))
        print("🔐", await ws.recv())

        # Step 2: Subscribe to trades
        await ws.send(json.dumps({
            "action": "subscribe",
            "trades": [SYMBOL]
        }))
        print(f"✅ Subscribed to {SYMBOL}")

        # Step 3: Listen to live ticks
        while True:
            message = await ws.recv()
            data = json.loads(message)

            for item in data:
                if item.get("T") == "t" and item.get("S") == SYMBOL:
                    price = item["p"]
                    timestamp = item["t"]
                    latest = {
                        "symbol": SYMBOL,
                        "price": price,
                        "timestamp": timestamp,
                        "updated": datetime.now().isoformat()
                    }
                    with open(PRICE_FILE, "w") as f:
                        json.dump(latest, f, indent=2)
                    print(f"💾 Wrote live price: {price}")

if __name__ == "__main__":
    asyncio.run(stream_to_file())
