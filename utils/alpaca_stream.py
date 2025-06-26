import asyncio
import json
import websockets
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET")
BASE_URL = "wss://stream.data.alpaca.markets/v2/iex"  # use /iex for free data

SYMBOL = "AAPL"

async def alpaca_stream():
    async with websockets.connect(BASE_URL) as ws:
        # Step 1: Authenticate
        auth_msg = {
            "action": "auth",
            "key": API_KEY,
            "secret": API_SECRET
        }
        await ws.send(json.dumps(auth_msg))
        print("🔐 Auth:", await ws.recv())

        # Step 2: Subscribe to trades
        await ws.send(json.dumps({
            "action": "subscribe",
            "trades": [SYMBOL]
        }))
        print(f"✅ Subscribed to {SYMBOL}...")

        # Step 3: Listen to ticks
        while True:
            msg = await ws.recv()
            print("📈 Tick:", msg)

if __name__ == "__main__":
    asyncio.run(alpaca_stream())
