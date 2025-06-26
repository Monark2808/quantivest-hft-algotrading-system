# utils/streamer.py
import os
import asyncio
from alpaca.data.live import StockDataStream
from dotenv import load_dotenv

load_dotenv()

ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")

stream = StockDataStream(ALPACA_KEY, ALPACA_SECRET)

tick_data = []

@stream.on_stock_trade("AAPL")
async def on_trade(trade):
    tick_data.append({
        "timestamp": trade.timestamp.isoformat(),
        "price": trade.price,
        "volume": trade.size
    })
    # Optional: Print or log trade for debugging
    print(f"Tick: {trade.price} @ {trade.timestamp}")

def run_stream():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(stream.subscribe_trades("AAPL"))
    loop.run_forever()

