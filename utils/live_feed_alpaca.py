import websocket
import threading
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
SYMBOL = "AAPL"
SOCKET = "wss://stream.data.alpaca.markets/v2/iex"  # use /sip if you have a paid plan

latest_trade = {"price": None}
ws = None

def on_open(wsapp):
    print("🔓 WebSocket connection opened.")
    
    auth_data = {
        "action": "auth",
        "key": API_KEY,
        "secret": SECRET_KEY
    }
    wsapp.send(json.dumps(auth_data))
    print("🔑 Auth data sent.")

    subscribe_msg = {
        "action": "subscribe",
        "trades": [SYMBOL]
    }
    wsapp.send(json.dumps(subscribe_msg))
    print(f"📡 Subscribed to trades for {SYMBOL}.")

def on_message(wsapp, message):
    data = json.loads(message)
    if isinstance(data, list) and len(data) > 0:
        event = data[0]
        if event.get("T") == "t":  # 't' = trade
            price = event.get("p")
            if price:
                latest_trade["price"] = price
                print(f"📈 Live Trade Received: {SYMBOL} @ ${price}")
        elif event.get("T") == "error":
            print("⚠️ Error message received:", event)
    else:
        print("📭 Message ignored:", data)

def on_error(wsapp, error):
    print("❌ WebSocket error:", error)

def on_close(wsapp, code, msg):
    print("🛑 WebSocket closed:", code, msg)

def start_ws():
    def run_ws():
        global ws
        ws = websocket.WebSocketApp(
            SOCKET,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        while True:
            try:
                ws.run_forever()
                print("🔁 Attempting reconnection...")
                time.sleep(5)
            except Exception as e:
                print("⚠️ WebSocket exception:", e)
                time.sleep(5)

    thread = threading.Thread(target=run_ws, daemon=True)
    thread.start()

def get_latest_price():
    return latest_trade.get("price", None)
