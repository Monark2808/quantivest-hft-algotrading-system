from alpaca_trade_api.rest import REST
from dotenv import load_dotenv
import os
import csv
from datetime import datetime
from utils.live_feed_alpaca import get_latest_price

# Load environment variables
load_dotenv()

# Alpaca credentials
API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

# Initialize API
api = REST(API_KEY, SECRET_KEY, BASE_URL)

# Supported diversified assets
TRADABLE_ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

def cancel_open_orders():
    """Cancel all open orders to avoid wash trade errors."""
    for order in api.list_orders(status="open"):
        api.cancel_order(order.id)

def log_trade(symbol, qty, side, price):
    """Append trade data to a CSV file."""
    log_file = "trades.csv"
    fieldnames = ["timestamp", "symbol", "side", "qty", "price"]

    with open(log_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price
        })

def place_order(symbol, quantity, side):
    """Submit a market order with safe-guarding and trade logging."""
    try:
        if symbol not in TRADABLE_ASSETS:
            return f"❌ {symbol} is not supported for trading."

        cancel_open_orders()

        # Get live price for logging
        price = get_latest_price() or 0.0

        # Submit order
        api.submit_order(
            symbol=symbol,
            qty=quantity,
            side=side,
            type="market",
            time_in_force="gtc"
        )

        log_trade(symbol, quantity, side, price)
        return f"✅ Order placed: {side.upper()} {quantity} share(s) of {symbol} @ ${price:.2f}"
    except Exception as e:
        return f"❌ Error placing order: {str(e)}"
