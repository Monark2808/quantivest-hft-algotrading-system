import requests
import time
from datetime import datetime
import matplotlib.pyplot as plt

def get_orderbook(symbol="NIFTY"):
    url = f"https://www.nseindia.com/api/quote-derivative?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        session = requests.Session()
        session.headers.update(headers)
        res = session.get(url, timeout=5)
        data = res.json()
        return data
    except Exception as e:
        print("❌ Error fetching orderbook:", e)
        return None

def compute_basp(order_data):
    try:
        bid_volume = order_data['marketDeptOrderBook']['buy'][0]['quantity']
        ask_volume = order_data['marketDeptOrderBook']['sell'][0]['quantity']

        total = ask_volume + bid_volume
        if total == 0:
            return 0

        basp = (ask_volume - bid_volume) / total
        return basp
    except Exception as e:
        print("❌ Error computing BASP:", e)
        return None

def live_basp_plot(interval=15):
    x_vals = []
    y_vals = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))

    while True:
        data = get_orderbook()
        if data:
            basp = compute_basp(data)
            time_now = datetime.now().strftime("%H:%M:%S")

            x_vals.append(time_now)
            y_vals.append(basp)

            ax.clear()
            ax.plot(x_vals, y_vals, marker='o', label="BASP")
            ax.axhline(0.5, color='red', linestyle='--', label="Sellers Pressure")
            ax.axhline(-0.5, color='green', linestyle='--', label="Buyers Pressure")
            ax.set_xlabel("Time")
            ax.set_ylabel("BASP")
            ax.set_title("Bid-Ask Spread Pressure (Live)")
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

            print(f"[{time_now}] BASP: {basp:.4f}")
            if basp > 0.5:
                print("🔻 SELLERS DOMINATING")
            elif basp < -0.5:
                print("🚀 BUYERS DOMINATING")

            plt.pause(1)
        time.sleep(interval)

if __name__ == "__main__":
    live_basp_plot()
