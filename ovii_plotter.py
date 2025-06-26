import requests
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from datetime import datetime

# Use this function from earlier
def get_nse_option_data(symbol="NIFTY"):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        session = requests.Session()
        session.headers.update(headers)
        response = session.get(url, timeout=5)
        data = response.json()
        return data
    except Exception as e:
        print("Error fetching data:", e)
        return None

def compute_ovii(data):
    ce_vol = 0
    pe_vol = 0
    atm_strike = data['records']['strikePrices'][len(data['records']['strikePrices']) // 2]

    for record in data['records']['data']:
        if record.get("strikePrice") == atm_strike:
            ce_vol = record['CE']['totalTradedVolume'] if 'CE' in record else 0
            pe_vol = record['PE']['totalTradedVolume'] if 'PE' in record else 0
            break

    if ce_vol + pe_vol == 0:
        return 0

    ovii = (ce_vol - pe_vol) / (ce_vol + pe_vol)
    return ovii

# Real-time plot
def live_plot(interval=30):
    x_vals = []
    y_vals = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))

    while True:
        data = get_nse_option_data()
        if data:
            ovii = compute_ovii(data)
            timestamp = datetime.now().strftime("%H:%M:%S")
            x_vals.append(timestamp)
            y_vals.append(ovii)

            ax.clear()
            ax.plot(x_vals, y_vals, marker='o', label="OVII")
            ax.axhline(0.6, color='green', linestyle='--', label="Bullish Trigger")
            ax.axhline(-0.6, color='red', linestyle='--', label="Bearish Trigger")
            ax.set_xlabel("Time")
            ax.set_ylabel("OVII")
            ax.set_title("Option Volume Imbalance Index (Live)")
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

            print(f"[{timestamp}] OVII: {ovii:.4f}")

            # Optional alert
            if ovii > 0.6:
                print("🚀 BUY SIGNAL")
            elif ovii < -0.6:
                print("🔻 SELL SIGNAL")

            plt.pause(1)
        time.sleep(interval)

if __name__ == "__main__":
    live_plot()
