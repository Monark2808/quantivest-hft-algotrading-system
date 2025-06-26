import requests
import numpy as np
import pandas as pd
import time

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

def run_tracker(interval=30):
    print("Tracking OVII (Option Volume Imbalance Index)...")
    while True:
        option_data = get_nse_option_data()
        if option_data:
            ovii = compute_ovii(option_data)
            print(f"OVII: {ovii:.4f}")
        time.sleep(interval)

if __name__ == "__main__":
    run_tracker()
