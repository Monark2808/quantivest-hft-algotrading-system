import requests
import time
import matplotlib.pyplot as plt
from datetime import datetime

def get_option_chain(symbol="NIFTY"):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        session = requests.Session()
        session.headers.update(headers)
        response = session.get(url, timeout=5)
        return response.json()
    except Exception as e:
        print("❌ Error fetching option chain:", e)
        return None

def extract_ivdi(data):
    try:
        atm_strikes = data['records']['strikePrices']
        mid_index = len(atm_strikes) // 2
        atm_strike = atm_strikes[mid_index]

        for item in data['records']['data']:
            if item.get('strikePrice') == atm_strike:
                iv_call = item['CE']['impliedVolatility'] if 'CE' in item else 0
                iv_put = item['PE']['impliedVolatility'] if 'PE' in item else 0
                ivdi = iv_call - iv_put
                return iv_call, iv_put, ivdi
    except Exception as e:
        print("❌ Error computing IVDI:", e)
        return 0, 0, 0

def live_ivdi_plot(interval=30):
    ivdi_vals, times = [], []
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))

    while True:
        data = get_option_chain()
        if data:
            iv_call, iv_put, ivdi = extract_ivdi(data)
            timestamp = datetime.now().strftime("%H:%M:%S")
            times.append(timestamp)
            ivdi_vals.append(ivdi)

            ax.clear()
            ax.plot(times, ivdi_vals, marker='o', label='IVDI')
            ax.axhline(0.5, color='green', linestyle='--', label='Bullish Divergence')
            ax.axhline(-0.5, color='red', linestyle='--', label='Bearish Divergence')
            ax.set_title("Implied Volatility Divergence Index (IVDI)")
            ax.set_xlabel("Time")
            ax.set_ylabel("IVDI = IV(Call) - IV(Put)")
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

            print(f"[{timestamp}] IV Call: {iv_call:.2f}, IV Put: {iv_put:.2f}, IVDI: {ivdi:.4f}")

            if ivdi > 0.5:
                print("🚀 Market bias toward bullish volatility")
            elif ivdi < -0.5:
                print("🔻 Market bias toward downside protection")

            plt.pause(1)
        time.sleep(interval)

if __name__ == "__main__":
    live_ivdi_plot()
