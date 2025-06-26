import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datetime import datetime
import time

# === Simulate Tick Data ===
def generate_tick_data(n_ticks=500):
    np.random.seed(42)
    price = 100 + np.cumsum(np.random.randn(n_ticks) * 0.2)
    bid_vol = np.random.randint(100, 500, size=n_ticks)
    ask_vol = np.random.randint(100, 500, size=n_ticks)

    df = pd.DataFrame({
        'price': price,
        'bid_vol': bid_vol,
        'ask_vol': ask_vol
    })

    df['return'] = df['price'].diff().fillna(0)
    df['direction'] = (df['return'] > 0).astype(int)
    df['ofi'] = df['bid_vol'] - df['ask_vol']
    df['momentum'] = df['price'].diff(3).fillna(0)

    return df.dropna()

# === Train Logistic Model ===
def train_model(df):
    X = df[['ofi', 'momentum']]
    y = df['direction']

    model = LogisticRegression()
    model.fit(X, y)
    return model

# === Live Predictor Loop ===
def live_predictor(model, df):
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))

    preds = []
    times = []

    for i in range(len(df)):
        x_live = df[['ofi', 'momentum']].iloc[i].values.reshape(1, -2)
        prob = model.predict_proba(x_live)[0][1]
        preds.append(prob)
        times.append(datetime.now().strftime("%H:%M:%S"))

        ax.clear()
        ax.plot(times, preds, label="P(Price Up)", marker='o')
        ax.axhline(0.7, color='green', linestyle='--', label='Buy Signal Threshold')
        ax.axhline(0.3, color='red', linestyle='--', label='Sell Signal Threshold')
        ax.set_title("MAP: Microstructure Alpha Predictor")
        ax.set_ylabel("Probability of Up Move")
        ax.set_xlabel("Time")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        signal = "👀 WAIT"
        if prob > 0.7:
            signal = "🚀 BUY SIGNAL"
        elif prob < 0.3:
            signal = "🔻 SELL SIGNAL"

        print(f"[{times[-1]}] Predicted Prob ↑: {prob:.4f} => {signal}")
        plt.pause(1)
        time.sleep(2)

if __name__ == "__main__":
    tick_df = generate_tick_data()
    model = train_model(tick_df)
    live_predictor(model, tick_df.tail(100))
