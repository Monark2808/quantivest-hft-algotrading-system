import time
import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv
import plotly.express as px
from io import BytesIO
import threading


# --- Load environment variables early ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Set Python path for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Custom imports ---
from utils.greeks import calculate_greeks
from utils.fetch import get_ivdi
from utils.map import generate_tick_data, train_model
from utils.backtest import generate_backtest_data
from utils.optimizer import optimize_thresholds
from utils.ml_signal_engine import generate_ml_training_data, train_classifier
from utils.webhook import send_trade_alert
from utils.trader import place_order
from utils.live_feed_alpaca import start_ws, get_latest_price


# --- Page setup ---
st.set_page_config(layout="wide")
st.title("HFT Indicator Dashboard")

# --- Sidebar ---
st.sidebar.title("⚙️ Backtest Tuning")
symbol = st.sidebar.selectbox("Select Stock Symbol", ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"], index=0)
threshold_map = st.sidebar.slider("MAP Threshold", 0.0, 1.0, 0.7, 0.01)
threshold_ovii = st.sidebar.slider("OVII Threshold", -1.0, 1.0, 0.6, 0.01)
threshold_ivdi = st.sidebar.slider("IVDI Threshold", -1.0, 1.0, 0.5, 0.01)

# --- Simulated OVII for US Markets ---
@st.cache_data(ttl=30)
def get_ovii(symbol):
    call_volume = np.random.randint(500, 2000)
    put_volume = np.random.randint(500, 2000)
    ovii = (call_volume - put_volume) / (call_volume + put_volume)
    return round(ovii, 4), call_volume, put_volume

# --- BASP Simulation ---
def simulate_basp():
    bid = np.random.randint(100, 500)
    ask = np.random.randint(100, 500)
    basp = (ask - bid) / (ask + bid)
    return round(basp, 4), bid, ask

# --- OVII + BASP ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Option Volume Imbalance Index (OVII)")
    ovii, ce_vol, pe_vol = get_ovii(symbol)
    st.metric(label="OVII", value=ovii)
    st.write(f"Call Volume: {ce_vol} | Put Volume: {pe_vol}")
    if ovii > 0.6:
        st.success("📈 Bullish Bias")
    elif ovii < -0.6:
        st.error("📉 Bearish Bias")
    else:
        st.info("📊 Neutral Market")
with col2:
    st.subheader("📉 Bid-Ask Spread Pressure (BASP)")
    basp, bid, ask = simulate_basp()
    st.metric(label="BASP", value=basp)
    st.write(f"Bid Vol: {bid} | Ask Vol: {ask}")
    if basp > 0.5:
        st.error("🔻 Sellers Dominating")
    elif basp < -0.5:
        st.success("🚀 Buyers Dominating")
    else:
        st.info("📊 Balanced Order Flow")

# --- GVT Section ---
st.markdown("---")
st.subheader("Greeks Velocity Tracker (GVT)")
spot, strike, r, sigma, days = 20000, 20000, 0.05, 0.18, 7
gvt_data = {"time": [], "delta": [], "gamma": [], "theta": []}
for _ in range(15):
    T = days / 365
    delta, gamma, theta = calculate_greeks(spot, strike, T, r, sigma)
    gvt_data["time"].append(datetime.now().strftime("%H:%M:%S"))
    gvt_data["delta"].append(delta)
    gvt_data["gamma"].append(gamma)
    gvt_data["theta"].append(theta)
    spot += np.random.uniform(-15, 15)
    sigma += np.random.uniform(-0.003, 0.003)
    days -= 0.1
df_gvt = pd.DataFrame(gvt_data)
st.line_chart(df_gvt.set_index("time"))

# --- IVDI ---
st.markdown("---")
st.subheader("Implied Volatility Divergence Index (IVDI)")
iv_call, iv_put, ivdi = get_ivdi()
col3, col4, col5 = st.columns(3)
with col3: st.metric("Call IV (ATM)", value=f"{iv_call} %")
with col4: st.metric("Put IV (ATM)", value=f"{iv_put} %")
with col5:
    st.metric("IVDI", value=ivdi)
    if ivdi > 0.5:
        st.success("📈 Bullish Volatility Bias")
    elif ivdi < -0.5:
        st.error("📉 Bearish Volatility Bias")
    else:
        st.info("📊 Neutral Volatility Outlook")

# --- MAP ---
st.markdown("---")
st.subheader("🤖 Microstructure Alpha Predictor (MAP)")
df_map = generate_tick_data(100)
model = train_model(df_map)
last_tick_df = df_map[['ofi', 'momentum']].iloc[[-1]]
prob = model.predict_proba(last_tick_df)[0][1]
st.metric(label="P(Price ↑)", value=f"{prob:.2f}")
if prob > 0.7:
    st.success("BUY SIGNAL")
elif prob < 0.3:
    st.error("🔻 SELL SIGNAL")
else:
    st.info("📊 HOLD / NO CLEAR DIRECTION")

# --- ML-Based Signal Engine ---
st.markdown("---")
st.subheader("🧠 ML-Based Signal Engine (Logistic Regression)")

df_ml = generate_ml_training_data()
model_ml = train_classifier(df_ml)
live_features = pd.DataFrame([{
    "map_prob": prob,
    "ovii": ovii,
    "ivdi": ivdi,
    "return": df_ml["return"].iloc[-1]
}])

ml_prediction = model_ml.predict(live_features)[0]
class_probs = model_ml.predict_proba(live_features)[0]
classes = model_ml.classes_
class_probs_dict = dict(zip(classes, class_probs))

emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}

st.markdown(f"### {emoji[ml_prediction]} **Prediction: {ml_prediction}**")
st.metric("📈 MAP Probability", f"{prob:.2f}")

conf_df = pd.DataFrame([class_probs_dict]).T.rename(columns={0: "Confidence"}).sort_values("Confidence", ascending=False)
conf_df["Confidence"] = conf_df["Confidence"].apply(lambda x: f"{x:.1%}")
st.subheader("Class Probabilities:")
st.dataframe(conf_df, use_container_width=True)
st.bar_chart(pd.DataFrame(class_probs, index=classes))
st.subheader("Features fed to model:")
st.json(live_features.to_dict(orient="records")[0])

WEBHOOK_URL = "https://discord.com/api/webhooks/1385350319327805542/sXNvH1ZyCGujwYDgSwH5tuHjKJW9Uuq_v4tqMMRA5KEkVgnGeO_ppYXxlHFT6YBuvZEX"

if ml_prediction in ["BUY", "SELL"]:
    send_trade_alert(ml_prediction, live_features.iloc[0], class_probs_dict, WEBHOOK_URL, symbol)
    st.success("✅ Trade alert sent to Discord!")

    trade_result = place_order(symbol, 1, ml_prediction.lower())
    st.info(f"💼 Trade Result: {trade_result}")
else:
    st.caption("🛁 No alert sent.")

# --- Final Signal Engine ---
st.markdown("---")
st.subheader("Final Trade Signal Engine")
if prob > 0.7 and ovii > 0.6 and ivdi > 0.5:
    signal = "📈 BUY"
    reason = "Strong MAP + OVII + IVDI alignment"
elif prob < 0.3 and ovii < -0.6 and ivdi < -0.5:
    signal = "📉 SELL"
    reason = "Bearish signal across MAP, OVII, and IVDI"
else:
    signal = "⏸️ HOLD"
    reason = "No high-confidence consensus"
st.metric(label="Trade Signal", value=signal)
st.info(f"Reason: {reason}")

# --- Backtest ---
st.markdown("---")
st.subheader("🔁 Backtesting Engine")
df_bt = generate_backtest_data(100)
signals, positions, returns, entry_price = [], [], [], None
for _, row in df_bt.iterrows():
    if row['map_prob'] > threshold_map and row['ovii'] > threshold_ovii and row['ivdi'] > threshold_ivdi:
        sig = "BUY"
    elif row['map_prob'] < (1 - threshold_map) and row['ovii'] < -threshold_ovii and row['ivdi'] < -threshold_ivdi:
        sig = "SELL"
    else:
        sig = "HOLD"
    signals.append(sig)
    if sig in ["BUY", "SELL"] and not entry_price:
        entry_price = row['price']
        positions.append(1 if sig == "BUY" else -1)
    elif sig == "HOLD":
        positions.append(positions[-1] if positions else 0)
    else:
        positions.append(positions[-1] if positions else 0)
    pnl = (row['price'] - entry_price) * positions[-1] if entry_price else 0
    returns.append(pnl)
df_bt["signal"] = signals
df_bt["position"] = positions
df_bt["pnl"] = returns
st.line_chart(df_bt["price"], height=200)
st.line_chart(df_bt["pnl"], height=200)
st.dataframe(df_bt[df_bt["signal"] != "HOLD"].tail(10))

# --- Threshold Optimizer ---
st.markdown("---")
st.subheader("Threshold Auto-Optimizer")
if st.button("Run Optimization"):
    with st.spinner("Running grid search..."):
        result_df, best_combo, best_pnl = optimize_thresholds()
        st.success(f"Best Combo: MAP ≥ {best_combo[0]}, OVII ≥ {best_combo[1]}, IVDI ≥ {best_combo[2]}")
        st.metric("Best PnL", f"{best_pnl:.2f}")
        st.dataframe(result_df.sort_values("PnL", ascending=False).head(10))

# --- Enhanced Live ML-Based Prediction + Auto-Trading ---

st.markdown("---")
st.subheader("Real-Time ML Signal Engine")

live_signal_placeholder = st.empty()
live_conf_placeholder = st.empty()
live_trade_placeholder = st.empty()

# ✅ Start WebSocket only once to avoid 'already opened' errors
if "ws_started" not in st.session_state:
    st.session_state.ws_started = False

if not st.session_state.ws_started:
    threading.Thread(target=start_ws, daemon=True).start()
    st.session_state.ws_started = True

st.info("📡 Monitoring live feed & auto-trading every 10s for 1 minute...")

for _ in range(6):  # 6 loops × 10s = 60 seconds
    price = get_latest_price()
    if price:
        # Recompute MAP model with new price
        df_map = generate_tick_data(100)
        model = train_model(df_map)
        last_tick_df = df_map[['ofi', 'momentum']].iloc[[-1]]
        prob = model.predict_proba(last_tick_df)[0][1]

        # Recalculate OVII & IVDI
        ovii, _, _ = get_ovii(symbol)
        iv_call, iv_put, ivdi = get_ivdi()

        # Predict using ML Model
        df_ml = generate_ml_training_data()
        model_ml = train_classifier(df_ml)

        live_features = pd.DataFrame([{
            "map_prob": prob,
            "ovii": ovii,
            "ivdi": ivdi,
            "return": df_ml["return"].iloc[-1]
        }])

        ml_prediction = model_ml.predict(live_features)[0]
        class_probs = model_ml.predict_proba(live_features)[0]
        classes = model_ml.classes_
        class_probs_dict = dict(zip(classes, class_probs))
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}

        live_signal_placeholder.markdown(
            f"### {emoji[ml_prediction]} **Live Prediction: {ml_prediction}** @ ${price:.2f}"
        )

        conf_df = pd.DataFrame([class_probs_dict]).T.rename(columns={0: "Confidence"}).sort_values("Confidence", ascending=False)
        conf_df["Confidence"] = conf_df["Confidence"].apply(lambda x: f"{x:.1%}")
        live_conf_placeholder.dataframe(conf_df, use_container_width=True)

        if ml_prediction in ["BUY", "SELL"]:
            send_trade_alert(ml_prediction, live_features.iloc[0], class_probs_dict, WEBHOOK_URL, symbol)
            trade_result = place_order(symbol, 1, ml_prediction.lower())
            live_trade_placeholder.success(f"✅ Trade Executed: {trade_result}")
        else:
            live_trade_placeholder.info("⏸ HOLD - No trade placed.")
    else:
        live_signal_placeholder.warning("📡 Waiting for live price...")

    time.sleep(10)


# --- Return Analyzer ---
st.markdown("---")
st.subheader("💰 Trade Performance Analyzer")

import os

if os.path.exists("trades.csv"):
    df_trades = pd.read_csv("trades.csv")
    df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"])
    df_trades.sort_values("timestamp", inplace=True)

    # PnL Calculation (assumes 1 share per trade)
    df_trades["side_numeric"] = df_trades["side"].map({"BUY": 1, "SELL": -1})
    df_trades["pnl"] = df_trades["side_numeric"] * df_trades["price"].diff().fillna(0)

    total_trades = len(df_trades)
    profitable_trades = (df_trades["pnl"] > 0).sum()
    total_pnl = df_trades["pnl"].sum()

    st.metric("📊 Total Trades", total_trades)
    st.metric("✅ Win Rate", f"{(profitable_trades / total_trades) * 100:.2f}%" if total_trades > 0 else "0%")
    st.metric("💵 Net PnL", f"${total_pnl:.2f}")

    st.line_chart(df_trades.set_index("timestamp")["pnl"].cumsum(), height=200)
    st.dataframe(df_trades.tail(10))
else:
    st.warning("🚫 No trades found. Execute a few trades to enable analysis.")


# --- Footer ---
st.caption("📡 OVII & IVDI simulated | MAP, GVT, BASP simulated | Auto-optimizer + ML engine + Webhook alerts enabled")

