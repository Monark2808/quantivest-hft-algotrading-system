import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from datetime import datetime

# === Black-Scholes Greeks Calculation ===
def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) - (r*K*np.exp(-r*T) * norm.cdf(d2))
    else:  # put
        delta = -norm.cdf(-d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) + (r*K*np.exp(-r*T) * norm.cdf(-d2))

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    return delta, gamma, theta

# === Simulated Real-time Tracker ===
def run_gvt_tracker():
    S = 20000  # Current spot price (e.g. NIFTY)
    K = 20000  # ATM Strike
    r = 0.05   # Risk-free rate
    sigma = 0.18  # Implied Volatility
    T_days = 7  # 1 week to expiry

    delta_hist, gamma_hist, theta_hist = [], [], []
    time_hist = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    while True:
        T = T_days / 365.0
        delta, gamma, theta = calculate_greeks(S, K, T, r, sigma)

        timestamp = datetime.now().strftime("%H:%M:%S")
        time_hist.append(timestamp)
        delta_hist.append(delta)
        gamma_hist.append(gamma)
        theta_hist.append(theta)

        ax.clear()
        ax.plot(time_hist, delta_hist, label="Delta")
        ax.plot(time_hist, gamma_hist, label="Gamma")
        ax.plot(time_hist, theta_hist, label="Theta")
        ax.set_title("Greeks Velocity Tracker (GVT)")
        ax.set_ylabel("Value")
        ax.set_xlabel("Time")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        print(f"[{timestamp}] Δ={delta:.4f}, Γ={gamma:.4f}, Θ={theta:.4f}")

        # Simulate tiny spot price movement to trigger change
        S += np.random.uniform(-10, 10)
        sigma += np.random.uniform(-0.002, 0.002)
        T_days -= 1/96  # simulate 15-min step

        if T_days <= 0:
            break

        plt.pause(1)
        time.sleep(15)

if __name__ == "__main__":
    run_gvt_tracker()
