import numpy as np
from scipy.stats import norm

def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) - (r*K*np.exp(-r*T) * norm.cdf(d2))
    else:
        delta = -norm.cdf(-d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) + (r*K*np.exp(-r*T) * norm.cdf(-d2))

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return delta, gamma, theta
