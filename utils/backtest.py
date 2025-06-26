import numpy as np
import pandas as pd

def generate_backtest_data(n=100, seed=42):
    np.random.seed(seed)
    prices = 100 + np.cumsum(np.random.normal(0, 1, n))  # random walk

    ovii = np.random.uniform(-1, 1, n)
    ivdi = np.random.uniform(-1, 1, n)
    map_prob = np.random.uniform(0, 1, n)

    df = pd.DataFrame({
        'price': prices,
        'ovii': ovii,
        'ivdi': ivdi,
        'map_prob': map_prob
    })

    return df
