import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def generate_tick_data(n=100):
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.2)
    bid_vol = np.random.randint(100, 500, size=n)
    ask_vol = np.random.randint(100, 500, size=n)

    df = pd.DataFrame({
        'price': prices,
        'bid_vol': bid_vol,
        'ask_vol': ask_vol
    })

    df['return'] = df['price'].diff().fillna(0)
    df['direction'] = (df['return'] > 0).astype(int)
    df['ofi'] = df['bid_vol'] - df['ask_vol']
    df['momentum'] = df['price'].diff(3).fillna(0)

    return df.dropna()

def train_model(df):
    X = df[['ofi', 'momentum']]
    y = df['direction']
    model = LogisticRegression()
    model.fit(X, y)
    return model
