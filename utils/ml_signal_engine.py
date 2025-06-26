import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def generate_ml_training_data():
    # Simulated dataset
    size = 300
    data = {
        "map_prob": np.random.rand(size),
        "ovii": np.random.uniform(-1, 1, size),
        "ivdi": np.random.uniform(-1, 1, size),
        "return": np.random.normal(0, 1, size)
    }

    df = pd.DataFrame(data)

    # Label generation
    def label_row(row):
        if row["map_prob"] > 0.7 and row["ovii"] > 0.5 and row["ivdi"] > 0.4:
            return "BUY"
        elif row["map_prob"] < 0.3 and row["ovii"] < -0.5 and row["ivdi"] < -0.4:
            return "SELL"
        else:
            return "HOLD"

    df["label"] = df.apply(label_row, axis=1)

    # Balance the classes
    df_buy = df[df.label == "BUY"]
    df_sell = df[df.label == "SELL"]
    df_hold = df[df.label == "HOLD"]

    min_size = min(len(df_buy), len(df_sell), len(df_hold))
    df_balanced = pd.concat([
        resample(df_buy, replace=True, n_samples=min_size, random_state=42),
        resample(df_sell, replace=True, n_samples=min_size, random_state=42),
        resample(df_hold, replace=True, n_samples=min_size, random_state=42)
    ])

    return df_balanced

def train_classifier(df):
    X = df[["map_prob", "ovii", "ivdi", "return"]]
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    return clf
