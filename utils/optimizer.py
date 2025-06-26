import numpy as np
import pandas as pd
from utils.backtest import generate_backtest_data

def optimize_thresholds():
    df = generate_backtest_data(100)
    best_pnl = -np.inf
    best_combo = None
    logs = []

    map_range = np.arange(0.5, 0.9, 0.1)
    ovii_range = np.arange(0.4, 0.8, 0.1)
    ivdi_range = np.arange(0.3, 0.7, 0.1)

    for map_th in map_range:
        for ovii_th in ovii_range:
            for ivdi_th in ivdi_range:
                pnl = simulate_strategy(df, map_th, ovii_th, ivdi_th)
                logs.append({
                    "MAP": map_th,
                    "OVII": ovii_th,
                    "IVDI": ivdi_th,
                    "PnL": pnl
                })
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_combo = (map_th, ovii_th, ivdi_th)

    log_df = pd.DataFrame(logs)
    return log_df, best_combo, best_pnl


def simulate_strategy(df, map_th, ovii_th, ivdi_th):
    position = 0
    entry_price = None
    pnl = 0

    for _, row in df.iterrows():
        if (
            row['map_prob'] > map_th and
            row['ovii'] > ovii_th and
            row['ivdi'] > ivdi_th
        ):
            signal = 1
        elif (
            row['map_prob'] < (1 - map_th) and
            row['ovii'] < -ovii_th and
            row['ivdi'] < -ivdi_th
        ):
            signal = -1
        else:
            signal = 0

        if signal != 0 and position == 0:
            position = signal
            entry_price = row['price']
        elif signal == 0 and position != 0:
            pnl += (row['price'] - entry_price) * position
            position = 0
            entry_price = None

    return pnl
