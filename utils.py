# smart_grid_federated/utils.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import numpy as np

def load_and_preprocess_data(path: str):
    df = pd.read_csv(path)
    df = df.drop(columns=["Timestamp"])
    X = df.drop(columns=["Predicted Load (kW)"]).values
    y = df["Predicted Load (kW)"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def split_data_among_clients(X, y, num_clients=5):
    X_splits = np.array_split(X, num_clients)
    y_splits = np.array_split(y, num_clients)
    return list(zip(X_splits, y_splits))

def load_test_data(path: str):
    X_scaled, y, _ = load_and_preprocess_data(path)
    split_idx = int(len(X_scaled) * 0.8)
    return X_scaled[split_idx:], y[split_idx:]
