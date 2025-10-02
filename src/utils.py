import joblib
import numpy as np
import pandas as pd
from pathlib import Path

def save_scaler(scaler, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)

def get_instrument_token(instrument_token_list, instrument):
    token = instrument_token_list[(instrument_token_list['tradingsymbol'] == instrument) & (
            instrument_token_list['exchange'] == "NSE")]['instrument_token'].values[0]
    return token

def create_sequences(data, lookback, horizon=1, target_indices=None):
    """
    Create input sequences (X) and output labels (y) for LSTM training.

    Parameters
    ----------
    data : np.ndarray
        Scaled feature matrix (num_samples x num_features).
    lookback : int
        Number of timesteps to look back (sequence length).
    horizon : int
        Prediction horizon (number of future steps).
    target_indices : list[int] or None
        Indices of columns in `data` to use as target.
        Example: [3] for 'Close' if 'Close' is the 4th column.

    Returns
    -------
    X : np.ndarray
        Shape (num_sequences, lookback, num_features).
    y : np.ndarray
        Shape (num_sequences, horizon * len(target_indices)) if multi-output,
        or (num_sequences, horizon) if single target.
    """
    X, y = [], []
    num_samples = len(data)

    for i in range(num_samples - lookback - horizon + 1):
        seq_x = data[i : i + lookback]  # sequence window
        seq_y = data[i + lookback : i + lookback + horizon]

        if target_indices is not None:
            seq_y = seq_y[:, target_indices]  # keep only target columns

        # flatten horizon*targets into 1D
        y.append(seq_y.flatten())
        X.append(seq_x)

    X = np.array(X)
    y = np.array(y)

    return X, y

