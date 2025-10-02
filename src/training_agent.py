import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import load_config
from data_loader import get_kite_client, load_from_kite
from features import add_technical_indicators
from utils import create_sequences, save_model, save_scaler, get_instrument_token
from model import build_lstm_model
from instrument_token import InstrumentTokenManager
import talib

instrument_token_manager = InstrumentTokenManager()

class TrainingAgent:

    def __init__(self):
        self.instrument_token_list = instrument_token_manager.get_instrument_tokens()

    def train_model(self, kite_instrument=None, from_date=None, to_date=None, test_split=0.2):
        cfg = load_config()
        data_cfg = cfg['data']
        train_cfg = cfg['training']
        lookback = data_cfg['lookback']

        # --- Load Data ---
        kite_cfg = cfg['kite']
        kite = get_kite_client(kite_cfg['api_key'], kite_cfg['access_token'])
        instrument_token = get_instrument_token(self.instrument_token_list, kite_instrument)
        df = load_from_kite(kite, instrument_token, from_date, to_date, interval="day")

        # --- Add technical indicators ---
        df = add_technical_indicators(df)

        # --- Compute returns for target ---
        df['return'] = df['close'].pct_change()
        df.dropna(inplace=True)

        feature_cols = ['open','high','low','close','volume','rsi','ema20','ema50','macd','macdsignal','bb_upper','bb_middle','bb_lower']
        X_df = df[feature_cols].copy()
        y = df['return'].values.reshape(-1,1)  # target

        # --- Scale features ---
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X_df.values)

        # --- Create sequences ---
        X_seq, y_seq = create_sequences(
            X_scaled,
            lookback=lookback,
            horizon=1,
            target_indices=[feature_cols.index('close')]
        )
        # Fix broadcasting error
        y_seq[:,0] = y[lookback:].flatten()

        # --- Train/test split ---
        split_idx = int(len(X_seq)*(1-test_split))
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        # --- Build model ---
        model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), output_dim=1)

        # --- Train ---
        history = model.fit(
            X_train, y_train,
            epochs=train_cfg['epochs'],
            batch_size=train_cfg['batch_size'],
            validation_split=0.1,
            verbose=1
        )

        # --- Predict ---
        y_pred = model.predict(X_test)
        # Reconstruct actual prices
        last_close_test = df['close'].values[lookback+split_idx-1:-1]
        pred_close = last_close_test * (1 + y_pred.flatten())
        actual_close = df['close'].values[lookback+split_idx:]

        # --- Metrics ---
        rmse = np.sqrt(mean_squared_error(actual_close, pred_close))
        mae = mean_absolute_error(actual_close, pred_close)
        r2 = r2_score(actual_close, pred_close)
        print(f"\n✅ Test Metrics for {kite_instrument}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")

        # --- Plot ---
        plt.figure(figsize=(12,6))
        plt.plot(actual_close, label='Actual Close')
        plt.plot(pred_close, label='Predicted Close')
        plt.title(f"Predicted vs Actual Close Price ({kite_instrument})")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

        # --- Save model & scaler ---
        model_path = os.path.join(train_cfg['model_save_path'], f"{kite_instrument}_return_lstm.keras")
        scaler_path = os.path.join(train_cfg['scaler_save_path'], f"{kite_instrument}_return.pkl")
        save_model(model, model_path)
        save_scaler(scaler_X, scaler_path)
        print(f"\n✅ Model saved to {model_path}")
        print(f"✅ Scaler saved to {scaler_path}")

        return model, scaler_X
