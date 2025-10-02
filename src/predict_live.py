import os
import numpy as np
import pandas as pd
from datetime import timedelta
import joblib
from config import load_config
from data_loader import get_kite_client, load_from_kite
from src.features import add_technical_indicators
from utils import get_instrument_token
from instrument_token import InstrumentTokenManager
from tensorflow.keras.models import load_model

instrument_token_manager = InstrumentTokenManager()

class PredictAgent:

    def __init__(self):
        self.instrument_token_list = InstrumentTokenManager().get_instrument_tokens()

    def load_model_and_scaler(self,cfg,instrument_name):
        model_path = os.path.join(cfg['training']['model_save_path'], f"{instrument_name}_return_lstm.keras")
        scaler_path = os.path.join(cfg['training']['scaler_save_path'], f"{instrument_name}_return.pkl")
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler

    def predict_next_close(self, kite_instrument, last_close, last_df, horizon=5, ci=0.95):
        """
        Predict next `horizon` days' Close with confidence interval using LSTM return model.
        """
        cfg = load_config()
        data_cfg = cfg['data']
        model, scaler = self.load_model_and_scaler(cfg, kite_instrument)

        # Add technical indicators
        df = add_technical_indicators(last_df.copy())
        df = df.dropna()
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'ema20', 'ema50', 'macd',
                        'macdsignal', 'bb_upper', 'bb_middle', 'bb_lower']
        X_scaled = scaler.transform(df[feature_cols].values)


        # Prepare lookback sequence
        lookback = min(data_cfg['lookback'], len(X_scaled))
        seq = X_scaled[-lookback:].reshape(1, lookback, len(feature_cols))

        predicted_returns = []
        predicted_close = []
        prev_close = last_close

        # Estimate historical volatility to compute CI
        returns_series = df['close'].pct_change().dropna()
        volatility = returns_series.std()

        for _ in range(horizon):
            pred_return = model.predict(seq, verbose=0)[0,0]
            pred_price = prev_close * (1 + pred_return)

            # Save predictions
            predicted_returns.append(pred_return)
            predicted_close.append(pred_price)

            # Update sequence with predicted Close
            seq[0, :-1, :] = seq[0, 1:, :]
            seq[0, -1, feature_cols.index('close')] = pred_price
            prev_close = pred_price

        # Confidence intervals
        from scipy.stats import norm
        z = norm.ppf(0.5 + ci/2)
        predicted_close = np.array(predicted_close)
        lower_bound = predicted_close * (1 - z * volatility)
        upper_bound = predicted_close * (1 + z * volatility)

        # Dates for predicted horizon
        last_df['Date'] = pd.to_datetime(last_df['date'])  # or whatever column has the timestamp
        last_df.set_index('Date', inplace=True)
        last_date = pd.to_datetime(last_df.index[-1])
        prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')

        # Build dataframe
        df_pred = pd.DataFrame({
            'predicted_close': predicted_close,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }, index=prediction_dates)

        return df_pred

