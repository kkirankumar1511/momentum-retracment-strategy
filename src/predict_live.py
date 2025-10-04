import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from config import load_config
from data_loader import get_kite_client
from instrument_token import InstrumentTokenManager
from pipelines.data_pipeline import MarketDataFetcher
from pipelines.feature_pipeline import IntradayFeatureEngineer


@dataclass
class PredictAgent:
    """Loads trained artefacts and produces a multi-step close forecast."""

    def __init__(self) -> None:
        self.cfg = load_config()
        data_cfg = self.cfg['data']
        kite_cfg = self.cfg['kite']

        self.lookback = int(data_cfg['lookback'])
        self.horizon = int(data_cfg.get('predict_horizon', 1))
        self.feature_engineer = IntradayFeatureEngineer(self.lookback, self.horizon)

        self.kite = get_kite_client(kite_cfg['api_key'], kite_cfg['access_token'])
        self.instrument_token_manager = InstrumentTokenManager()
        if self.instrument_token_manager.get_instrument_tokens() is None:
            self.instrument_token_manager.set_instrument_tokens()
        self.fetcher = MarketDataFetcher(
            kite_client=self.kite,
            interval=data_cfg.get('interval', '15minute'),
            instrument_tokens=self.instrument_token_manager.get_instrument_tokens(),
        )

    def load_model_and_scaler(self, instrument_name: str):
        model_path = os.path.join(
            self.cfg['training']['model_save_path'],
            f"{instrument_name}_return_lstm.keras",
        )
        scaler_path = os.path.join(
            self.cfg['training']['scaler_save_path'],
            f"{instrument_name}_return.pkl",
        )
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            missing = []
            if not os.path.exists(model_path):
                missing.append(model_path)
            if not os.path.exists(scaler_path):
                missing.append(scaler_path)
            missing_str = ', '.join(missing)
            raise FileNotFoundError(
                f"Missing trained artefacts for {instrument_name}. "
                f"Expected files: {missing_str}. Run training once to create them."
            )

        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler

    def predict_next_intervals(
        self,
        instrument_name: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        model, scaler = self.load_model_and_scaler(instrument_name)
        raw_df = self.fetcher.fetch(instrument_name, from_date, to_date)
        feature_frame = self.feature_engineer.prepare_inference_frame(raw_df)
        scaled_features, _ = self.feature_engineer.scale_features(feature_frame, scaler=scaler)
        sequence = self.feature_engineer.build_prediction_sequence(scaled_features)

        predicted_closes = model.predict(sequence, verbose=0)[0]
        latest_row = feature_frame.iloc[-1]
        forecast_frame = self._build_forecast_frame(latest_row, predicted_closes)
        return forecast_frame

    def _build_forecast_frame(self, row: pd.Series, predicted_closes: np.ndarray) -> pd.DataFrame:
        base_timestamp = pd.to_datetime(row['date'])
        interval_delta = self._resolve_interval_delta()
        predicted_series = np.asarray(predicted_closes, dtype=float).reshape(-1)
        horizon = predicted_series.size
        if horizon == 0:
            raise ValueError('predicted_closes must contain at least one forecast value.')

        steps = np.arange(1, horizon + 1)
        timestamps = [base_timestamp + int(step) * interval_delta for step in steps]
        current_close = float(row['close'])
        predicted_returns = (predicted_series - current_close) / current_close

        records = []
        for step, ts, pred_close, pred_return in zip(steps, timestamps, predicted_series, predicted_returns):
            records.append(
                {
                    'interval_ahead': int(step),
                    'forecast_timestamp': ts,
                    'predicted_close': float(pred_close),
                    'predicted_return': float(pred_return),
                }
            )

        frame = pd.DataFrame.from_records(records)
        frame['current_close'] = current_close
        frame['predicted_move'] = frame['predicted_close'] - current_close
        return frame[
            [
                'interval_ahead',
                'forecast_timestamp',
                'current_close',
                'predicted_close',
                'predicted_move',
                'predicted_return',
            ]
        ]

    def _resolve_interval_delta(self) -> pd.Timedelta:
        interval = str(self.cfg['data'].get('interval', '15minute')).lower()
        if interval.endswith('minute'):
            minutes = int(interval.replace('minute', ''))
            return pd.Timedelta(minutes=minutes)
        if interval.endswith('hour'):
            hours = int(interval.replace('hour', ''))
            return pd.Timedelta(hours=hours)
        raise ValueError(f"Unsupported interval format: {interval}")
