import os
from dataclasses import dataclass
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from config import load_config
from data_loader import get_kite_client
from instrument_token import InstrumentTokenManager
from pipelines.data_pipeline import MarketDataFetcher
from pipelines.feature_pipeline import IntradayFeatureEngineer
from strategies.intraday import IntradayStrategy


@dataclass
class PredictAgent:
    """Loads trained artefacts to deliver the next intraday signal."""

    def __init__(self) -> None:
        self.cfg = load_config()
        data_cfg = self.cfg['data']
        kite_cfg = self.cfg['kite']

        self.lookback = int(data_cfg['lookback'])
        self.horizon = int(data_cfg.get('predict_horizon', 1))
        self.feature_engineer = IntradayFeatureEngineer(self.lookback, self.horizon)
        strategy_cfg = self.cfg.get('strategy', {})
        self.strategy = IntradayStrategy(**strategy_cfg)

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

    def predict_next_interval(
        self,
        instrument_name: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        model, scaler = self.load_model_and_scaler(instrument_name)
        raw_df = self.fetcher.fetch(instrument_name, from_date, to_date)
        feature_frame = self.feature_engineer.prepare_inference_frame(raw_df)
        self._calibrate_min_predicted_return(feature_frame)
        scaled_features, _ = self.feature_engineer.scale_features(feature_frame, scaler=scaler)
        sequence = self.feature_engineer.build_prediction_sequence(scaled_features)

        predicted_return = float(model.predict(sequence, verbose=0)[0, 0])
        latest_row = feature_frame.iloc[-1]
        predicted_close = float(latest_row['close'] * (1 + predicted_return))

        strategy_row = self._build_strategy_row(latest_row, predicted_return, predicted_close)
        strategy_df = self.strategy.apply(strategy_row)
        return strategy_df

    def _build_strategy_row(
        self,
        row: pd.Series,
        predicted_return: float,
        predicted_close: float,
    ) -> pd.DataFrame:
        payload: Dict[str, float] = {
            'timestamp': row['date'],
            'Close': row['close'],
            'RSI': row['rsi'],
            'EMA_200': row['ema200'],
            'EMA_50': row['ema50'],
            'EMA_20': row['ema20'],
            'Signal': int(predicted_return > 0),
            'predicted_return': predicted_return,
            'predicted_close': predicted_close,
            'projected_move': predicted_close - row['close'],
            'future_close': row.get('future_close', np.nan),
        }
        return pd.DataFrame([payload])

    def _calibrate_min_predicted_return(self, frame: pd.DataFrame) -> float:
        strategy_cfg = self.cfg.get('strategy', {})
        configured = strategy_cfg.get('min_predicted_return')
        percentile = float(strategy_cfg.get('min_return_percentile', 85))
        if configured is None or (isinstance(configured, str) and configured.lower() == 'auto'):
            abs_returns = frame['target_return'].abs().dropna()
            if abs_returns.empty:
                resolved = 0.0
            else:
                resolved = float(np.percentile(abs_returns, percentile))
            self.strategy.min_predicted_return = resolved
        else:
            resolved = float(configured)
            self.strategy.min_predicted_return = resolved
        return resolved
