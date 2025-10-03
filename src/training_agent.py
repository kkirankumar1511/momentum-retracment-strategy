import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import load_config
from data_loader import get_kite_client
from instrument_token import InstrumentTokenManager
from model import LSTMReturnForecaster
from pipelines.data_pipeline import MarketDataFetcher
from pipelines.feature_pipeline import IntradayFeatureEngineer
from strategies.intraday import IntradayStrategy
from utils import save_model, save_scaler


@dataclass
class TrainingAgent:
    """Coordinates data loading, feature prep, model training and evaluation."""

    def __init__(self) -> None:
        self.cfg = load_config()
        data_cfg = self.cfg['data']
        kite_cfg = self.cfg['kite']

        self.lookback = data_cfg['lookback']
        self.feature_engineer = IntradayFeatureEngineer(self.lookback)
        self.strategy = IntradayStrategy()

        self.kite = get_kite_client(kite_cfg['api_key'], kite_cfg['access_token'])
        self.instrument_token_manager = InstrumentTokenManager()
        if self.instrument_token_manager.get_instrument_tokens() is None:
            self.instrument_token_manager.set_instrument_tokens()
        self.fetcher = MarketDataFetcher(
            kite_client=self.kite,
            interval=data_cfg.get('interval', '15minute'),
            instrument_tokens=self.instrument_token_manager.get_instrument_tokens(),
        )

    def train_model(
        self,
        kite_instrument: str,
        from_date: str,
        to_date: str,
        test_split: float = 0.2,
    ) -> Tuple[LSTMReturnForecaster, object, Dict[str, float]]:
        raw_df = self.fetcher.fetch(kite_instrument, from_date, to_date)
        feature_frame = self.feature_engineer.prepare_training_frame(raw_df)
        scaled_features, scaler = self.feature_engineer.scale_features(feature_frame)
        X, y, meta = self.feature_engineer.build_supervised_dataset(feature_frame, scaled_features)

        split_idx = int(len(X) * (1 - test_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        meta_test = meta.iloc[split_idx:].reset_index(drop=True)

        if len(X_test) == 0:
            raise ValueError(
                "Test set is empty. Increase the data window or adjust the test_split proportion."
            )

        model = LSTMReturnForecaster(
            input_shape=(self.lookback, len(self.feature_engineer.FEATURE_COLUMNS))
        )
        model.fit(
            X_train,
            y_train,
            epochs=self.cfg['training']['epochs'],
            batch_size=self.cfg['training']['batch_size'],
            validation_split=0.1,
            verbose=1,
        )

        predicted_returns = model.predict(X_test).flatten()
        actual_returns = y_test.flatten()
        predicted_close = meta_test['close'].values * (1 + predicted_returns)
        actual_close = meta_test['future_close'].values

        rmse = float(np.sqrt(mean_squared_error(actual_close, predicted_close)))
        mae = float(mean_absolute_error(actual_close, predicted_close))
        r2 = float(r2_score(actual_close, predicted_close))
        directional_accuracy = float(
            (np.sign(predicted_returns) == np.sign(actual_returns)).mean() * 100
        )

        strategy_frame = self._build_strategy_frame(meta_test, predicted_returns, predicted_close)
        strategy_accuracy = float(self.strategy.evaluate(strategy_frame) * 100)

        print(
            f"\n✅ Test Metrics for {kite_instrument}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}, "
            f"Directional Accuracy={directional_accuracy:.2f}%"
        )
        print(f"✅ Strategy accuracy on actionable signals: {strategy_accuracy:.2f}%")

        train_cfg = self.cfg['training']
        model_path = os.path.join(train_cfg['model_save_path'], f"{kite_instrument}_return_lstm.keras")
        scaler_path = os.path.join(train_cfg['scaler_save_path'], f"{kite_instrument}_return.pkl")
        save_model(model.model, model_path)
        save_scaler(scaler, scaler_path)
        print(f"\n✅ Model saved to {model_path}")
        print(f"✅ Scaler saved to {scaler_path}")

        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'strategy_accuracy': strategy_accuracy,
        }

        return model, scaler, metrics

    def _build_strategy_frame(
        self,
        meta: pd.DataFrame,
        predicted_returns: np.ndarray,
        predicted_close: np.ndarray,
    ) -> pd.DataFrame:
        frame = meta.copy()
        frame['RSI'] = frame['rsi']
        frame['Close'] = frame['close']
        frame['EMA_200'] = frame['ema200']
        frame['EMA_50'] = frame['ema50']
        frame['EMA_20'] = frame['ema20']
        frame['prev_day_touch_EMA20'] = frame['prev_day_touch_ema20']
        frame['prev_day_touch_EMA50'] = frame['prev_day_touch_ema50']
        frame['prev_day_Close'] = frame['prev_day_close']
        frame['prev_day_Open'] = frame['prev_day_open']
        frame['prev_day_Low'] = frame['prev_day_low']
        frame['5_day_min_of_close'] = frame['min_5_day_close']
        frame['Avg_2_days_Volume'] = frame['avg_2_days_volume']
        frame['Avg_10_days_Volume'] = frame['avg_10_days_volume']
        frame['Signal'] = (predicted_returns > 0).astype(int)
        frame['predicted_return'] = predicted_returns
        frame['predicted_close'] = predicted_close
        return frame
