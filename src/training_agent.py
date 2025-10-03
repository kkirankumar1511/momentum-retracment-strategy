import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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

    def train_model(
        self,
        kite_instrument: str,
        from_date: str,
        to_date: str,
        test_split: float = 0.2,
    ) -> Tuple[LSTMReturnForecaster, object, Dict[str, float]]:
        raw_df = self.fetcher.fetch(kite_instrument, from_date, to_date)
        feature_frame = self.feature_engineer.prepare_training_frame(raw_df)
        resolved_threshold = self._calibrate_min_predicted_return(feature_frame)
        print(
            f"ℹ️ Using min_predicted_return threshold of {resolved_threshold:.5f} "
            "based on configured strategy settings."
        )
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
            input_shape=(self.lookback, len(self.feature_engineer.PREDICTOR_COLUMNS))
        )
        train_cfg = self.cfg['training']
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=int(train_cfg.get('early_stopping_patience', 5)),
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                patience=int(train_cfg.get('reduce_lr_patience', 3)),
                factor=0.5,
                min_lr=float(train_cfg.get('min_learning_rate', 1e-5)),
            ),
        ]
        model.fit(
            X_train,
            y_train,
            epochs=train_cfg['epochs'],
            batch_size=train_cfg['batch_size'],
            validation_split=train_cfg.get('validation_split', 0.1),
            verbose=1,
            callbacks=callbacks,
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
        strategy_accuracy_raw = self.strategy.evaluate(strategy_frame)
        if np.isnan(strategy_accuracy_raw):
            print("ℹ️ Strategy generated no actionable trades on the held-out set.")
            strategy_accuracy = float('nan')
        else:
            strategy_accuracy = float(strategy_accuracy_raw * 100)

        print(
            f"\n✅ Test Metrics for {kite_instrument}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}, "
            f"Directional Accuracy={directional_accuracy:.2f}%"
        )
        if np.isnan(strategy_accuracy):
            print("ℹ️ Strategy accuracy on actionable signals: not available (no trades).")
        else:
            print(f"✅ Strategy accuracy on actionable signals: {strategy_accuracy:.2f}%")

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
            'resolved_min_predicted_return': resolved_threshold,
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
        frame['ATR'] = frame['atr']
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
        frame['projected_move'] = frame['predicted_close'] - frame['Close']
        frame['stop_loss_buy'] = frame['Close'] - frame['ATR'] * self.strategy.atr_multiple
        frame['stop_loss_sell'] = frame['Close'] + frame['ATR'] * self.strategy.atr_multiple
        return frame

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
            self.strategy.min_predicted_return = float(configured)
            resolved = self.strategy.min_predicted_return
        return resolved
