import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import load_config
from data_loader import get_kite_client
from instrument_token import InstrumentTokenManager
from model import LSTMReturnForecaster
from pipelines.data_pipeline import MarketDataFetcher
from pipelines.feature_pipeline import IntradayFeatureEngineer
from utils import save_model, save_scaler


@dataclass
class TrainingAgent:
    """Coordinates data loading, feature prep, model training and evaluation."""

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
            input_shape=(self.lookback, len(self.feature_engineer.PREDICTOR_COLUMNS)),
            output_dim=self.feature_engineer.horizon,
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

        predicted_close = model.predict(X_test)
        actual_close = y_test

        rmse_per_step = np.sqrt(((predicted_close - actual_close) ** 2).mean(axis=0))
        mae_per_step = np.abs(predicted_close - actual_close).mean(axis=0)

        overall_rmse = float(rmse_per_step.mean())
        overall_mae = float(mae_per_step.mean())

        r2_scores = []
        sample_count = actual_close.shape[0]
        if sample_count > 1:
            for step in range(self.feature_engineer.horizon):
                r2_scores.append(
                    r2_score(
                        actual_close[:, step],
                        predicted_close[:, step],
                    )
                )
        else:
            r2_scores = [float('nan')] * self.feature_engineer.horizon
        overall_r2 = float(np.nanmean(r2_scores))

        current_close = meta_test['close'].to_numpy().reshape(-1, 1)
        actual_direction = np.sign(actual_close - current_close)
        predicted_direction = np.sign(predicted_close - current_close)
        directional_accuracy_per_step = (
            (predicted_direction == actual_direction).mean(axis=0) * 100
        )
        directional_accuracy = float(directional_accuracy_per_step.mean())

        print(
            f"\n✅ Test Metrics for {kite_instrument}: "
            f"Avg RMSE={overall_rmse:.2f}, Avg MAE={overall_mae:.2f}, Avg R²={overall_r2:.3f}, "
            f"Directional Accuracy={directional_accuracy:.2f}%"
        )

        horizon_summary = " | ".join(
            [
                f"t+{idx+1}: RMSE={rmse_per_step[idx]:.2f}, MAE={mae_per_step[idx]:.2f}, "
                f"DA={directional_accuracy_per_step[idx]:.2f}%"
                for idx in range(self.feature_engineer.horizon)
            ]
        )
        print(f"ℹ️ Horizon breakdown -> {horizon_summary}")

        target_accuracy = 80.0
        if directional_accuracy >= target_accuracy:
            print(
                f"✅ Forecast directional accuracy meets the ~80% goal "
                f"({directional_accuracy:.2f}%)."
            )
        else:
            print(
                f"⚠️ Forecast directional accuracy {directional_accuracy:.2f}% "
                "is below the ~80% goal. Consider retraining with more data or tuning hyperparameters."
            )

        model_path = os.path.join(train_cfg['model_save_path'], f"{kite_instrument}_return_lstm.keras")
        scaler_path = os.path.join(train_cfg['scaler_save_path'], f"{kite_instrument}_return.pkl")
        save_model(model.model, model_path)
        save_scaler(scaler, scaler_path)
        print(f"\n✅ Model saved to {model_path}")
        print(f"✅ Scaler saved to {scaler_path}")

        metrics = {
            'rmse_avg': overall_rmse,
            'mae_avg': overall_mae,
            'r2_avg': overall_r2,
            'directional_accuracy': directional_accuracy,
            'rmse_per_step': rmse_per_step.tolist(),
            'mae_per_step': mae_per_step.tolist(),
            'directional_accuracy_per_step': directional_accuracy_per_step.tolist(),
        }

        return model, scaler, metrics
