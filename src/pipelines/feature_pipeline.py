from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from features import add_intraday_strategy_features


@dataclass
class IntradayFeatureEngineer:
    lookback: int
    horizon: int = 1

    def __post_init__(self) -> None:
        if self.lookback <= 0:
            raise ValueError("lookback must be a positive integer.")
        if self.horizon <= 0:
            raise ValueError("predict_horizon must be a positive integer.")
        self.horizon = int(self.horizon)

    # Columns used strictly as model inputs when forecasting the next close.
    PREDICTOR_COLUMNS = [
        'open',
        'high',
        'low',
        'close',
        'volume',
        'rsi',
        'ema20',
        'ema50',
        'ema200',
        'macd',
        'macdsignal',
        'bb_upper',
        'bb_middle',
        'bb_lower',
    ]

    def prepare_training_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = add_intraday_strategy_features(df)
        frame['future_close'] = frame['close'].shift(-self.horizon)
        frame['target_return'] = (frame['future_close'] - frame['close']) / frame['close']
        frame = frame.dropna(subset=self.PREDICTOR_COLUMNS + ['target_return']).reset_index(drop=True)
        return frame

    def prepare_inference_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = add_intraday_strategy_features(df)
        frame = frame.dropna(subset=self.PREDICTOR_COLUMNS).reset_index(drop=True)
        frame['future_close'] = frame['close'].shift(-self.horizon)
        frame['target_return'] = (frame['future_close'] - frame['close']) / frame['close']
        return frame

    def scale_features(
        self,
        frame: pd.DataFrame,
        scaler: Optional[MinMaxScaler] = None,
    ) -> Tuple[np.ndarray, MinMaxScaler]:
        features = frame[self.PREDICTOR_COLUMNS].copy()
        for col in features.columns:
            if features[col].dtype == bool:
                features[col] = features[col].astype(int)
        if scaler is None:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(features.values)
        else:
            scaled = scaler.transform(features.values)
        return scaled, scaler

    def build_supervised_dataset(
        self,
        frame: pd.DataFrame,
        scaled_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        if len(frame) <= self.lookback:
            raise ValueError("Not enough rows to create sequences with the configured lookback.")

        sequences, targets, meta_rows = [], [], []
        for idx in range(self.lookback, len(frame)):
            sequences.append(scaled_features[idx - self.lookback: idx])
            targets.append(frame['target_return'].iloc[idx])
            meta_rows.append(frame.iloc[idx])

        X = np.array(sequences)
        y = np.array(targets).reshape(-1, 1)
        meta = pd.DataFrame(meta_rows).reset_index(drop=True)
        return X, y, meta

    def build_prediction_sequence(self, scaled_features: np.ndarray) -> np.ndarray:
        if len(scaled_features) < self.lookback:
            raise ValueError("Insufficient history to build the prediction sequence.")
        seq = scaled_features[-self.lookback:]
        return seq.reshape(1, self.lookback, scaled_features.shape[1])
