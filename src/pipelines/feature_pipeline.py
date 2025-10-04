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

    @property
    def target_columns(self) -> list[str]:
        return [f'future_close_{step}' for step in range(1, self.horizon + 1)]

    @property
    def return_columns(self) -> list[str]:
        return [f'target_return_{step}' for step in range(1, self.horizon + 1)]

    def _annotate_targets(self, frame: pd.DataFrame) -> pd.DataFrame:
        for step in range(1, self.horizon + 1):
            future_close_col = f'future_close_{step}'
            target_return_col = f'target_return_{step}'
            frame[future_close_col] = frame['close'].shift(-step)
            frame[target_return_col] = (
                frame[future_close_col] - frame['close']
            ) / frame['close']
        horizon_close_col = f'future_close_{self.horizon}'
        horizon_return_col = f'target_return_{self.horizon}'
        frame['future_close'] = frame[horizon_close_col]
        frame['target_return'] = frame[horizon_return_col]
        return frame

    def prepare_training_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = add_intraday_strategy_features(df)
        frame = self._annotate_targets(frame)
        required_columns = self.PREDICTOR_COLUMNS + self.target_columns
        frame = frame.dropna(subset=required_columns).reset_index(drop=True)
        return frame

    def prepare_inference_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = add_intraday_strategy_features(df)
        frame = frame.dropna(subset=self.PREDICTOR_COLUMNS).reset_index(drop=True)
        frame = self._annotate_targets(frame)
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
        target_cols = self.target_columns
        for idx in range(self.lookback, len(frame)):
            sequences.append(scaled_features[idx - self.lookback: idx])
            targets.append(frame[target_cols].iloc[idx].to_numpy())
            meta_rows.append(frame.iloc[idx])

        X = np.array(sequences)
        y = np.array(targets)
        meta = pd.DataFrame(meta_rows).reset_index(drop=True)
        return X, y, meta

    def build_prediction_sequence(self, scaled_features: np.ndarray) -> np.ndarray:
        if len(scaled_features) < self.lookback:
            raise ValueError("Insufficient history to build the prediction sequence.")
        seq = scaled_features[-self.lookback:]
        return seq.reshape(1, self.lookback, scaled_features.shape[1])
