from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


def build_lstm_model(input_shape, output_dim: int = 1) -> Sequential:
    """Legacy helper kept for backward compatibility."""
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))
    model.add(Dense(units=output_dim))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


@dataclass
class LSTMReturnForecaster:
    """Thin OO wrapper around the LSTM model used for return forecasting."""

    input_shape: tuple
    output_dim: int = 1
    model: Optional[Sequential] = None

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = build_lstm_model(self.input_shape, self.output_dim)

    def fit(self, X_train, y_train, **kwargs):
        return self.model.fit(X_train, y_train, **kwargs)

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def save(self, path: str) -> None:
        self.model.save(path)

    def summary(self) -> None:
        self.model.summary()
