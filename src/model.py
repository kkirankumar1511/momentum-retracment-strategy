import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

def build_lstm_model(input_shape, output_dim=1):
    """
    3 LSTM layers (stacked), returns a model that outputs `predict_horizon` values (close prices).
    input_shape: (lookback, n_features)
    """
    model = Sequential()

    # Layer 1
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    # Layer 2
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))

    # Layer 3
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))

    # Output Layer
    model.add(Dense(units=output_dim))

    # Compile the model
    # 'adam' is a popular and effective optimizer
    # 'mean_squared_error' is a standard loss function for regression problems
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model
