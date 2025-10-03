import math
import os
from datetime import timedelta

import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import load_config
from data_loader import csv_loader
from instrument_token import InstrumentTokenManager
from predict_live import PredictAgent
from training_agent import TrainingAgent


if __name__ == "__main__":
    cfg = load_config()
    data_cfg = cfg['data']

    instrument_manager = InstrumentTokenManager()
    instrument_manager.set_instrument_tokens()

    training_agent = TrainingAgent()
    predict_agent = PredictAgent()

    instruments_df = csv_loader(data_cfg['local_csv'])

    interval = str(data_cfg.get('interval', '15minute'))
    try:
        interval_minutes = int(''.join(filter(str.isdigit, interval)))
    except ValueError as exc:
        raise ValueError(
            f"Invalid interval format '{interval}'. Expected minutes like '15minute'."
        ) from exc
    if interval_minutes <= 0:
        raise ValueError(
            f"Invalid interval minutes derived from '{interval}'. Must be positive."
        )

    trading_minutes = 375  # 09:15 to 15:30 NSE trading session
    bars_per_day = max(1, trading_minutes // interval_minutes)

    inference_days = int(data_cfg.get('inference_window_days', 5))
    lookback_bars = int(data_cfg.get('lookback', 0))
    required_days = math.ceil(max(lookback_bars, 200) / bars_per_day)
    inference_days = max(inference_days, required_days)
    train_from = data_cfg.get('train_from', '2024-09-01 09:15:00')
    train_to = data_cfg.get('train_to', '2025-03-01 15:30:00')

    for instrument_name in instruments_df['Symbol']:
        print(f"\n=== Processing {instrument_name} ===")
        try:
            training_agent.train_model(
                kite_instrument=instrument_name,
                from_date=train_from,
                to_date=train_to,
            )
        except ValueError as exc:
            print(f"⚠️ Skipping {instrument_name}: {exc}")
            continue

        inference_start = (
            pd.to_datetime(train_to) - timedelta(days=inference_days)
        ).strftime('%Y-%m-%d')
        signal_df = predict_agent.predict_next_interval(
            instrument_name=instrument_name,
            from_date=inference_start,
            to_date=train_to,
        )
        print(signal_df)
