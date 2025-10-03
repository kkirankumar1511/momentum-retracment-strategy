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
    inference_days = int(data_cfg.get('inference_window_days', 5))
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
