import os

from src.utils import get_instrument_token

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import load_config
from src.training_agent import instrument_token_manager
from training_agent import TrainingAgent
from predict_live import PredictAgent
from data_loader import csv_loader, get_kite_client, load_from_kite
from instrument_token import InstrumentTokenManager

if __name__ == "__main__":
    cfg = load_config()

    # Setup
    instrument_token_manager = InstrumentTokenManager()
    instrument_token_manager.set_instrument_tokens()
    training_agent = TrainingAgent()
    predict_agent = PredictAgent()

    instrumentDf = csv_loader(cfg['data']['local_csv'])
    kite_cfg = cfg['kite']
    kite = get_kite_client(kite_cfg['api_key'], kite_cfg['access_token'])
    all_tokens = instrument_token_manager.get_instrument_tokens()

    for instrument_name in instrumentDf['Symbol']:
        print(f"\n=== Processing {instrument_name} ===")
        model_path = os.path.join(cfg['training']['model_save_path'], f"{instrument_name}_lstm.keras")
        scaler_path = os.path.join(cfg['training']['scaler_save_path'], f"{instrument_name}.pkl")

        instrument_token = get_instrument_token(all_tokens,instrument_name)

        # Fetch historical data for lookback and prediction
        from_date = '2025-05-15'
        to_date = '2025-09-15'
        last_df = load_from_kite(kite, instrument_token, from_date, to_date, interval="day")

        last_close = last_df['close'].iloc[-1]

        # Train model if missing or retrain is requested
        if (not os.path.exists(model_path) or not os.path.exists(scaler_path)) or cfg['training']['re-train']:
            print(f"Training model for {instrument_name}...")
            training_agent.train_model(instrument_name, '2020-06-01', '2025-09-01')

        # Predict next closes
        df_predicted = predict_agent.predict_next_close(
            kite_instrument=instrument_name,
            last_close=last_close,
            last_df=last_df,
            horizon=5
        )

        print(f"\nPredicted Close Prices for {instrument_name}:")
        print(df_predicted)
