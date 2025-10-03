from config import load_config
from data_loader import get_kite_client, get_instrument_tokens

class InstrumentTokenManager:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InstrumentTokenManager, cls).__new__(cls)
            cls._instance.instrument_list = None  # Initialize the global variable
        return cls._instance

    def set_instrument_tokens(self):
        if self._instance.instrument_list is None:
            kite_cfg = load_config()
            kite = get_kite_client(kite_cfg['kite']['api_key'], kite_cfg['kite']['access_token'])
            self._instance.instrument_list = get_instrument_tokens(kite)

    def get_instrument_tokens(self):
        return self._instance.instrument_list

