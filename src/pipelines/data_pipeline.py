from dataclasses import dataclass
from typing import Optional

import pandas as pd

from data_loader import load_from_kite
from instrument_token import InstrumentTokenManager
from utils import get_instrument_token


@dataclass
class MarketDataFetcher:
    """Thin wrapper that retrieves market data for a given instrument/interval."""

    kite_client: object
    interval: str = "15minute"
    instrument_tokens: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        if self.instrument_tokens is None:
            self.instrument_tokens = InstrumentTokenManager().get_instrument_tokens()

    def fetch(self, instrument: str, from_date: str, to_date: str) -> pd.DataFrame:
        instrument_token = get_instrument_token(self.instrument_tokens, instrument)
        df = load_from_kite(
            self.kite_client,
            instrument_token,
            from_date,
            to_date,
            interval=self.interval,
        )
        return df.sort_values('date').reset_index(drop=True)
