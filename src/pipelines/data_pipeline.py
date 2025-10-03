from dataclasses import dataclass
from datetime import timedelta
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
    max_days_per_request: int = 200

    def __post_init__(self) -> None:
        if self.instrument_tokens is None:
            self.instrument_tokens = InstrumentTokenManager().get_instrument_tokens()

    def fetch(self, instrument: str, from_date: str, to_date: str) -> pd.DataFrame:
        """Fetch market data between ``from_date`` and ``to_date`` (inclusive).

        The inputs can be simple ``YYYY-MM-DD`` strings or full ISO datetime
        strings; any time component is ignored so the request still spans whole
        trading days.
        """
        instrument_token = get_instrument_token(self.instrument_tokens, instrument)

        start = pd.to_datetime(from_date, errors="raise").normalize()
        end = pd.to_datetime(to_date, errors="raise").normalize()
        if start > end:
            raise ValueError("from_date must be earlier than to_date")

        frames = []
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + timedelta(days=self.max_days_per_request - 1), end)
            chunk = load_from_kite(
                self.kite_client,
                instrument_token,
                cursor.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
                interval=self.interval,
            )
            if not chunk.empty:
                frames.append(chunk)
            cursor = chunk_end + timedelta(days=1)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        return df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
