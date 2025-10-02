import pandas as pd
from kiteconnect import KiteConnect

from features import add_technical_indicators

def csv_loader(path):
    df = pd.read_csv(path)
    return df

def load_from_csv(path: str):
    df = pd.read_csv(path, parse_dates=['date'])
    # normalize column names to lower
    df.columns = [c.lower() for c in df.columns]
    # ensure at least open,high,low,close exist
    for c in ['open','high','low','close']:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")
    df = add_technical_indicators(df)
    return df

def load_from_kite(kite: KiteConnect, instrument_token: int, from_date: str, to_date: str, interval: str = "day"):
    """
    from_date and to_date in 'YYYY-MM-DD' strings
    """
   # print(kite.profile())
    data = kite.historical_data(instrument_token, from_date, to_date, interval)
    df = pd.DataFrame(data)
    # API returns 'date','open','high','low','close','volume'
    df['date'] = pd.to_datetime(df['date'])
   # df = add_technical_indicators(df)
    return df

def get_kite_client(api_key, access_token):
    # 1. Create the checksum
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite

def get_instrument_tokens(kite: KiteConnect):
    instruments = kite.instruments()
    df = pd.DataFrame(instruments)
    return df

