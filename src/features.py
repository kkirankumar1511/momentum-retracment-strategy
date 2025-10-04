import numpy as np
import pandas as pd
import talib


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append core technical indicators used across the pipeline."""
    df = df.copy()
    if 'date' in df.columns:
        df = df.sort_values('date')
    close = df['close']

    # Core momentum/trend indicators
    df['rsi'] = talib.RSI(close, timeperiod=14)
    df['ema20'] = talib.EMA(close, timeperiod=20)
    df['ema50'] = talib.EMA(close, timeperiod=50)
    df['ema200'] = talib.EMA(close, timeperiod=200)

    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macdsignal'] = macdsignal

    upper, middle, lower = talib.BBANDS(close, timeperiod=20)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower

    # TALIB requires long lookbacks for several indicators.  On short samples it
    # can yield entirely empty columns which would later be dropped, leaving zero
    # training rows.  Provide lightweight pandas fallbacks in that situation so
    # feature engineering still succeeds.
    if df['rsi'].isna().all():
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14.0, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14.0, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

    for span, column in ((20, 'ema20'), (50, 'ema50'), (200, 'ema200')):
        if df[column].isna().all():
            df[column] = close.ewm(span=span, adjust=False).mean()

    if df['macd'].isna().all() or df['macdsignal'].isna().all():
        fast = close.ewm(span=12, adjust=False).mean()
        slow = close.ewm(span=26, adjust=False).mean()
        fallback_macd = fast - slow
        df['macd'] = fallback_macd
        df['macdsignal'] = fallback_macd.ewm(span=9, adjust=False).mean()

    df.bfill(inplace=True)
    df.ffill(inplace=True)
    return df


def add_intraday_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature block tailored for the intraday strategy rules."""
    if 'date' not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column for feature engineering.")

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    df = add_technical_indicators(df)
    return df
