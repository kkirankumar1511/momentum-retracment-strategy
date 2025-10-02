import pandas as pd
import talib


def add_technical_indicators(df):
    df = df.copy()
    # RSI
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    # EMA
    df['ema20'] = talib.EMA(df['close'], timeperiod=20)
    df['ema50'] = talib.EMA(df['close'], timeperiod=50)
    #df['ema200'] = talib.EMA(df['close'], timeperiod=200)
    # MACD
    macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macdsignal'] = macdsignal
    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    # Fill NaNs
    df.bfill(inplace=True)
    return df
