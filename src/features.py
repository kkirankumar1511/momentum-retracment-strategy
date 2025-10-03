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
    df['session_date'] = df['date'].dt.normalize()

    daily = (
        df.groupby('session_date')
        .agg(
            daily_low=('low', 'min'),
            daily_close=('close', 'last'),
            daily_open=('open', 'first'),
            daily_volume=('volume', 'sum')
        )
        .sort_index()
    )

    daily['prev_day_low'] = daily['daily_low'].shift(1)
    daily['prev_day_close'] = daily['daily_close'].shift(1)
    daily['prev_day_open'] = daily['daily_open'].shift(1)
    daily['prev_ema_20'] = daily['daily_close'].shift(1).rolling(window=20).mean()
    daily['prev_ema_50'] = daily['daily_close'].shift(1).rolling(window=50).mean()

    daily['prev_day_touch_ema20'] = (
        (daily['prev_day_close'] > daily['prev_ema_20'])
        & (daily['prev_day_low'] < daily['prev_ema_20'])
    )
    daily['prev_day_touch_ema50'] = (
        (daily['prev_day_close'] > daily['prev_ema_50'])
        & (daily['prev_day_low'] < daily['prev_ema_50'])
    )

    daily['min_5_day_close'] = daily['daily_close'].rolling(window=5).min()
    daily['avg_2_days_volume'] = daily['daily_volume'].rolling(window=2).mean()
    daily['avg_10_days_volume'] = daily['daily_volume'].rolling(window=10).mean()

    feature_cols = [
        'prev_day_low',
        'prev_day_close',
        'prev_day_open',
        'prev_ema_20',
        'prev_ema_50',
        'prev_day_touch_ema20',
        'prev_day_touch_ema50',
        'min_5_day_close',
        'avg_2_days_volume',
        'avg_10_days_volume',
    ]

    df = df.merge(daily[feature_cols], left_on='session_date', right_index=True, how='left')

    # Divergence heuristic: RSI disagreeing with price direction
    price_change = df['close'].diff()
    rsi_change = df['rsi'].diff()
    divergence = np.where(
        (price_change > 0) & (rsi_change < 0),
        -1,
        np.where((price_change < 0) & (rsi_change > 0), -1, 1)
    )
    df['divergence'] = divergence

    bool_cols = ['prev_day_touch_ema20', 'prev_day_touch_ema50']
    for col in bool_cols:
        df[col] = df[col].fillna(False).astype(bool)

    df[['avg_2_days_volume', 'avg_10_days_volume', 'min_5_day_close', 'prev_ema_20', 'prev_ema_50']] = (
        df[['avg_2_days_volume', 'avg_10_days_volume', 'min_5_day_close', 'prev_ema_20', 'prev_ema_50']]
        .fillna(0.0)
    )
    df[['prev_day_low', 'prev_day_close', 'prev_day_open']] = (
        df[['prev_day_low', 'prev_day_close', 'prev_day_open']].fillna(method='ffill').fillna(method='bfill')
    )

    df.drop(columns=['session_date'], inplace=True)
    return df
