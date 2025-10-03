from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Decision = Literal['BUY', 'SELL', 'HOLD']


@dataclass
class IntradayStrategy:
    """Encapsulates the user supplied intraday rule-set."""

    min_predicted_return: float = 0.003
    min_buy_rsi: float = 52.0
    max_sell_rsi: float = 45.0
    require_ema_alignment: bool = True
    min_volume_ratio: float = 1.0

    def _volume_ratio(self, row: pd.Series) -> float:
        avg_10 = row.get('Avg_10_days_Volume', 0.0)
        if avg_10 <= 0:
            return np.inf
        return row.get('Avg_2_days_Volume', 0.0) / avg_10

    def _is_bullish_trend(self, row: pd.Series) -> bool:
        ema20 = row.get('EMA_20', 0.0)
        ema50 = row.get('EMA_50', 0.0)
        ema200 = row.get('EMA_200', 0.0)
        close = row.get('Close', 0.0)
        if not self.require_ema_alignment:
            return close > ema200
        return close > ema200 and ema20 >= ema50 >= ema200

    def _is_bearish_trend(self, row: pd.Series) -> bool:
        ema20 = row.get('EMA_20', 0.0)
        ema50 = row.get('EMA_50', 0.0)
        ema200 = row.get('EMA_200', 0.0)
        close = row.get('Close', 0.0)
        if not self.require_ema_alignment:
            return close < ema50
        return close < ema50 and ema20 <= ema50 <= ema200

    def _passes_buy_filters(self, row: pd.Series) -> bool:
        if row.get('RSI', 0.0) < self.min_buy_rsi:
            return False
        if row.get('divergence', 0) == -1:
            return False
        if not bool(row.get('prev_day_touch_EMA20', False)):
            return False
        if not bool(row.get('prev_day_touch_EMA50', False)):
            return False
        if row.get('Close', 0.0) <= row.get('prev_day_Close', -np.inf):
            return False
        if row.get('Close', 0.0) <= row.get('prev_day_Open', -np.inf):
            return False
        if row.get('5_day_min_of_close', 0.0) <= row.get('EMA_50', 0.0):
            return False
        if self._volume_ratio(row) < self.min_volume_ratio:
            return False
        return self._is_bullish_trend(row)

    def _passes_sell_filters(self, row: pd.Series) -> bool:
        if row.get('RSI', 100.0) > self.max_sell_rsi:
            return False
        if self._volume_ratio(row) < self.min_volume_ratio:
            return False
        return self._is_bearish_trend(row)

    def _generate_signal(self, row: pd.Series) -> Decision:
        predicted_return = float(row.get('predicted_return', 0.0))
        if predicted_return >= self.min_predicted_return and row.get('Signal', 1) == 1:
            if self._passes_buy_filters(row):
                return 'BUY'
            return 'HOLD'

        if predicted_return <= -self.min_predicted_return:
            if self._passes_sell_filters(row):
                return 'SELL'
            return 'HOLD'

        return 'HOLD'

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result['decision'] = result.apply(self._generate_signal, axis=1)
        return result

    def evaluate(self, df: pd.DataFrame) -> float:
        evaluated = self.apply(df)
        actionable = evaluated[evaluated['decision'] != 'HOLD']
        if actionable.empty:
            return 0.0

        def _is_correct(row: pd.Series) -> bool:
            future_close = row.get('future_close')
            if pd.isna(future_close):
                return False
            if row['decision'] == 'BUY':
                return future_close > row.get('Close', future_close)
            if row['decision'] == 'SELL':
                return future_close < row.get('Close', future_close)
            return False

        return float(actionable.apply(_is_correct, axis=1).mean())
