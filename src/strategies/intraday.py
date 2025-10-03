from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Decision = Literal['BUY', 'SELL', 'HOLD']


@dataclass
class IntradayStrategy:
    """Encapsulates the user supplied intraday rule-set."""

    def _generate_signal(self, row: pd.Series) -> Decision:
        condition1 = row.get('RSI', 0) > 50
        condition2 = row.get('Close', 0) > row.get('EMA_200', 0)
        condition3 = bool(row.get('prev_day_touch_EMA20', False))
        condition6 = bool(row.get('prev_day_touch_EMA50', False))
        condition4 = (
            row.get('Close', 0) > row.get('prev_day_Close', -np.inf)
            and row.get('Close', 0) > row.get('prev_day_Open', -np.inf)
        )
        condition5 = row.get('divergence', 0) != -1
        condition7 = row.get('5_day_min_of_close', 0) > row.get('EMA_50', 0)
        is_signal_predict = row.get('Signal', 1) == 1
        is_avg_vol_grt_curr_vol = (
            row.get('Avg_2_days_Volume', 0) >= row.get('Avg_10_days_Volume', 0)
        )

        if all(
            [
                condition1,
                condition2,
                condition3,
                condition4,
                condition5,
                condition6,
                condition7,
                is_signal_predict,
                is_avg_vol_grt_curr_vol,
            ]
        ):
            return 'BUY'

        if row.get('Close', 0) < row.get('EMA_50', float('inf')) and row.get('RSI', 100) < 40:
            return 'SELL'

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
