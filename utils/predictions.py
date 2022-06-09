from typing import List

import pandas as pd

from utils.const import HORIZON


class Predictions:
    def __init__(self, pred_horizon: pd.DataFrame):
        self.pred_horizon: pd.DataFrame = pred_horizon

    def returnHorizon(self) -> List[str]:
        return self.pred_horizon.columns

    def shift(self, shift=0) -> pd.Series:
        if shift >= HORIZON:
            raise ValueError('Shift must be smaller than HORIZON value')
        return self.pred_horizon.iloc[:, shift]
