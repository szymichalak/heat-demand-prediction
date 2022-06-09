import pandas as pd

from utils.const import HORIZON


class GroundTruth:
    def __init__(self, gt: pd.DataFrame):
        self.gt: pd.DataFrame = gt

    def shift(self, shift=0) -> pd.DataFrame:
        if shift >= HORIZON:
            raise ValueError('Shift must be smaller than HORIZON value')
        return self.gt[shift:-(HORIZON-shift)]
