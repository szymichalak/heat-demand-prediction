import pandas as pd

from typing import Tuple


class DataSplit:
    def __init__(self, data: Tuple[pd.DataFrame, pd.DataFrame]):
        (self.training, self.testing) = data
