import pandas as pd
from typing import Tuple

from prediction.abstractPrediction import AbstractPrediction


class SarimaxPrediction(AbstractPrediction):
    def __init__(self, data: pd.DataFrame, order: Tuple, seasonalOrder: Tuple):
        super().__init__(data, order=order, seasonalOrder=seasonalOrder)
        self.__train = self.getTrainingData()
        self.__test = self.getTestingData()
