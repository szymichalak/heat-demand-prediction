import pandas as pd
from typing import Tuple

from prediction.abstractPrediction import AbstractPrediction


class ArmaPrediction(AbstractPrediction):
    def __init__(self, data: pd.DataFrame, order: Tuple):
        super().__init__(data, order=order)
        self.__train = self.getTrainingData()
        self.__test = self.getTestingData()
