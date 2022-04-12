import pandas as pd
from typing import Tuple

from prediction.abstractPrediction import AbstractPrediction


class SarimaPrediction(AbstractPrediction):
    def __init__(self, data: pd.DataFrame, seasonalOrder: Tuple):
        super().__init__(data, True, seasonalOrder=seasonalOrder)
        self.__seasonalOrder = seasonalOrder
        self.__checkOrderValidity()

    def __checkOrderValidity(self):
        if self.__seasonalOrder[0] * self.__seasonalOrder[1] * self.__seasonalOrder[2] * self.__seasonalOrder[3] == 0:
            raise ValueError('Passed wrong order parameter')
