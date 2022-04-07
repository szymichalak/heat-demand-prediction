import pandas as pd
from typing import Tuple

from prediction.abstractPrediction import AbstractPrediction


class SarimaxPrediction(AbstractPrediction):
    def __init__(self, data: pd.DataFrame, order: Tuple, seasonalOrder: Tuple):
        super().__init__(data, order=order, seasonalOrder=seasonalOrder)
        self.__order = order
        self.__seasonalOrder = seasonalOrder
        self.__checkOrderValidity()

    def __checkOrderValidity(self):
        if self.__order[0] * self.__order[1] * self.__order[2] == 0:
            raise ValueError('Passed wrong order parameter')
