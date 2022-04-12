import pandas as pd
from typing import Tuple

from prediction.abstractPrediction import AbstractPrediction


class ArmaPrediction(AbstractPrediction):
    def __init__(self, data: pd.DataFrame, order: Tuple):
        super().__init__(data, False, order=order)
        self.__order = order
        self.__checkOrderValidity()

    def __checkOrderValidity(self):
        if self.__order[1] != 0:
            raise ValueError('Passed wrong order parameter')
        if self.__order[0] * self.__order[2] == 0:
            raise ValueError('Passed wrong order parameter')
