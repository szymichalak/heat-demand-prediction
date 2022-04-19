from typing import Tuple

from data.split import DataSplit
from prediction.abstractPrediction import AbstractPrediction


class ArmaPrediction(AbstractPrediction):
    def __init__(self, data: DataSplit, order: Tuple):
        super().__init__(data, order=order)
        self.__order = order
        self.__checkOrderValidity()

    def __checkOrderValidity(self):
        if self.__order[1] != 0:
            raise ValueError('Passed wrong order parameter')
        if self.__order[0] * self.__order[2] == 0:
            raise ValueError('Passed wrong order parameter')
