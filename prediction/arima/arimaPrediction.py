import time
import pandas as pd

from typing import Tuple
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from customSelectors.columns import Energy, exog
from data.split import DataSplit


class ArimaPrediction:
    def __init__(self, data: DataSplit, order: Tuple = (0, 0, 0), seasonalOrder: Tuple = (0, 0, 0, 0)):
        self.__data = data
        self.__order = order
        self.__seasonalOrder = seasonalOrder
        (self.__trainingData, self.__testingData) = (data.training, data.testing)
        self.__fittingTime: float = 0

    def getTrainingData(self) -> pd.DataFrame:
        return self.__trainingData

    def getTestingData(self) -> pd.DataFrame:
        return self.__testingData

    def __fitModel(self) -> ARIMAResults:
        model = ARIMA(
            self.__trainingData[Energy],
            exog=self.__trainingData[exog],
            order=self.__order,
            seasonal_order=self.__seasonalOrder
        )
        start = time.time()
        modelFitted = model.fit(low_memory=True)
        end = time.time()
        self.__fittingTime = round(end - start, 2)
        return modelFitted

    def __getForecast(self, model: ARIMAResults) -> pd.DataFrame:
        testLen = len(self.__testingData.index)
        return model.forecast(testLen, exog=self.__testingData[exog])

    def calculateForecast(self) -> Tuple[pd.DataFrame, float]:
        return self.__getForecast(self.__fitModel()), self.__fittingTime
