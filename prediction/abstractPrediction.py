import time
import pandas as pd

from typing import Tuple
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from customSelectors.columns import Energy, exog
from data.splitter import DataSplitter


class AbstractPrediction:
    def __init__(self, data: pd.DataFrame, order: Tuple = (0, 0, 0), seasonalOrder: Tuple = (0, 0, 0, 0)):
        self.__data = data
        self.__order = order
        self.__seasonalOrder = seasonalOrder
        (self.__trainingData, self.__testingData) = DataSplitter(data).getSplittedData()

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
        print('Fitting is starting')
        start = time.time()
        modelFitted = model.fit()
        end = time.time()
        print(f"Model fitted within {int(end - start)} seconds")
        return modelFitted

    def __getForecast(self, model: ARIMAResults) -> pd.DataFrame:
        testLen = len(self.__testingData.index)
        return model.forecast(testLen, exog=self.__testingData[exog])

    def calculateForecast(self) -> pd.DataFrame:
        return self.__getForecast(self.__fitModel())
