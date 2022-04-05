import time
import pandas as pd

from typing import Tuple
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from customSelectors.columns import Timestamp, Energy, exog


class AbstractPrediction:
    def __init__(self, data: pd.DataFrame, order: Tuple = (0, 0, 0), seasonalOrder: Tuple = (0, 0, 0, 0)):
        self.data = data
        self.__order = order
        self.__seasonalOrder = seasonalOrder

        self.__startTraining = '2016-01-01 00:00:00'
        self.__endTraining = '2016-11-30 23:00:00'
        self.__startTesting = '2016-12-01 00:00:00'
        self.__endTesting = '2016-12-31 23:00:00'
        self.__trainingData = self.__sliceTimeSeries(self.__startTraining, self.__endTraining)
        self.__testingData = self.__sliceTimeSeries(self.__startTesting, self.__endTesting)

    def getTrainingData(self) -> pd.DataFrame:
        return self.__trainingData

    def getTestingData(self) -> pd.DataFrame:
        return self.__testingData

    def __sliceTimeSeries(self, startTime: str, endTime: str) -> pd.DataFrame:
        startIndex = self.data.loc[self.data[Timestamp] == startTime].index[0]
        endIndex = self.data.loc[self.data[Timestamp] == endTime].index[0]
        assert (0 <= startIndex <= len(self.data.index))
        assert (0 <= endIndex <= len(self.data.index))
        assert (startIndex < endIndex)
        return self.data[startIndex:(endIndex + 1)]

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
