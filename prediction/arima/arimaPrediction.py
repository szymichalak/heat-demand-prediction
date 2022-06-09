import time
import pandas as pd

from typing import Tuple
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from customSelectors.columns import Energy, exog
from data.split import DataSplit
from utils.const import HORIZON
from utils.predictions import Predictions


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
        modelFitted = model.fit()
        end = time.time()
        self.__fittingTime = round(end - start, 2)
        return modelFitted

    def __getForecast(self, model: ARIMAResults, iteration: int) -> pd.DataFrame:
        return model.forecast(HORIZON, exog=self.__testingData[exog].iloc[iteration:iteration+HORIZON])

    def calculateForecast(self) -> Tuple[Predictions, float]:
        model = self.__fitModel()
        result = pd.DataFrame(columns=[h * pd.Timedelta(1, unit='h') for h in range(1, HORIZON+1)], index=self.__testingData.index[:-HORIZON])

        start = time.time()
        for i in range(len(self.__testingData.index) - HORIZON):
            prediction = self.__getForecast(model, i)
            result.at[self.__testingData.index.values[i]] = prediction.values
        end = time.time()
        return Predictions(result), self.__fittingTime + round(end - start, 2)
