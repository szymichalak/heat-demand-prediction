import time
import pandas as pd

from typing import Tuple
from sklearn.ensemble import RandomForestRegressor

from customSelectors.columns import Energy, exog
from data.split import DataSplit
from utils.const import HORIZON
from utils.predictions import Predictions


class RandomForestPrediction:
    def __init__(
            self,
            data: DataSplit,
            n_estimators: int = 100,
            max_depth: int = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1
    ):
        self.__data = data
        (self.__trainingData, self.__testingData) = (data.training, data.testing)
        self.__fittingTime: float = 0
        self.__model: RandomForestRegressor = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1
        )

    def getTrainingData(self) -> pd.DataFrame:
        return self.__trainingData

    def getTestingData(self) -> pd.DataFrame:
        return self.__testingData

    def __fitModel(self) -> RandomForestRegressor:
        start = time.time()
        modelFitted = self.__model.fit(
            self.__trainingData[exog],
            self.__trainingData[Energy]
        )
        end = time.time()
        self.__fittingTime = round(end - start, 2)
        return modelFitted

    def __getForecast(self, model: RandomForestRegressor, iteration: int) -> pd.DataFrame:
        return model.predict(self.__testingData[exog].iloc[iteration:iteration+HORIZON])

    def calculateForecast(self) -> Tuple[Predictions, float]:
        model = self.__fitModel()
        result = pd.DataFrame(columns=[h * pd.Timedelta(1, unit='h') for h in range(1, HORIZON+1)], index=self.__testingData.index[:-HORIZON])

        start = time.time()
        for i in range(len(self.__testingData.index) - HORIZON):
            prediction = self.__getForecast(model, i)
            result.at[self.__testingData.index.values[i]] = prediction
        end = time.time()
        return Predictions(result), self.__fittingTime + round(end - start, 2)
