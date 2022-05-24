import time
import pandas as pd

from typing import Tuple

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation

from customSelectors.columns import Energy, exog
from data.split import DataSplit


class NeuralNetworkPrediction:
    def __init__(self, data: DataSplit, first_layer: int, second_layer: int, activation: str, epochs: int, batch_size: int):
        self.__data = data
        (self.__trainingData, self.__testingData) = (data.training, data.testing)
        self.__fittingTime: float = 0
        self.__epochs = epochs
        self.__batch_size = batch_size

        self.__regressor = Sequential()
        self.__regressor.add(Dense(first_layer, input_dim=4, activation=activation))
        self.__regressor.add(Dense(second_layer, activation=activation))
        self.__regressor.add(Dense(1))
        self.__regressor.compile(optimizer='adam', loss='mean_squared_error')

    def getTrainingData(self) -> pd.DataFrame:
        return self.__trainingData

    def getTestingData(self) -> pd.DataFrame:
        return self.__testingData

    def __fitModel(self) -> None:
        start = time.time()
        self.__regressor.fit(
            self.__trainingData[exog],
            self.__trainingData[Energy],
            epochs=self.__epochs,
            batch_size=self.__batch_size
        )
        end = time.time()
        self.__fittingTime = round(end - start, 2)

    def __getForecast(self) -> pd.DataFrame:
        return self.__regressor.predict(self.__testingData[exog])

    def calculateForecast(self) -> Tuple[pd.DataFrame, float]:
        self.__fitModel()
        return self.__getForecast(), self.__fittingTime
