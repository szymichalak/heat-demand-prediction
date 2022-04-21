import pandas as pd

from customSelectors.columns import Timestamp
from data.split import DataSplit


class DataSplitter:
    def __init__(self, data: pd.DataFrame):
        self.__data = data
        
        self.__startTraining = '2016-01-01 00:00:00'
        self.__endTraining = '2017-11-30 23:00:00'
        self.__startTesting = '2017-12-01 00:00:00'
        self.__endTesting = '2017-12-31 23:00:00'
        
        self.__trainingData = self.__sliceTimeSeries(self.__startTraining, self.__endTraining)
        self.__testingData = self.__sliceTimeSeries(self.__startTesting, self.__endTesting)

    def getTrainingData(self) -> pd.DataFrame:
        return self.__trainingData

    def getTestingData(self) -> pd.DataFrame:
        return self.__testingData

    def getSplittedData(self) -> DataSplit:
        return DataSplit((self.getTrainingData(), self.getTestingData()))

    def __sliceTimeSeries(self, startTime: str, endTime: str) -> pd.DataFrame:
        startIndex = self.__data.loc[self.__data[Timestamp] == startTime].index[0]
        endIndex = self.__data.loc[self.__data[Timestamp] == endTime].index[0]
        assert (0 <= startIndex <= len(self.__data.index))
        assert (0 <= endIndex <= len(self.__data.index))
        assert (startIndex < endIndex)
        return self.__data[startIndex:(endIndex + 1)]
