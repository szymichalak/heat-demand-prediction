import pandas as pd

from customSelectors.columns import Energy
from data.split import DataSplit


class DataSplitter:
    def __init__(self, data: pd.DataFrame):
        self.__data = data

        self.__startTraining = '2016-01-01 00:00:00'
        self.__endTraining = '2017-12-31 23:00:00'
        self.__startTesting = '2018-10-01 00:00:00'
        self.__endTesting = '2019-02-25 23:00:00'
        
        self.__trainingData = self.__sliceTimeSeries(self.__startTraining, self.__endTraining)
        self.__testingData = self.__sliceTimeSeries(self.__startTesting, self.__endTesting)
        self.__handleZeroEnergyConsumption()
        self.__verifySplit()

    def getTrainingData(self) -> pd.DataFrame:
        return self.__trainingData

    def getTestingData(self) -> pd.DataFrame:
        return self.__testingData

    def getSplittedData(self) -> DataSplit:
        return DataSplit((self.getTrainingData(), self.getTestingData()))

    def __sliceTimeSeries(self, startTime: str, endTime: str) -> pd.DataFrame:
        startIndex = self.__data.index.get_loc(startTime)
        endIndex = self.__data.index.get_loc(endTime)
        assert (0 <= startIndex <= len(self.__data.index))
        assert (0 <= endIndex <= len(self.__data.index))
        assert (startIndex < endIndex)
        return self.__data[startIndex:(endIndex+1)]

    def __handleZeroEnergyConsumption(self):
        self.__trainingData.drop(self.__trainingData[self.__trainingData[Energy] == 0].index, inplace=True)
        self.__testingData.drop(self.__testingData[self.__testingData[Energy] == 0].index, inplace=True)

    def __verifySplit(self):
        if len(self.__trainingData.index) < 5000:
            raise ValueError('Too few training observations')
        if len(self.__testingData.index) < 1500:
            raise ValueError('To few testing observations')
