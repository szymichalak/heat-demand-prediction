import copy

import pandas as pd

from customSelectors.columns import Energy, Timestamp, Temperature, Wind, Hum, Clouds, DayLength, TypeOfDay, Season
from data.split import DataSplit


class DataSplitter:
    def __init__(self, data: pd.DataFrame):
        self.__data = data

        self.__startTraining = '2016-01-01 00:00:00'
        self.__endTraining = '2017-12-31 23:00:00'
        self.__startTesting = '2018-10-01 00:00:00'
        self.__endTesting = '2019-02-25 23:00:00'

        self.__aggregateHours: int = 24
        
        self.__trainingData = self.__sliceTimeSeries(self.__startTraining, self.__endTraining)
        self.__testingData = self.__sliceTimeSeries(self.__startTesting, self.__endTesting)
        self.__handleZeroEnergyConsumption()
        self.__verifySplit()

    def getTrainingData(self, aggregateByDay: int = False) -> pd.DataFrame:
        return self.__trainingData if not aggregateByDay else self.__getAggregateDataByDay(self.__trainingData)

    def getTestingData(self, aggregateByDay: int = False) -> pd.DataFrame:
        return self.__testingData if not aggregateByDay else self.__getAggregateDataByDay(self.__testingData)

    def getSplittedData(self, aggregateByDay: int = False) -> DataSplit:
        return DataSplit((self.getTrainingData(aggregateByDay), self.getTestingData(aggregateByDay)))

    def setAggregateHour(self, value: int):
        self.__aggregateHours = value
        return self

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

    def __getAggregateDataByDay(self, data: pd.DataFrame) -> pd.DataFrame:
        data['Timestamp'] = data[Timestamp].apply(self.__groupBy)
        dates = list(data.groupby('Timestamp').groups.keys())
        data = data.groupby('Timestamp').agg(
            {
                Energy: 'sum',
                Temperature: 'mean',
                Wind: 'mean',
                Hum: 'mean',
                Clouds: 'mean',
                DayLength: 'mean',
                TypeOfDay: 'mean',
                Season: 'mean'
            }
        )
        data = data.assign(Date=dates)
        return data.rename(columns={'Date': Timestamp})

    def __groupBy(self, date):
        factor = date.time().hour // self.__aggregateHours
        return date.replace(hour=factor)
