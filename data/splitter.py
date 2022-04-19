import pandas as pd

from customSelectors.columns import Timestamp, Energy, Temperature, Hum, Wind, TypeOfDay, Clouds, DayLength, Season
from data.split import DataSplit


class DataSplitter:
    def __init__(self, data: pd.DataFrame, aggregateByDay: bool = False):
        self.__data = data
        self.__aggregateByDay = aggregateByDay
        
        self.__startTraining = '2016-01-01 00:00:00'
        self.__endTraining = '2017-11-30 23:00:00'
        self.__startTesting = '2017-12-01 00:00:00'
        self.__endTesting = '2017-12-31 23:00:00'
        
        self.__trainingData = self.__sliceTimeSeries(self.__startTraining, self.__endTraining)
        self.__testingData = self.__sliceTimeSeries(self.__startTesting, self.__endTesting)

    def getTrainingData(self, aggregateByDay: int = False) -> pd.DataFrame:
        return self.__trainingData if not aggregateByDay else self.__getAggregateDataByDay(self.__trainingData)

    def getTestingData(self, aggregateByDay: int = False) -> pd.DataFrame:
        return self.__testingData if not aggregateByDay else self.__getAggregateDataByDay(self.__testingData)

    def getSplittedData(self, aggregateByDay: int = False) -> DataSplit:
        return DataSplit((self.getTrainingData(aggregateByDay), self.getTestingData(aggregateByDay)))

    def __sliceTimeSeries(self, startTime: str, endTime: str) -> pd.DataFrame:
        startIndex = self.__data.loc[self.__data[Timestamp] == startTime].index[0]
        endIndex = self.__data.loc[self.__data[Timestamp] == endTime].index[0]
        assert (0 <= startIndex <= len(self.__data.index))
        assert (0 <= endIndex <= len(self.__data.index))
        assert (startIndex < endIndex)
        return self.__data[startIndex:(endIndex + 1)]

    def __getAggregateDataByDay(self, data: pd.DataFrame) -> pd.DataFrame:
        data[Timestamp] = data[Timestamp].apply(lambda x: x.date())
        dates = list(data.groupby(Timestamp).groups.keys())
        data = data.groupby(Timestamp).agg(
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
