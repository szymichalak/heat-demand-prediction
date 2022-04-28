import numpy as np
import pandas as pd
from typing import List, Dict
from customSelectors.columns import Temperature, DayLength, Timestamp, Energy


class DataCleaner:
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.__data = data
        self.__cleanTemperature()
        self.__cleanDayLength()
        self.__updateIndex()

    # replace odds values (-inf, -50> u <50, inf) with previous value
    def __cleanTemperature(self):
        keys: List[str] = self.__data.keys()
        for key in keys:
            df = self.__data.get(key)
            df[Temperature].loc[df[Temperature] >= 50] = np.NaN
            df[Temperature].loc[df[Temperature] <= -50] = np.NaN
            df[Temperature].fillna(method='ffill', inplace=True)
            self.__data[key] = df

    # replace empty values of day length for 29th of February
    def __cleanDayLength(self):
        keys: List[str] = self.__data.keys()
        for key in keys:
            df = self.__data.get(key)
            df[DayLength].loc[df[DayLength] == 0] = np.NaN
            df[DayLength].fillna(method='ffill', inplace=True)
            self.__data[key] = df

    def __updateIndex(self):
        for key, data in self.__data.items():
            data.set_index(Timestamp, inplace=True, drop=False)

    def getCleanedData(self) -> Dict[str, pd.DataFrame]:
        return self.__data
