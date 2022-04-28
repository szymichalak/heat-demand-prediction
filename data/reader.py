import pickle
import gzip
import random
import pandas as pd

from typing import Dict, Set

from data.cleaner import DataCleaner
from customSelectors.columns import DeviceId, DeviceType


class DataReader:
    def __init__(self):
        self.__data: Dict[str, pd.DataFrame]
        self.__loadData()
        self.__clean()
        self.__devicesInfo: pd.DataFrame = None
        self.__loadDevicesInfo()
        self.__splitDataByType()
        self.__heatingDevices: Dict[str, pd.DataFrame]
        self.__waterDevices: Dict[str, pd.DataFrame]
        self.__water_ids: Set
        self.__heating_ids: Set

    def __loadData(self):
        pickle_file = 'data/data/data.pickle.gz'
        try:
            with gzip.open(pickle_file, 'rb') as f:
                self.__data: Dict[str, pd.DataFrame] = pickle.load(f)
        except:
            raise ValueError("Can't load data")

    def getData(self):
        return self.__data

    def __clean(self):
        dataCleaner = DataCleaner(self.__data)
        self.__data = dataCleaner.getCleanedData()

    def __loadDevicesInfo(self):
        def_info_file = 'data/data/urzadzenia_rozliczeniowe.csv'
        try:
            self.__devicesInfo = pd.read_csv(def_info_file).set_index(DeviceId)
        except:
            raise ValueError("Can't load data")

    def getDevicesInfo(self):
        return self.__devicesInfo

    def __splitDataByType(self):
        self.__water_ids = set(self.__devicesInfo[self.__devicesInfo[DeviceType] == 'Licznik CW(O)'].index)
        self.__heating_ids = set(self.__devicesInfo[self.__devicesInfo[DeviceType] == 'Licznik CO(O)'].index)
        all_ids = set([int(x) for x in self.__data.keys()])
        self.__water_ids = all_ids.intersection(self.__water_ids)
        self.__heating_ids = all_ids.intersection(self.__heating_ids)

        self.__waterDevices = {k: v for k, v in self.__data.items() if int(k) in self.__water_ids}
        self.__heatingDevices = {k: v for k, v in self.__data.items() if int(k) in self.__heating_ids}

    def getWaterDevices(self) -> Dict[str, pd.DataFrame]:
        return self.__waterDevices

    def getHeatingDevices(self) -> Dict[str, pd.DataFrame]:
        return self.__heatingDevices

    def getWaterDeviceById(self, index: int) -> pd.DataFrame:
        if index in self.__water_ids:
            return self.__waterDevices.get(str(index))
        else:
            raise ValueError("Given id doesnt exist")

    def getHeatingDeviceById(self, index: int) -> pd.DataFrame:
        if index in self.__heating_ids:
            return self.__heatingDevices.get(str(index))
        else:
            raise ValueError("Given id doesnt exist")

    def getRandomWaterDevice(self) -> pd.DataFrame:
        return self.__waterDevices.get(str(random.sample(self.__water_ids, 1)[0]))

    def getRandomHeatingDevice(self) -> pd.DataFrame:
        return self.__heatingDevices.get(str(random.sample(self.__heating_ids, 1)[0]))

    def getWaterIds(self) -> set:
        return self.__water_ids

    def getHeatingIds(self) -> set:
        return self.__heating_ids
