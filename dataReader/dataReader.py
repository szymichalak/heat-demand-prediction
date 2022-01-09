import pickle
import gzip
import pandas as pd
from typing import Dict

from dataCleaner.dataCleaner import DataCleaner
from selectors.columns import DeviceId, DeviceType


class DataReader:
    def __init__(self):
        self.__data: Dict[str, pd.DataFrame] = None
        self.__loadData()
        self.__clean()
        self.__devicesInfo: pd.DataFrame = None
        self.__loadDevicesInfo()
        self.__splitDataByType()
        self.__heatingDevices: Dict[str, pd.DataFrame]
        self.__waterDevices: Dict[str, pd.DataFrame]

    def __loadData(self):
        pickle_file = 'data/data.pickle.gz'
        try:
            with gzip.open(pickle_file, 'rb') as f:
                self.__data = pickle.load(f)
        except:
            raise ValueError("Can't load __data")

    def getData(self):
        return self.__data

    def __clean(self):
        dataCleaner = DataCleaner(self.__data)
        self.__data = dataCleaner.getCleanedData()

    def __loadDevicesInfo(self):
        def_info_file = 'data/urzadzenia_rozliczeniowe.csv'
        try:
            self.__devicesInfo = pd.read_csv(def_info_file).set_index(DeviceId)
        except:
            raise ValueError("Can't load __data")

    def getDevicesInfo(self):
        return self.__devicesInfo

    def __splitDataByType(self):
        water_ids = set(self.__devicesInfo[self.__devicesInfo[DeviceType] == 'Licznik CW(O)'].index)
        heating_ids = set(self.__devicesInfo[self.__devicesInfo[DeviceType] == 'Licznik CO(O)'].index)
        self.__waterDevices = {k: v for k, v in self.__data.items() if int(k) in water_ids}
        self.__heatingDevices = {k: v for k, v in self.__data.items() if int(k) in heating_ids}

    def getWaterDevices(self) -> Dict[str, pd.DataFrame]:
        return self.__waterDevices

    def getHeatingDevices(self) -> Dict[str, pd.DataFrame]:
        return self.__heatingDevices
