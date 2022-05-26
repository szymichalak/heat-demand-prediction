import pandas as pd
from matplotlib import pyplot as plt
from typing import List

from customSelectors.columns import Timestamp, Temperature, Energy, Wind
from prediction.arima.tuneResult import TuneResult, stringRowToTuneResult, stringRowToTuneResultNN


class Plotter:
    def __init__(self, data: pd.DataFrame, title: str = None):
        self.__data = data
        self.__title = title

    def generateTemperaturePlot(self):
        plt.figure(figsize=(12, 4), dpi=300)
        plt.plot(self.__data[Timestamp], self.__data[Temperature])
        plt.xlabel("Time")
        plt.ylabel("Temperature [°C]")
        plt.title(self.__title)
        plt.show()

    def generateWindPlot(self):
        plt.figure(figsize=(12, 4), dpi=300)
        plt.plot(self.__data[Timestamp], self.__data[Wind])
        plt.xlabel("Time")
        plt.ylabel("Wind [m/s]")
        plt.title(self.__title)
        plt.show()

    def generateEnergyConsumptionPlot(self):
        plt.figure(figsize=(12, 4), dpi=300)
        plt.plot(self.__data[Timestamp], self.__data[Energy])
        plt.xlabel("Time")
        plt.ylabel("Energy [kWh]")
        plt.title(self.__title)
        plt.show()

    def compare(self, actualData: pd.DataFrame, forecast: pd.DataFrame, records: int = None, offset: int = 0):
        if records is not None:
            actualData = actualData[offset:offset+records+1]
            forecast = forecast[offset:offset+records+1]
        plt.figure(figsize=(12, 4), dpi=300)
        plt.plot(actualData[Timestamp], actualData[Energy], label="Actual")
        plt.plot(actualData[Timestamp], forecast, label="Forecast")
        plt.legend(loc="upper left")
        plt.xlabel("Time")
        plt.ylabel("Energy [kWh]")
        plt.title(self.__title)
        plt.show()

    def tuneCompare(self, data: List[TuneResult], errorLabel: str = 'MAPE', sliceResult: bool = False):
        if sliceResult:
            data = [row for row in data if row.error < 0.2]
            data = [row for row in data if row.time < 10]

        plt.figure(figsize=(8, 8), dpi=300)
        plt.scatter([res.time for res in data], [res.error for res in data])
        plt.xlabel("Time [s]")
        plt.ylabel(errorLabel)

        for res in data:
            plt.annotate(res.order, (res.time, res.error), fontsize=6.5)

        plt.title(self.__title)
        plt.show()

    def tuneCompareFromFile(self, fileName: str, errorLabel: str = 'MAPE', sliceResult: bool = False):
        with open(f"prediction/neuralNetwork/tuneResults/{fileName}", 'r') as file:
            data = [stringRowToTuneResultNN(row) for row in file.readlines()]
        self.tuneCompare(data, errorLabel, sliceResult)
