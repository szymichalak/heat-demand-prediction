import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Tuple

from customSelectors.columns import Timestamp, Temperature, Energy, Wind
from prediction.arima.tuneResult import TuneResult


class Plotter:
    def __init__(self, data: pd.DataFrame):
        self.__data = data

    def generateTemperaturePlot(self):
        plt.figure(figsize=(12, 4), dpi=100)
        plt.plot(self.__data[Timestamp], self.__data[Temperature])
        plt.xlabel("Time")
        plt.ylabel("Temperature [°C]")
        plt.show()

    def generateWindPlot(self):
        plt.figure(figsize=(12, 4), dpi=100)
        plt.plot(self.__data[Timestamp], self.__data[Wind])
        plt.xlabel("Time")
        plt.ylabel("Wind [m/s]")
        plt.show()

    def generateEnergyConsumptionPlot(self):
        plt.figure(figsize=(12, 4), dpi=100)
        plt.plot(self.__data[Timestamp], self.__data[Energy])
        plt.xlabel("Time")
        plt.ylabel("Energy [kWh]")
        plt.show()

    def compare(self, actualData: pd.DataFrame, forecast: pd.DataFrame):
        plt.figure(figsize=(12, 4), dpi=100)
        plt.plot(actualData[Timestamp], actualData[Energy], label="Actual")
        plt.plot(actualData[Timestamp], forecast, label="Forecast")
        plt.legend(loc="upper left")
        plt.xlabel("Time")
        plt.ylabel("Energy [kWh]")
        plt.show()

    def tuneCompare(self, data: List[TuneResult], errorLabel: str = 'MAPE'):
        plt.figure(figsize=(6, 6), dpi=100)
        plt.scatter([res.time for res in data], [res.error for res in data])
        plt.xlabel("Time")
        plt.ylabel(errorLabel)

        for res in data:
            plt.annotate(res.order, (res.time, res.error), fontsize=6.5)

        plt.show()
