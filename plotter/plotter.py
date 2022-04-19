import pandas as pd
from matplotlib import pyplot as plt
from customSelectors.columns import Timestamp, Temperature, Energy, Wind


class Plotter:
    def __init__(self, data: pd.DataFrame, testingData: pd.DataFrame = None):
        self.__data = data
        self.__testingData = testingData

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

    def compare(self, prediction: pd.DataFrame):
        if self.__testingData is None:
            raise ValueError('No testing data provided! Can not create compare plot')
        plt.figure(figsize=(12, 4), dpi=100)
        plt.plot(self.__testingData[Timestamp], self.__testingData[Energy], label="Testing")
        plt.plot(self.__testingData[Timestamp], prediction, label="Prediction")
        plt.legend(loc="upper left")
        plt.xlabel("Time")
        plt.ylabel("Energy [kWh]")
        plt.show()
