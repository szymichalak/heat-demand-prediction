import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from customSelectors.columns import Timestamp, Temperature, Energy


class Plotter:
    def __init__(self, data: pd.DataFrame, testingData: pd.DataFrame = None):
        self.data = data
        self.__testingData = testingData

    def generateTemperaturePlot(self):
        plt.figure(figsize=(12, 4), dpi=100)
        plt.plot(self.data[Timestamp], self.data[Temperature])
        plt.show()

    def generateEnergyConsumptionPlot(self):
        plt.figure(figsize=(12, 4), dpi=100)
        plt.plot(self.data[Timestamp], self.data[Energy])
        plt.show()

    def compare(self, prediction: pd.DataFrame):
        if self.__testingData is None:
            raise ValueError('No testing data provided! Can not create compare plot')
        plt.figure(figsize=(12, 4), dpi=100)
        plt.plot(self.__testingData[Timestamp], self.__testingData[Energy])
        plt.plot(self.__testingData[Timestamp], prediction)
        plt.show()
        print(np.square(np.subtract(self.data[Energy], prediction)).mean())
