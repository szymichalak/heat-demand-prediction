import pandas as pd
from matplotlib import pyplot as plt
from selectors.columns import Timestamp, Temperature, Energy


class Plotter:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def generateTemperaturePlot(self):
        plt.figure(figsize=(12, 4), dpi=100)
        plt.plot(self.data[Timestamp], self.data[Temperature])
        plt.show()

    def generateEnergyConsumptionPlot(self):
        plt.figure(figsize=(12, 4), dpi=100)
        plt.plot(self.data[Timestamp], self.data[Energy])
        plt.show()
