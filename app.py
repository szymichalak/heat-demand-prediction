import pandas as pd
from dataReader.dataReader import DataReader
from plotter.plotter import Plotter
pd.options.mode.chained_assignment = None


def main():
    dataReader = DataReader()
    waterDevices = dataReader.getWaterDevices()
    heatingDevices = dataReader.getHeatingDevices()
    plotter = Plotter(heatingDevices.get('5066'))
    plotter.generateEnergyConsumptionPlot()


if __name__ == '__main__':
    main()
