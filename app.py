import pandas as pd

from dataReader.dataReader import DataReader
from plotter.plotter import Plotter
from prediction.arima import ArimaPrediction
from prediction.arma import ArmaPrediction

pd.options.mode.chained_assignment = None


def main():
    dataReader = DataReader()
    heatingData = dataReader.getHeatingDeviceById(5066)

    arima = ArimaPrediction(heatingData, (1, 1, 1))
    arimaPrediction = arima.calculateForecast()

    arma = ArmaPrediction(heatingData, (1, 0, 1))
    armaPrediction = arma.calculateForecast()

    plotter = Plotter(heatingData, arima.getTestingData())
    plotter.compare(arimaPrediction)
    plotter.compare(armaPrediction)


if __name__ == '__main__':
    main()
