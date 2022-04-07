import pandas as pd

from data.reader import DataReader
from plotter.plotter import Plotter
from data.splitter import DataSplitter
from prediction.arima import ArimaPrediction
from prediction.arma import ArmaPrediction
from prediction.sarima import SarimaPrediction

pd.options.mode.chained_assignment = None


def main():
    dataReader = DataReader()
    heatingData = dataReader.getHeatingDeviceById(5066)
    (trainingData, testingData) = DataSplitter(heatingData).getSplittedData()

    # arima = ArimaPrediction(heatingData, (1, 1, 1))
    # arimaPrediction = arima.calculateForecast()
    #
    # arma = ArmaPrediction(heatingData, (1, 0, 1))
    # armaPrediction = arma.calculateForecast()

    sarima = SarimaPrediction(heatingData, (1, 1, 1, 8760))
    sarimaPrediction = sarima.calculateForecast()

    plotter = Plotter(heatingData, testingData)
    plotter.compare(sarimaPrediction)


if __name__ == '__main__':
    main()
