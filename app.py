import pandas as pd

from data.reader import DataReader
from data.split import DataSplit
from plotter.plotter import Plotter
from data.splitter import DataSplitter
from prediction.arima import ArimaPrediction
from prediction.arma import ArmaPrediction
from prediction.sarima import SarimaPrediction

pd.options.mode.chained_assignment = None


def main():
    dataReader = DataReader()
    heatingData = dataReader.getHeatingDeviceById(5058)
    data: DataSplit = DataSplitter(heatingData).getSplittedData(aggregateByDay=False)
    plotter = Plotter(heatingData, data.testing)

    arima = ArimaPrediction(data, (1, 1, 1))
    arimaPrediction = arima.calculateForecast()
    plotter.compare(arimaPrediction)

    arma = ArmaPrediction(data, (1, 0, 1))
    armaPrediction = arma.calculateForecast()
    plotter.compare(armaPrediction)

    # sarima = SarimaPrediction(data, (1, 1, 1, 365))
    # sarimaPrediction = sarima.calculateForecast()
    # plotter.compare(sarimaPrediction)


if __name__ == '__main__':
    main()
