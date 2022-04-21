import pandas as pd

from data.reader import DataReader
from data.split import DataSplit
from plotter.plotter import Plotter
from data.splitter import DataSplitter
from prediction.arima.tuner import Tuner

pd.options.mode.chained_assignment = None


def main():
    dataReader = DataReader()
    heatingData = dataReader.getHeatingDeviceById(5058)
    data: DataSplit = DataSplitter(heatingData).getSplittedData()
    plotter = Plotter(heatingData)

    tuner = Tuner(data)
    result = tuner.tuneOrder([1], [1], [1])
    plotter.tuneCompare(result)


if __name__ == '__main__':
    main()
