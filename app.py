import pandas as pd
import warnings

from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse

from customSelectors.columns import Energy
from data.reader import DataReader
from data.split import DataSplit
from data.splitter import DataSplitter
from plotter.plotter import Plotter
from prediction.arima.arimaPrediction import ArimaPrediction
from prediction.randomForest.tuner import Tuner
from prediction.randomForest.randomForestPrediction import RandomForestPrediction

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')


def main():
    dataReader = DataReader()
    # deviceData = dataReader.getWaterDeviceById(5005)
    deviceData = dataReader.getHeatingDeviceById(5006)
    data: DataSplit = DataSplitter(deviceData).getSplittedData()
    plotter = Plotter(deviceData)

    rf = RandomForestPrediction(data)  # , 50, None, 5, 5)
    prediction, time = rf.calculateForecast()
    plotter.compare(data.testing, prediction)
    plotter.compare(data.testing, prediction, 720)
    mapeRes = mape(data.testing[Energy], prediction)
    mseRes = mse(data.testing[Energy], prediction)
    print(mapeRes, mseRes)

    # tuner = Tuner(data)
    # tuneRes = tuner.tuneParameters([10, 50, 100, 200], [None, 1, 2], [2, 5], [1, 2, 5])
    # plotter.tuneCompare(tuneRes)


if __name__ == '__main__':
    main()
