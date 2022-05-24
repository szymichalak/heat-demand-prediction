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
from prediction.neuralNetwork.neuralNetworkPrediction import NeuralNetworkPrediction
from prediction.neuralNetwork.tuner import Tuner
# from prediction.randomForest.tuner import Tuner
from prediction.randomForest.randomForestPrediction import RandomForestPrediction

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')


def main():
    dataReader = DataReader()
    deviceData = dataReader.getWaterDeviceById(5005)
    # deviceData = dataReader.getHeatingDeviceById(5006)
    data: DataSplit = DataSplitter(deviceData).getSplittedData()
    plotter = Plotter(deviceData)

    tuner = Tuner(data)
    tuneRes = tuner.tuneParameters([2, 4, 8, 12, 24], [2, 6, 8, 12], ['relu', 'tanh'], [5, 10], [16, 32, 64])
    plotter.tuneCompare(tuneRes)
    # plotter.tuneCompare(tuneRes, sliceResult=True)


if __name__ == '__main__':
    main()
