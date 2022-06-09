import pandas as pd
import warnings

from data.reader import DataReader
from data.split import DataSplit
from data.splitter import DataSplitter
from plotter.plotter import Plotter
from prediction.arima.arimaPrediction import ArimaPrediction
from prediction.neuralNetwork.neuralNetworkPrediction import NeuralNetworkPrediction
from prediction.randomForest.randomForestPrediction import RandomForestPrediction
from utils.groundTruth import GroundTruth
from utils.measures import computeMSEWithHorizon, computeMAPEWithHorizon

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')


def main():
    dataReader = DataReader()
    # deviceData = dataReader.getWaterDeviceById(5005)
    deviceData = dataReader.getHeatingDeviceById(5006)
    plotter = Plotter(deviceData)

    data: DataSplit = DataSplitter(deviceData).getSplittedData()
    arima = ArimaPrediction(data, order=(0, 0, 1))
    arimaPrediction, time = arima.calculateForecast()
    gt = GroundTruth(data.testing)
    shift = 71
    plotter.compare(gt.shift(shift), arimaPrediction.shift(shift))
    plotter.compare(gt.shift(shift), arimaPrediction.shift(shift), 250)
    print('time', time)
    print('mape', computeMAPEWithHorizon(gt, arimaPrediction))
    print('mse', computeMSEWithHorizon(gt, arimaPrediction))


if __name__ == '__main__':
    main()
