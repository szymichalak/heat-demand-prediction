from typing import List

import pandas as pd
import warnings

from data.reader import DataReader
from data.split import DataSplit
from data.splitter import DataSplitter
from plotter.plotter import Plotter
from prediction.arima.arimaPrediction import ArimaPrediction
from prediction.neuralNetwork.neuralNetworkPrediction import NeuralNetworkPrediction
from prediction.randomForest.randomForestPrediction import RandomForestPrediction
from tuner.tuneResult import TuneResult
from tuner.tuner import Tuner
from utils.groundTruth import GroundTruth
from utils.measures import computeMSEWithHorizon, computeMAPEWithHorizon

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')


def main():
    dataReader = DataReader()
    # deviceData = dataReader.getWaterDeviceById(5005)
    deviceData = dataReader.getHeatingDeviceById(5006)
    data: DataSplit = DataSplitter(deviceData).getSplittedData()
    gt = GroundTruth(data.testing)
    plotter = Plotter(deviceData)

    model = ArimaPrediction(data, order=(0, 0, 1))
    pred, t = model.calculateForecast()
    plotter.compare(gt.shift(0), pred.shift(0))


if __name__ == '__main__':
    main()
