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

    plotter = Plotter(deviceData)
    tuner = Tuner(data)

    tuneResult: List[TuneResult] = [TuneResult(row) for row in tuner.tuneArimaParameters([0, 1, 5], [0, 1, 2], [0, 1, 5])]
    plotter.tuneCompare(tuneResult)

    tuneResult: List[TuneResult] = [TuneResult(row) for row in tuner.tuneRfParameters([10, 50], [None, 1, 2], [2, 5], [1, 2, 5])]
    plotter.tuneCompare(tuneResult)

    tuneResult: List[TuneResult] = [TuneResult(row) for row in tuner.tuneNnParameters([2, 4, 8], [2, 4, 8], ['relu', 'tanh'], [5, 10], [32, 64])]
    plotter.tuneCompare(tuneResult)


if __name__ == '__main__':
    main()
