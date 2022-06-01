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
    # deviceData = dataReader.getWaterDeviceById(5005)
    deviceData = dataReader.getHeatingDeviceById(5006)
    plotter = Plotter(deviceData)

    aggHours = [24, 12, 6, 4, 2, 1]
    for agg in aggHours:
        data: DataSplit = DataSplitter(deviceData).setAggregateHour(agg).getSplittedData(True)
        arima = ArimaPrediction(data, seasonalOrder=(0, 0, 1, 365 * 24 / agg))
        arimaPrediction, time = arima.calculateForecast()
        plotter.compare(data.testing, arimaPrediction)
        print('#####################')
        print('agg', agg)
        print('time', time)
        print('mape', mape(data.testing[Energy], arimaPrediction))
        print('mse', mse(data.testing[Energy], arimaPrediction))
        print('#####################')


if __name__ == '__main__':
    main()
