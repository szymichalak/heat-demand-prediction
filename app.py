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
from prediction.arima.tuner import Tuner

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')


def main():
    dataReader = DataReader()
    # for deviceId in dataReader.getHeatingIds():
    #     heatingData = dataReader.getHeatingDeviceById(deviceId)
    #     try:
    #         data: DataSplit = DataSplitter(heatingData).getSplittedData()
    #     except ValueError:
    #         print(f"Not enough observations on deviceId {deviceId}. Skipping...")
    #         continue
    #     title = f"Heating energy measures, deviceId: {deviceId}"
    #     plotter = Plotter(heatingData, title)
    #
    #     arima = ArimaPrediction(data, (1, 0, 0))
    #     prediction, time = arima.calculateForecast()
    #     plotter.compare(data.testing, prediction)
    #     mapeRes = mape(data.testing[Energy], prediction)
    #     mseRes = mse(data.testing[Energy], prediction)
    #     print(mapeRes, mseRes, deviceId)

    # heatingData = dataReader.getWaterDeviceById(5005)
    # data: DataSplit = DataSplitter(heatingData).getSplittedData()
    # plotter = Plotter(data.training)
    #
    # arima = ArimaPrediction(data, (0,0,1))
    # prediction, time = arima.calculateForecast()
    # plotter.compare(data.testing, prediction, 720, 0)
    # plotter.compare(data.testing, prediction)  # , 720, 0)
    # mapeRes = mape(data.testing[Energy], prediction)
    # mseRes = mse(data.testing[Energy], prediction)
    # print(mapeRes, mseRes)

    heatingData = dataReader.getHeatingDeviceById(5006)
    data: DataSplit = DataSplitter(heatingData).getSplittedData()
    tuner = Tuner(data)
    tuneRes = tuner.tuneSeasonalOrder([1], [1], [1])
    plotter = Plotter(heatingData)
    plotter.tuneCompare(tuneRes)


if __name__ == '__main__':
    main()
