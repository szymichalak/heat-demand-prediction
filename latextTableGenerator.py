import warnings

from tabulate import tabulate

from data.reader import DataReader
from data.split import DataSplit
from data.splitter import DataSplitter
from prediction.arima.arimaPrediction import ArimaPrediction
from prediction.neuralNetwork.neuralNetworkPrediction import NeuralNetworkPrediction
from prediction.randomForest.randomForestPrediction import RandomForestPrediction
from utils.groundTruth import GroundTruth
from utils.measures import computeMAPEWithHorizon, computeMSEWithHorizon

warnings.filterwarnings('ignore')


def generateRows():
    dataReader = DataReader()
    rows = [['ID', 'MAPE', "MSE", "t [s]", 'MAPE', "MSE", "t [s]", 'MAPE', "MSE", "t [s]"]]
    for i, deviceId in enumerate(dataReader.getHeatingIds()):
        deviceData = dataReader.getHeatingDeviceById(deviceId)
        try:
            data: DataSplit = DataSplitter(deviceData).getSplittedData()
            gt = GroundTruth(data.testing)
        except ValueError:
            continue

        arima = ArimaPrediction(data, order=(0, 0, 1))  # both
        arimaPrediction, arimaTime = arima.calculateForecast()
        arimaMAPE = computeMAPEWithHorizon(gt, arimaPrediction)
        arimaMSE = computeMSEWithHorizon(gt, arimaPrediction)

        rf = RandomForestPrediction(data, 50, None, 2, 5)  # heating
        # rf = RandomForestPrediction(data, 10, None, 2, 5)  # water
        rfPrediction, rfTime = rf.calculateForecast()
        rfMAPE = computeMAPEWithHorizon(gt, rfPrediction)
        rfMSE = computeMSEWithHorizon(gt, rfPrediction)

        nn = NeuralNetworkPrediction(data, 8, 2, 'relu', 10, 64)  # heating
        # nn = NeuralNetworkPrediction(data, 4, 2, 'relu', 5, 64)  # water
        nnPrediction, nnTime = nn.calculateForecast()
        nnMAPE = computeMAPEWithHorizon(gt, nnPrediction)
        nnMSE = computeMSEWithHorizon(gt, nnPrediction)

        rows.append(
            [
                deviceId,
                round(arimaMAPE, 3),
                round(arimaMSE, 3),
                arimaTime,
                round(rfMAPE, 3),
                round(rfMSE, 3),
                rfTime,
                round(nnMAPE, 3),
                round(nnMSE, 3),
                nnTime
            ]
        )

    avg = []
    for colId in range(10):
        col = [row[colId] for row in rows if type(row[colId]) != str]
        colAvg = sum(col) / len(col)
        avg.append(colAvg)
    rows.append(avg)
    return rows


def generateLatexTable(rows):
    print(tabulate(rows, headers='firstrow', tablefmt='latex_raw'))


dataRows = generateRows()
generateLatexTable(dataRows)
