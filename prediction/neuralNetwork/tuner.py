import time

from sklearn.metrics import mean_absolute_percentage_error as mape
from typing import List

from data.split import DataSplit
from prediction.arima.arimaPrediction import ArimaPrediction
from customSelectors.columns import Energy
from prediction.arima.tuneResult import TuneResult
from prediction.neuralNetwork.neuralNetworkPrediction import NeuralNetworkPrediction
from prediction.randomForest.randomForestPrediction import RandomForestPrediction


class Tuner:
    def __init__(self, data: DataSplit):
        self.__data = data

    def tuneParameters(
        self,
        first_layers: List[int],
        second_layers: List[int],
        activations: List[str],
        epochs: List[int],
        batch_sizes: List[int],
        saveResult: bool = True
    ) -> List[TuneResult]:
        result = []
        for first in first_layers:
            for second in second_layers:
                for act in activations:
                    for epoch in epochs:
                        for batch in batch_sizes:
                            order = (first, second, act, epoch, batch)
                            nn = NeuralNetworkPrediction(self.__data, first, second, act, epoch, batch)
                            prediction, fittingTime = nn.calculateForecast()
                            mapeResult = mape(self.__data.testing[Energy], prediction)
                            result.append(TuneResult(round(mapeResult, 4), fittingTime, order))
                            print(f"Tuned {int(len(result) / (len(first_layers) * len(second_layers) * len(activations) * len(epochs) *len(batch_sizes)) * 100)} %")

        if saveResult:
            self.__writeToFile(result)

        return result

    def __writeToFile(self, data: List[TuneResult]):
        fileName = time.strftime('%d_%b_%Y_%H_%M_%S')
        with open(f"prediction/neuralNetwork/tuneResults/{fileName}", 'w') as file:
            file.writelines([res.toString() + '\n' for res in data])
