import time
from typing import List

from data.split import DataSplit
from prediction.arima.arimaPrediction import ArimaPrediction
from prediction.neuralNetwork.neuralNetworkPrediction import NeuralNetworkPrediction
from prediction.randomForest.randomForestPrediction import RandomForestPrediction
from utils.groundTruth import GroundTruth
from utils.measures import computeMSEWithHorizon, computeMAPEWithHorizon


class Tuner:
    def __init__(self, data: DataSplit):
        self.__data = data
        self.__gt = GroundTruth(data.testing)
        self.__measure = 'MSE'

    def setMeasure(self, measure: str):
        try:
            if ['MSE', 'MAPE'].index(measure):
                self.__measure = measure
                return self
        except ValueError:
            raise ValueError('Required method parameter: "MSE" or "MAPE"')

    def tuneArimaParameters(
        self,
        p_orders: List[int],
        d_orders: List[int],
        q_orders: List[int],
        saveResult: bool = True
    ) -> List[str]:
        result = []
        for p in p_orders:
            for d in d_orders:
                for q in q_orders:
                    if p + d + q == 0:
                        continue
                    order = (p, d, q)
                    arima = ArimaPrediction(self.__data, order)
                    prediction, predictionTime = arima.calculateForecast()

                    error = 0
                    if self.__measure == "MSE":
                        error = computeMSEWithHorizon(self.__gt, prediction)
                    if self.__measure == "MAPE":
                        error = computeMAPEWithHorizon(self.__gt, prediction)

                    result.append(f"{error},{int(predictionTime)},{p},{d},{q}")
                    print(f"Tuned {int(len(result) / (len(p_orders) * len(d_orders) * len(q_orders)) * 100)} %")

        if saveResult:
            self.__writeToFile(result, 'arima')

        return result

    def tuneRfParameters(
        self,
        n_estimators: List[int],
        max_depth: List[int],
        min_samples_split: List[int],
        min_samples_leaf: List[int],
        saveResult: bool = True
    ) -> List[str]:
        result = []
        for est in n_estimators:
            for depth in max_depth:
                for split in min_samples_split:
                    for leaf in min_samples_leaf:
                        randomForest = RandomForestPrediction(self.__data, est, depth, split, leaf)
                        prediction, predictionTime = randomForest.calculateForecast()

                        error = 0
                        if self.__measure == "MSE":
                            error = computeMSEWithHorizon(self.__gt, prediction)
                        if self.__measure == "MAPE":
                            error = computeMAPEWithHorizon(self.__gt, prediction)

                        result.append(f"{error},{int(predictionTime)},{est},{depth},{split},{leaf}")
                        print(f"Tuned {int(len(result) / (len(n_estimators) * len(max_depth) * len(min_samples_split) * len(min_samples_leaf)) * 100)} %")

        if saveResult:
            self.__writeToFile(result, 'rf')

        return result

    def tuneNnParameters(
        self,
        first_layers: List[int],
        second_layers: List[int],
        activations: List[str],
        epochs: List[int],
        batch_sizes: List[int],
        saveResult: bool = True
    ) -> List[str]:
        result = []
        for first in first_layers:
            for second in second_layers:
                for act in activations:
                    for epoch in epochs:
                        for batch in batch_sizes:
                            nn = NeuralNetworkPrediction(self.__data, first, second, act, epoch, batch)
                            prediction, predictionTime = nn.calculateForecast()

                            error = 0
                            if self.__measure == "MSE":
                                error = computeMSEWithHorizon(self.__gt, prediction)
                            if self.__measure == "MAPE":
                                error = computeMAPEWithHorizon(self.__gt, prediction)

                            result.append(f"{error},{int(predictionTime)},{first},{second},{act},{epoch},{batch}")
                            print(f"Tuned {int(len(result) / (len(first_layers) * len(second_layers) * len(activations) * len(epochs) *len(batch_sizes)) * 100)} %")

        if saveResult:
            self.__writeToFile(result, 'nn')

        return result

    def __writeToFile(self, data: List[str], method: str):
        fileName = f"{time.strftime('%d_%b_%Y_%H_%M_%S')}_{self.__measure}_{method}"
        with open(f"tuner/tuneResults/{fileName}", 'w') as file:
            file.writelines([f"{res}\n" for res in data])
