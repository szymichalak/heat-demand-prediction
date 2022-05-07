import time

from sklearn.metrics import mean_absolute_percentage_error as mape
from typing import List

from data.split import DataSplit
from prediction.arima.arimaPrediction import ArimaPrediction
from customSelectors.columns import Energy
from prediction.arima.tuneResult import TuneResult
from prediction.randomForest.randomForestPrediction import RandomForestPrediction


class Tuner:
    def __init__(self, data: DataSplit):
        self.__data = data

    def tuneParameters(
        self,
        n_estimators: List[int],
        max_depth: List[int],
        min_samples_split: List[int],
        min_samples_leaf: List[int],
        saveResult: bool = True
    ) -> List[TuneResult]:
        result = []
        for est in n_estimators:
            for depth in max_depth:
                for split in min_samples_split:
                    for leaf in min_samples_leaf:
                        order = (est, depth, split, leaf)
                        randomForest = RandomForestPrediction(self.__data, est, depth, split, leaf)
                        prediction, fittingTime = randomForest.calculateForecast()
                        mapeResult = mape(self.__data.testing[Energy], prediction)
                        result.append(TuneResult(round(mapeResult, 4), fittingTime, order))
                        print(f"Tuned {int(len(result) / (len(n_estimators) * len(max_depth) * len(min_samples_split) * len(min_samples_leaf)) * 100)} %")

        if saveResult:
            self.__writeToFile(result)

        return result

    def __writeToFile(self, data: List[TuneResult]):
        fileName = time.strftime('%d_%b_%Y_%H_%M_%S')
        with open(f"prediction/arima/tuneResults/{fileName}", 'w') as file:
            file.writelines([res.toString() + '\n' for res in data])
