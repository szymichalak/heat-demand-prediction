import time

from sklearn.metrics import mean_absolute_percentage_error as mape
from typing import List

from data.split import DataSplit
from prediction.arima.arimaPrediction import ArimaPrediction
from customSelectors.columns import Energy
from prediction.arima.tuneResult import TuneResult


class Tuner:
    def __init__(self, data: DataSplit):
        self.__data = data

    def tuneOrder(self, p_orders, d_orders, q_orders, saveResult: bool = True) -> List[TuneResult]:
        result = []
        for p in p_orders:
            for d in d_orders:
                for q in q_orders:
                    order = (p, d, q)
                    arima = ArimaPrediction(self.__data, order)
                    prediction, time = arima.calculateForecast()
                    mapeResult = mape(self.__data.testing[Energy], prediction)
                    result.append(TuneResult(round(mapeResult, 4), time, order))
                    print(f"Tuned {round(len(result) / (len(p_orders) * len(d_orders) * len(q_orders)), 2) * 100} %")

        if saveResult:
            self.__writeToFile(result)

        return result

    def __writeToFile(self, data: List[TuneResult]):
        fileName = time.strftime('%d_%b_%Y_%H_%M_%S')
        with open(f"prediction/arima/tuneResults/{fileName}", 'w') as file:
            file.writelines([res.toString() + '\n' for res in data])
