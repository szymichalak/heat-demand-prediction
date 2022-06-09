import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

from customSelectors.columns import Energy
from utils.groundTruth import GroundTruth
from utils.predictions import Predictions

ROUND = 3


def computeMSE(gt: pd.Series, pred: pd.Series) -> float:
    return round(mean_squared_error(gt, pred), ROUND)


def computeMAPE(gt: pd.Series, pred: pd.Series) -> float:
    return round(mean_absolute_percentage_error(gt, pred) * 100, ROUND)


def computeMSEWithHorizon(gt: GroundTruth, pred_horizon: Predictions) -> float:
    error = 0
    columns = pred_horizon.returnHorizon()
    for shift, column in enumerate(columns):
        error += computeMSE(gt.shift(shift)[Energy], pred_horizon.shift(shift))
    return round(error/len(columns), ROUND)


def computeMAPEWithHorizon(gt: GroundTruth, pred_horizon: Predictions) -> float:
    error = 0
    columns = pred_horizon.returnHorizon()
    for shift, column in enumerate(columns):
        error += computeMAPE(gt.shift(shift)[Energy], pred_horizon.shift(shift))
    return round(error/len(columns), ROUND)
