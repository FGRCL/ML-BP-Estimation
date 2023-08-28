from tensorflow import Tensor

from mlbpestimation.models.metricreducer.base import MetricReducer


class MultiStep(MetricReducer):
    def pressure_reduce(self, y_true: Tensor, y_pred: Tensor) -> [Tensor, Tensor]:
        return y_true[:, -1], y_pred[:, -1]

    def sbp_reduce(self, y_true: Tensor, y_pred: Tensor) -> [Tensor, Tensor]:
        return y_true[:, -1, 0], y_pred[:, -1, 0]

    def dbp_reduce(self, y_true: Tensor, y_pred: Tensor) -> [Tensor, Tensor]:
        return y_true[:, -1, 1], y_pred[:, -1, 1]
