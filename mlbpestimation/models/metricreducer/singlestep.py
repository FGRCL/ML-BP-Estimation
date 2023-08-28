from tensorflow import Tensor

from mlbpestimation.models.metricreducer.base import MetricReducer


class SingleStep(MetricReducer):
    def pressure_reduce(self, y_true: Tensor, y_pred: Tensor) -> [Tensor, Tensor]:
        return y_true, y_pred

    def sbp_reduce(self, y_true: Tensor, y_pred: Tensor) -> [Tensor, Tensor]:
        return y_true[:, 0], y_pred[:, 0]

    def dbp_reduce(self, y_true: Tensor, y_pred: Tensor) -> [Tensor, Tensor]:
        return y_true[:, 1], y_pred[:, 1]
