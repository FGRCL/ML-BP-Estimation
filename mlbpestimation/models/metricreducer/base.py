from abc import abstractmethod

from tensorflow import Tensor


class MetricReducer:
    @abstractmethod
    def pressure_reduce(self, y_true: Tensor, y_pred: Tensor) -> [Tensor, Tensor]:
        pass

    @abstractmethod
    def sbp_reduce(self, y_true: Tensor, y_pred: Tensor) -> [Tensor, Tensor]:
        pass

    @abstractmethod
    def dbp_reduce(self, y_true: Tensor, y_pred: Tensor) -> [Tensor, Tensor]:
        pass
