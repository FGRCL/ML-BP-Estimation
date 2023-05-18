from abc import ABC, abstractmethod

from tensorflow import Variable, concat
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.ops.math_ops import reduce_std


class StandardDeviationMetric(ABC, Metric):
    def __init__(self, **kwargs):
        super(StandardDeviationMetric, self).__init__(**kwargs)
        self.measures = Variable([], shape=(None,), validate_shape=False)

    def update_state(self, y_true, y_pred, sample_weight=None):
        measure = self.compute_metric(y_true, y_pred)
        self.measures.assign(concat([self.measures.value(), measure[:, 0]], axis=0))

    def result(self):
        return reduce_std(self.measures)

    def reset_state(self):
        self.measures = Variable([], shape=(None,), validate_shape=False)

    @abstractmethod
    def compute_metric(self, y_true, y_pred):
        ...


class StandardDeviationAbsoluteError(StandardDeviationMetric):

    def compute_metric(self, y_true, y_pred):
        return abs(y_pred - y_true)


class StandardDeviationPrediction(StandardDeviationMetric):

    def compute_metric(self, y_true, y_pred):
        return y_pred
