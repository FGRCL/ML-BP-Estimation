from abc import ABC, abstractmethod

from keras.metrics import Metric
import tensorflow as tf


class StandardDeviationMetric(ABC):

    @abstractmethod
    def compute_metric(self, y_true, y_pred):
        ...


class AbsoluteError(StandardDeviationMetric):

    def compute_metric(self, y_true, y_pred):
        return abs(y_pred - y_true)


class StandardDeviation(Metric):

    def __init__(self, metric: StandardDeviationMetric, **kwargs):
        super(StandardDeviation, self).__init__(**kwargs)
        self.metric = metric
        self.std = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        metrics = self.metric.compute_metric(y_true, y_pred)
        self.std = tf.math.reduce_std(metrics)

    def result(self):
        return self.std
