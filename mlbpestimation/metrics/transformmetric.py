from typing import Callable

from keras.metrics import Metric


class TransformMetric(Metric):
    def __init__(self, metric: Metric, transform: Callable, **kwargs):
        super().__init__(**kwargs)
        self.metric = metric
        self.transform = transform

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = self.transform(y_true, y_pred)
        self.metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.metric.result()

    def reset_state(self):
        return self.metric.reset_state()
