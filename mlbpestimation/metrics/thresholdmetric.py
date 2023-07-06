from keras.metrics import Metric
from numpy import inf
from tensorflow import greater, less, logical_and
from tensorflow.python.ops.array_ops import boolean_mask


class ThresholdMetric(Metric):
    def __init__(self, metric: Metric, lower: float = -inf, upper: float = inf, **kwargs):
        super().__init__(**kwargs)
        self.metric = metric
        self.lower = lower
        self.upper = upper

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = logical_and(less(y_true, self.upper), greater(y_true, self.lower))

        y_true = boolean_mask(y_true, mask)
        y_pred = boolean_mask(y_pred, mask)
        sample_weight = boolean_mask(sample_weight, mask) if sample_weight is not None else sample_weight
        self.metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.metric.result()

    def reset_states(self):
        return self.metric.reset_state()
