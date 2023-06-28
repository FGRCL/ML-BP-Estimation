from keras.metrics import Metric
from tensorflow.python.ops.array_ops import boolean_mask


class MaskedMetric(Metric):
    def __init__(self, metric: Metric, mask, **kwargs):
        super().__init__(**kwargs)
        self.metric = metric
        self.mask = mask

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_mask = boolean_mask(y_true, self.mask, axis=1)
        y_pred_mask = boolean_mask(y_pred, self.mask, axis=1)
        self.metric.update_state(y_true_mask, y_pred_mask, sample_weight)

    def result(self):
        return self.metric.result()

    def reset_state(self):
        return self.metric.reset_state()
