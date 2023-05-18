from tensorflow import Variable, concat, reduce_mean
from tensorflow.python.keras.metrics import Metric


class MeanPrediction(Metric):
    def __init__(self, **kwargs):
        super(MeanPrediction, self).__init__(**kwargs)
        self.predictions = Variable([], shape=(None,), validate_shape=False)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.predictions.assign(concat([self.predictions, y_pred[:, 0]], axis=0))

    def result(self):
        return reduce_mean(self.predictions)

    def reset_state(self):
        self.predictions = Variable([], shape=(None,), validate_shape=False)
