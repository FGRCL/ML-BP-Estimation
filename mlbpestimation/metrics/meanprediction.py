from keras.metrics import Metric
from tensorflow import Variable, concat, reduce_mean


class MeanPrediction(Metric):
    def __init__(self, **kwargs):
        super(MeanPrediction, self).__init__(**kwargs)
        self.predictions = Variable([], shape=(None,), validate_shape=False)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.predictions.assign(concat([self.predictions, y_pred], axis=0))

    def result(self):
        return reduce_mean(self.predictions)

    def reset_state(self):
        self.predictions.assign(Variable([], shape=(None,), validate_shape=False))
