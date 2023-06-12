import tensorflow
from tensorflow import Tensor, reduce_mean, reduce_sum


class TotalMeanAbsoluteErrorLoss:
    def __call__(self, y_true: Tensor, y_pred: Tensor):
        return reduce_sum(reduce_mean(tensorflow.abs(y_true - y_pred), axis=0))

    def get_config(self):
        return {}
