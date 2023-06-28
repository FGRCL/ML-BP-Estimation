from keras.metrics import MeanMetricWrapper

from mlbpestimation.losses.totalmeanabsoluteerror import TotalMeanAbsoluteErrorLoss


class TotalMeanAbsoluteErrorMetric(MeanMetricWrapper):
    def __init__(self, **kwargs):
        super().__init__(TotalMeanAbsoluteErrorLoss(), **kwargs)
