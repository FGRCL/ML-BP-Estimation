from abc import abstractmethod

from keras import Model
from tensorflow import TensorSpec

from mlbpestimation.models.metricreducer.base import MetricReducer


class BloodPressureModel(Model):

    @abstractmethod
    def set_input(self, input_spec: TensorSpec):
        pass

    @abstractmethod
    def set_output(self, output_spec: TensorSpec):
        pass

    @abstractmethod
    def get_metric_reducer_strategy(self) -> MetricReducer:
        pass
