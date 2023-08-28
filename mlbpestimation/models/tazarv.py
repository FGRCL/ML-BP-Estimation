from keras import Sequential
from keras.layers import BatchNormalization, Conv1D, Dense, Dropout, LSTM, MaxPooling1D
from tensorflow import TensorSpec

from mlbpestimation.models.basemodel import BloodPressureModel
from mlbpestimation.models.metricreducer.base import MetricReducer
from mlbpestimation.models.metricreducer.singlestep import SingleStep


class Tazarv(BloodPressureModel):
    def __init__(self):
        super().__init__()

        self.metric_reducer = SingleStep()

        self._layers = Sequential([
            Conv1D(64, 15, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(4),
            Dropout(0.1),
            LSTM(64, return_sequences=True),
            LSTM(64),
            Dense(2),
        ])

    def set_input(self, input_spec: TensorSpec):
        pass

    def set_output(self, output_spec: TensorSpec):
        pass

    def call(self, inputs, training=None, mask=None):
        return self._layers(inputs)

    def get_metric_reducer_strategy(self) -> MetricReducer:
        return self.metric_reducer

    def set_input_shape(self, dataset_spec):
        pass

    def get_config(self):
        return {}
