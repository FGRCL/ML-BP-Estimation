from keras import Sequential
from keras.layers import Dense, Dropout, LSTM
from tensorflow import TensorSpec

from mlbpestimation.models.basemodel import BloodPressureModel
from mlbpestimation.models.metricreducer.base import MetricReducer
from mlbpestimation.models.metricreducer.singlestep import SingleStep


class Harfiya(BloodPressureModel):
    def __init__(self):
        super().__init__()

        self.metric_reducer = SingleStep()

        self.sequential = Sequential([
            LSTM(128, return_sequences=True),
            LSTM(128, return_sequences=True),
            LSTM(128, return_sequences=True),
            LSTM(128),
            Dropout(0.2),
            Dense(2)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.sequential(inputs)

    def set_input(self, input_spec: TensorSpec):
        pass

    def set_output(self, output_spec: TensorSpec):
        pass

    def get_metric_reducer_strategy(self) -> MetricReducer:
        return self.metric_reducer
