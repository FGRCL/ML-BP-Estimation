from keras.layers import BatchNormalization, Conv1D, Dense, Dropout, Flatten, MaxPooling1D, ReLU
from tensorflow import TensorSpec

from mlbpestimation.models.basemodel import BloodPressureModel
from mlbpestimation.models.metricreducer.base import MetricReducer
from mlbpestimation.models.metricreducer.singlestep import SingleStep


class AlexNet(BloodPressureModel):
    def __init__(self):
        super().__init__()

        self._metric_reducer = SingleStep()

        self._input_layer = None
        self._layers = [
            Conv1D(96, 11, 4),
            BatchNormalization(),
            ReLU(),
            MaxPooling1D(3, 2),

            Conv1D(256, 5),
            ReLU(),
            BatchNormalization(),
            MaxPooling1D(3, 2),

            Conv1D(384, 3),
            ReLU(),
            Conv1D(384, 3),
            ReLU(),
            Conv1D(256, 3),
            ReLU(),
            MaxPooling1D(3, 2),

            Flatten(),
            Dropout(0.5),
            Dense(4096),
            ReLU(),
            Dropout(0.5),
            Dense(4096),
            ReLU(),
            Dense(2),
        ]

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x

    def set_input(self, input_spec: TensorSpec):
        pass

    def set_output(self, output_spec: TensorSpec):
        pass

    def get_metric_reducer_strategy(self) -> MetricReducer:
        return self._metric_reducer
