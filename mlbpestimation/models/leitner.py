from keras import Sequential
from keras.engine.base_layer import Layer
from keras.layers import BatchNormalization, Concatenate, Conv1D, Dense, Flatten, GRU, ReLU
from tensorflow import TensorSpec

from mlbpestimation.models.basemodel import BloodPressureModel
from mlbpestimation.models.metricreducer.base import MetricReducer
from mlbpestimation.models.metricreducer.singlestep import SingleStep


class Leitner(BloodPressureModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metric_reducer = SingleStep()
        self._layers = [
            ConvBlock(),
            GRU(25, return_sequences=True),
            BatchNormalization(),
            Flatten(),
            Dense(64),
            BatchNormalization(),
            ReLU(),
            Dense(2),
            BatchNormalization(),
            ReLU(),
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


class ConvBlock(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first_block = Sequential([
            Conv1D(50, 7, padding='same'),
            BatchNormalization(),
            ReLU()
        ])

        self.second_block = Sequential([
            Conv1D(50, 7, padding='same'),
            BatchNormalization(),
            ReLU(),
            Conv1D(50, 7, padding='same'),
            BatchNormalization(),
            ReLU()
        ])

        self.concatenate = Concatenate()

    def call(self, inputs, *args, **kwargs):
        first_block_output = self.first_block(inputs)
        second_block_output = self.second_block(first_block_output)
        return self.concatenate([first_block_output, second_block_output])
