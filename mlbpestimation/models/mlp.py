from keras import Sequential
from keras.engine.input_layer import InputLayer
from keras.layers import Dense, Flatten
from tensorflow import TensorSpec

from mlbpestimation.models.basemodel import BloodPressureModel
from mlbpestimation.models.metricreducer.base import MetricReducer
from mlbpestimation.models.metricreducer.singlestep import SingleStep


class MLP(BloodPressureModel):
    def __init__(self, n_layers, n_units, output_units, activation):
        super().__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        self.output_units = output_units
        self.activation = activation

        self.metric_reducer = SingleStep()

        self.input_layer = None
        self.flatten = Flatten()
        self.dense_layers = Sequential()
        for neuron_count in range(n_layers):
            self.dense_layers.add(Dense(n_units, activation=self.activation))
        self.output_layer = None

    def set_input(self, input_spec: TensorSpec):
        self.input_layer = InputLayer(input_spec[0].shape[1:])

    def set_output(self, output_spec: TensorSpec):
        output_units = output_spec.shape[1]
        self.output_layer = Dense(output_units)

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        x = self.flatten(x)
        x = self.dense_layers(x, training, mask)
        return self.output_layer(x)

    def get_metric_reducer_strategy(self) -> MetricReducer:
        return self.metric_reducer

    def get_config(self):
        return {
            'n_layers': self.n_layers,
            'n_units': self.n_units,
            'output_units': self.output_units,
            'activation': self.activation,
        }
