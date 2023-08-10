from keras import Sequential
from keras.activations import selu
from keras.engine.input_layer import InputLayer
from keras.initializers.initializers import LecunNormal, Zeros
from keras.layers import AlphaDropout, Dense, Flatten

from mlbpestimation.models.basemodel import BloodPressureModel


class Snnn(BloodPressureModel):
    def __init__(self, n_layers: int, n_units: int, output_units: int):
        super().__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        self.output_units = output_units

        self.input_layer = None
        self.flatten = Flatten()
        self.dense_layers = Sequential()
        for neuron_count in range(n_layers):
            self.dense_layers.add(
                Dense(n_units, activation=selu, kernel_initializer=LecunNormal(), bias_initializer=Zeros()))
            self.dense_layers.add(
                AlphaDropout(0.05)
            )
        self.output_layer = Dense(output_units, kernel_initializer=LecunNormal(), bias_initializer=Zeros())

    def set_input_shape(self, dataset_spec):
        self.input_layer = InputLayer(dataset_spec[0].shape[1:])

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        x = self.flatten(x)
        x = self.dense_layers(x, training, mask)
        x = self.output_layer(x)
        return x

    def get_config(self):
        return {
            'n_layers': self.n_layers,
            'n_units': self.n_units,
            'output_units': self.output_units,
        }
