from keras import Sequential
from keras.engine.input_layer import InputLayer
from keras.layers import Dense, Flatten

from mlbpestimation.models.basemodel import BloodPressureModel


class MLP(BloodPressureModel):
    def __init__(self, n_layers, n_units, output_units, activation):
        super().__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        self.output_units = output_units
        self.activation = activation

        self.input_layer = None
        self.flatten = Flatten()
        self.dense_layers = Sequential()
        for neuron_count in range(n_layers):
            self.dense_layers.add(Dense(n_units, activation=self.activation))
        self.output_layer = Dense(output_units)

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        x = self.flatten(x)
        x = self.dense_layers(x, training, mask)
        return self.output_layer(x)

    def set_input_shape(self, dataset_spec):
        self.input_layer = InputLayer(dataset_spec[0].shape[1:])

    def get_config(self):
        return {
            'n_layers': self.n_layers,
            'n_units': self.n_units,
            'output_units': self.output_units,
            'activation': self.activation,
        }
