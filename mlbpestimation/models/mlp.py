from keras import Sequential
from keras.engine.input_layer import InputLayer
from keras.layers import Dense, Flatten

from mlbpestimation.models.basemodel import BloodPressureModel


class MLP(BloodPressureModel):
    def __init__(self, neurons, activation):
        super().__init__()
        self.neurons = neurons
        self.activation = activation

        self.input_layer = None
        self.flatten = Flatten()
        self.dense_layers = Sequential()
        for neuron_count in self.neurons:
            self.dense_layers.add(Dense(neuron_count, activation=self.activation))

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        x = self.flatten(x)
        return self.dense_layers(x, training, mask)

    def set_input_shape(self, dataset_spec):
        self.input_layer = InputLayer(dataset_spec[0].shape[1:])

    def get_config(self):
        return {
            'neurons': self.neurons,
            'activation': self.activation
        }
