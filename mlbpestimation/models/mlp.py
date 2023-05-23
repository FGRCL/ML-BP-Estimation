from tensorflow import reshape
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers import Dense

from mlbpestimation.models.basemodel import BloodPressureModel


class MLP(BloodPressureModel):
    def __init__(self, neurons, *args, **kwargs):  # TODO activations
        super().__init__(*args, **kwargs)
        self.input_layer = None
        self.dense_layers = Sequential()
        for neuron_count in neurons:
            self.dense_layers.add(Dense(neuron_count, use_bias=False))

    def call(self, inputs, training=None, mask=None):
        x = reshape(inputs, [*inputs.shape[:-1]])
        x = self.input_layer(x)
        return self.dense_layers(x, training, mask)

    def set_input_shape(self, dataset_spec):
        input_spec = dataset_spec[0]
        self.input_layer = InputLayer(input_spec.shape[1])
