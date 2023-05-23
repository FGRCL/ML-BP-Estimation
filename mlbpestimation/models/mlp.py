from tensorflow import reshape
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers import Dense


class MLP(Model):
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

    def set_input_output_shape(self, input_output_shape):
        input_shape, output_shape = input_output_shape
        self.input_layer = InputLayer(input_shape[1])
