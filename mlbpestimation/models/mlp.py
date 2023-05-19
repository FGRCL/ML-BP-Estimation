from tensorflow import reshape
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers import Dense


class MLP(Model):
    def __init__(self, neurons, *args, **kwargs):  # TODO activations
        super().__init__(*args, **kwargs)
        self._layers = Sequential()
        self._layers.add(InputLayer(504))
        for neuron_count in neurons:
            self._layers.add(Dense(neuron_count, use_bias=False))

    def call(self, inputs, training=None, mask=None):
        inputs_reshaped = reshape(inputs, [*inputs.shape[:-1]])
        return self._layers(inputs_reshaped, training, mask)
