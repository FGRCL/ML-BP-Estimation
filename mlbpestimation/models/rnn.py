from typing import Tuple

from keras import Sequential
from keras.engine.input_layer import InputLayer
from keras.layers import GRU, LSTM, SimpleRNN
from tensorflow import TensorSpec

from mlbpestimation.models.basemodel import BloodPressureModel


class Rnn(BloodPressureModel):
    _rnn_implementations = {
        'GRU': GRU,
        'LSTM': LSTM,
        'RNN': SimpleRNN,
    }

    def __init__(self, n_units: int, n_layers: int, rnn_implementation: str, output_size: int):
        super().__init__()
        self.n_units = n_units
        self.n_layers = n_layers
        self.rnn_implementation = rnn_implementation
        self.output_size = output_size

        self._input_layer = None
        rnn = self._rnn_implementations[rnn_implementation.upper()]
        self._rnn = Sequential()
        for _ in range(n_layers):
            self._rnn.add(rnn(n_units, return_sequences=True))
        self._output = rnn(output_size, return_sequences=True)

    def call(self, inputs, training=None, mask=None):
        x = self._input_layer(inputs, training, mask)
        x = self._rnn(x, training=training, mask=mask)
        return self._output(x, training=training, mask=mask)

    def set_input_shape(self, dataset_spec: Tuple[TensorSpec]):
        input_shape = dataset_spec[0].shape
        input_type = dataset_spec[0].dtype
        self._input_layer = InputLayer(input_shape[1:], input_shape[0], input_type)

    def get_config(self):
        return {
            'units': self.n_units,
            'layers': self.n_layers,
            'rnn_implementation': self.rnn_implementation,
            'output_size': self.output_size,
        }
