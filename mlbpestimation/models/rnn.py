from keras import Sequential
from keras.activations import relu
from keras.engine.input_layer import InputLayer
from keras.layers import Bidirectional, GRU, LSTM, SimpleRNN
from tensorflow import TensorSpec

from mlbpestimation.models.basemodel import BloodPressureModel
from mlbpestimation.models.metricreducer.base import MetricReducer
from mlbpestimation.models.metricreducer.mutlistep import MultiStep


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

        self.metric_reducer = MultiStep()

        self._input_layer = None
        self._hidden = Sequential()
        for _ in range(2):
            self._hidden.add(
                Bidirectional(GRU(128, return_sequences=True, activation=relu))
            )
        self._out = None

    def set_input(self, input_spec: TensorSpec):
        shape = input_spec[0].shape
        dtype = input_spec[0].dtype
        self._input_layer = InputLayer(shape[1:], shape[0], dtype)

    def set_output(self, output_spec: TensorSpec):
        shape = output_spec.shape
        self._out = GRU(shape[-1], return_sequences=True, activation=None)

    def call(self, inputs, training=None, mask=None):
        x = self._input_layer(inputs)
        x = self._hidden(x)
        return self._out(x)

    def get_metric_reducer_strategy(self) -> MetricReducer:
        return self.metric_reducer

    def get_config(self):
        return {
            'units': self.n_units,
            'layers': self.n_layers,
            'rnn_implementation': self.rnn_implementation,
            'output_size': self.output_size,
        }
