from keras import Sequential
from keras.engine.input_layer import InputLayer
from keras.layers import BatchNormalization, Bidirectional, ConvLSTM1D, Dropout, GRU, LSTM, MaxPooling1D, ReLU, Reshape, SimpleRNN, TimeDistributed
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
        for _ in range(4):
            self._hidden.add(
                Bidirectional(ConvLSTM1D(16, 5, return_sequences=True))
            )
            self._hidden.add(
                TimeDistributed(BatchNormalization())
            )
            self._hidden.add(
                TimeDistributed(ReLU())
            )
            self._hidden.add(
                TimeDistributed(MaxPooling1D(2))
            )
            self._hidden.add(
                TimeDistributed(Dropout(0.1))
            )
        self._reshape = None
        self._out = None

    def set_input(self, input_spec: TensorSpec):
        shape = input_spec[0].shape
        dtype = input_spec[0].dtype
        self._input_layer = InputLayer(shape[1:], shape[0], dtype)
        self._reshape = Reshape((shape[1], -1))

    def set_output(self, output_spec: TensorSpec):
        shape = output_spec.shape
        self._out = GRU(shape[-1], return_sequences=True, activation=None)

    def call(self, inputs, training=None, mask=None):
        x = self._input_layer(inputs)
        x = self._hidden(x)
        x = self._reshape(x)
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
