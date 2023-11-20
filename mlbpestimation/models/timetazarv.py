import tensorflow
from keras.activations import relu
from keras.engine.input_layer import InputLayer
from keras.layers import BatchNormalization, Bidirectional, Conv1D, Dense, Dropout, Flatten, GRU, LSTM, MaxPooling1D, ReLU, SimpleRNN, TimeDistributed
from tensorflow import TensorSpec

from mlbpestimation.models.basemodel import BloodPressureModel
from mlbpestimation.models.metricreducer.base import MetricReducer
from mlbpestimation.models.metricreducer.mutlistep import MultiStep
from mlbpestimation.models.metricreducer.singlestep import SingleStep


class TimeTazarv(BloodPressureModel):
    _rnn_implementations = {
        'GRU': GRU,
        'LSTM': LSTM,
        'RNN': SimpleRNN,
    }

    def __init__(self, n_units: int, n_layers: int, dropout: float, recurrent_dropout: float, output_size: int):
        super().__init__()
        self.n_units = n_units
        self.n_layers = n_layers
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.output_size = output_size

        self.metric_reducer = None

        self._input_layer = None
        self._layers = [
            TimeDistributed(Conv1D(32, 5)),
            TimeDistributed(ReLU()),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(MaxPooling1D(4)),
            TimeDistributed(Dropout(0.01)),
            TimeDistributed(Flatten()),
            TimeDistributed(Dense(256, activation=relu)),
            TimeDistributed(Dense(32, activation=None)),
            Bidirectional(SimpleRNN(256, return_sequences=True, activation=relu, unroll=True)),
        ]
        self._output_layer = None

    def set_input(self, input_spec: TensorSpec):
        shape = input_spec[0].shape
        dtype = input_spec[0].dtype
        self._input_layer = InputLayer(shape[1:], shape[0], dtype)

    def set_output(self, output_spec: TensorSpec):
        shape = output_spec.shape
        if shape.ndims == 2:
            self._output_layer = SimpleRNN(shape[-1], return_sequences=False, activation=None, unroll=True)
            self.metric_reducer = SingleStep()
        elif shape.ndims == 3:
            self._output_layer = SimpleRNN(shape[-1], return_sequences=True, activation=None, unroll=True)
            self.metric_reducer = MultiStep()

    def call(self, inputs, training=None, mask=None):
        x = self._input_layer(inputs)
        x = x[:, :, :, tensorflow.newaxis]
        for layer in self._layers:
            x = layer(x)
        x = self._output_layer(x)

        return x

    def get_metric_reducer_strategy(self) -> MetricReducer:
        return self.metric_reducer

    def get_config(self):
        return {
            'units': self.n_units,
            'layers': self.n_layers,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'output_size': self.output_size,
        }
