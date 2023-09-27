from keras import Sequential
from keras.engine.base_layer import Layer
from keras.engine.input_layer import InputLayer
from keras.layers import Dense, GRU, LSTM, Reshape, SimpleRNN
from tensorflow import TensorSpec, reduce_prod

from mlbpestimation.models.basemodel import BloodPressureModel
from mlbpestimation.models.metricreducer.base import MetricReducer
from mlbpestimation.models.metricreducer.singlestep import SingleStep


class RnnMlp(BloodPressureModel):
    def __init__(self, rnn_first: bool, rnn_layers: int, rnn_units: int, rnn_type: str, mlp_layers, mlp_units: int, activation: str, output_units: int):
        super().__init__()
        self.rnn_first = rnn_first
        self.rnn_layers = rnn_layers
        self.rnn_units = rnn_units
        self.rnn_type = rnn_type
        self.mlp_layers = mlp_layers
        self.activation = activation
        self.output_units = output_units

        self.metric_reducer = SingleStep()

        self._input_layer = None
        self._reshape = None
        if rnn_first:
            self._layers = Sequential([
                RnnModule(rnn_layers, rnn_units, rnn_type),
                MlpModule(mlp_layers, mlp_units, activation),
            ])
        else:
            self._layers = Sequential([
                MlpModule(mlp_layers, mlp_units, activation),
                RnnModule(rnn_layers, rnn_units, rnn_type),
            ])
        self._output = None

    def set_input(self, input_spec: TensorSpec):
        shape = input_spec[0].shape
        dtype = input_spec[0].dtype
        self._input_layer = InputLayer(shape[1:], shape[0], dtype)

        feature_size = reduce_prod(shape[2:]).numpy()  # TODO check if we still need this reshape
        self._reshape = Reshape((shape[1], feature_size))

    def set_output(self, output_spec: TensorSpec):
        output_units = output_spec.shape[1]
        self._output = Dense(output_units)

    def call(self, inputs, training=None, mask=None):
        x = self._input_layer(inputs, training, mask)
        x = self._reshape(x)
        x = self._layers(x, training, mask)
        x = self._output(x)
        return x

    def get_metric_reducer_strategy(self) -> MetricReducer:
        return self.metric_reducer

    def get_config(self):
        return {
            'rnn_first': self.rnn_first,
            'rnn_layers': self.rnn_layers,
            'rnn_units': self.rnn_units,
            'rnn_type': self.rnn_type,
            'mlp_layer': self.mlp_layers,
            'activation': self.activation,
            'output_units': self.output_units,
        }


class RnnModule(Layer):
    _rnn_implementations = {
        'GRU': GRU,
        'LSTM': LSTM,
        'RNN': SimpleRNN,
    }

    def __init__(self, n_layers: int, n_units: int, rnn_implementation: str, **kwargs):
        super().__init__(**kwargs)

        try:
            rnn = self._rnn_implementations[rnn_implementation.upper()]
        except KeyError as e:
            raise UnknownRnnImplementationException(self._rnn_implementations.keys()).with_traceback(e.__traceback__) from e

        self._layers = Sequential()
        for _ in range(n_layers - 1):
            self._layers.add(
                rnn(n_units, return_sequences=True)
            )
        self._layers.add(rnn(n_units))

    def call(self, inputs, *args, **kwargs):
        return self._layers(inputs, *args, **kwargs)


class MlpModule(Layer):
    def __init__(self, n_layers, n_units, activation, **kwargs):
        super().__init__(**kwargs)

        self._layers = Sequential()
        for _ in range(n_layers):
            self._layers.add(
                Dense(n_units, activation)
            )

    def call(self, inputs, *args, **kwargs):
        return self._layers(inputs, *args, **kwargs)


class UnknownRnnImplementationException(BaseException):
    def __init__(self, implementations):
        super().__init__(f'Unknow RNN implementation should be one of: {implementations}')
