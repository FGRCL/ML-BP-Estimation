from keras import Sequential
from keras.engine.base_layer import Layer
from keras.engine.input_layer import InputLayer
from keras.layers import Dense, GRU, LSTM

from mlbpestimation.models.basemodel import BloodPressureModel


class RnnMlp(BloodPressureModel):
    def __init__(self, rnn_first: bool, rnn_layers: int, rnn_units: int, rnn_type: str, mlp_layers, mlp_units: int, activation: str, output_units: int):
        super().__init__()

        self._input_layer = None
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
        self._layers.add(Dense(output_units))

    def call(self, inputs, training=None, mask=None):
        x = self._input_layer(inputs, training, mask)
        return self._layers(x, training, mask)

    def set_input_shape(self, dataset_spec):
        self._input_layer = InputLayer(dataset_spec[0].shape[1:])


class RnnModule(Layer):
    _rnn_implementations = {
        'GRU': GRU,
        'LSTM': LSTM,
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
