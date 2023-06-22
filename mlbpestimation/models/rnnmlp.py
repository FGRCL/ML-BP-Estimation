from typing import Tuple

from keras import Sequential
from keras.engine.base_layer import Layer
from keras.engine.input_layer import InputLayer
from keras.layers import Dense, GRU, LSTM, Reshape
from tensorflow import TensorSpec, reduce_prod

from mlbpestimation.models.basemodel import BloodPressureModel


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
        self._layers.add(Dense(output_units))

    def call(self, inputs, training=None, mask=None):
        x = self._input_layer(inputs, training, mask)
        x = self._reshape(x)
        return self._layers(x, training, mask)

    def set_input_shape(self, dataset_spec: Tuple[TensorSpec]):
        input_shape = dataset_spec[0].shape
        input_type = dataset_spec[0].dtype
        self._input_layer = InputLayer(input_shape[1:], input_shape[0], input_type)

        feature_size = reduce_prod(input_shape[2:]).numpy()
        self._reshape = Reshape((input_shape[1], feature_size))

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

    def get_config(self):
        return {

        }


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
