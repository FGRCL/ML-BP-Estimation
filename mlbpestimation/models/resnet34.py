import tensorflow
from keras import Sequential
from keras.engine.base_layer import Layer
from keras.engine.input_layer import InputLayer
from keras.layers import Add, AveragePooling1D, AveragePooling2D, BatchNormalization, Conv1D, Conv2D, Dense, Dropout, Flatten, MaxPooling1D, MaxPooling2D, ReLU
from keras.regularizers import L2
from numpy import arange, concatenate, full, log2
from tensorflow import TensorSpec

from mlbpestimation.models.basemodel import BloodPressureModel
from mlbpestimation.models.metricreducer.base import MetricReducer
from mlbpestimation.models.metricreducer.singlestep import SingleStep


class ResNet34(BloodPressureModel):
    def __init__(self,
                 start_filters: int,
                 max_filters: int,
                 first_octave_modules: int,
                 second_octave_modules: int,
                 third_octave_modules: int,
                 fourth_octave_modules: int,
                 regressor_layers: int,
                 regressor_units: int,
                 regressor_dropout: float,
                 regressor_activation: str,
                 l2_lambda: float,
                 output_units: int):
        super().__init__()
        self.start_filters = start_filters
        self.max_filters = max_filters
        self.first_octave_modules = first_octave_modules
        self.second_octave_modules = second_octave_modules
        self.third_octave_modules = third_octave_modules
        self.fourth_octave_modules = fourth_octave_modules
        self.regressor_layers = regressor_layers
        self.regressor_units = regressor_units
        self.regressor_dropout = regressor_dropout
        self.regressor_activation = regressor_activation
        self.l2_lambda = l2_lambda
        self.output_units = output_units

        self.metric_reducer = SingleStep()

        self._output = None
        self._regressor = None
        self._flatten = None
        self._avg_pool = None
        self._octaves = None
        self._initial_conv = None
        self._input_layer = None

    def set_input(self, input_spec: TensorSpec):
        self._input_layer = InputLayer(input_spec[0].shape[1:])
        use_2d = input_spec[0].shape[-1] != 1
        print(use_2d)

        self._initial_conv = InitialConv(use_2d)

        self._octaves = Sequential()
        filters = self._compute_filters(self.start_filters, self.max_filters)
        octave_modules = [self.first_octave_modules, self.second_octave_modules, self.third_octave_modules, self.fourth_octave_modules]
        for n_residual_block, n_filter in zip(octave_modules, filters):
            self._octaves.add(ResidualOctave(n_filter, n_residual_block, use_2d))

        if use_2d:
            self._avg_pool = AveragePooling2D()
        else:
            self._avg_pool = AveragePooling1D()
        self._flatten = Flatten()
        self._regressor = Sequential()
        for _ in range(self.regressor_layers):
            self._regressor.add(Dense(self.regressor_units, self.regressor_activation, kernel_regularizer=L2(self.l2_lambda)))
            self._regressor.add(Dropout(self.regressor_dropout))

    def set_output(self, output_spec: TensorSpec):
        output_units = output_spec.shape[1]
        self._output = Dense(output_units)

    def call(self, inputs, training=None, mask=None):
        x = self._initial_conv(inputs)
        x = self._octaves(x)
        x = self._flatten(x)
        x = self._regressor(x)
        return self._output(x)

    def get_metric_reducer_strategy(self) -> MetricReducer:
        return self.metric_reducer

    def get_config(self):
        return {
            'start_filters': self.start_filters,
            'max_filters': self.max_filters,
            'first_octave_modules': self.first_octave_modules,
            'second_octave_modules': self.second_octave_modules,
            'third_octave_modules': self.third_octave_modules,
            'fourth_octave_modules': self.fourth_octave_modules,
            'regressor_layers': self.regressor_layers,
            'regressor_units': self.regressor_units,
            'regressor_dropout': self.regressor_dropout,
            'regressor_activation': self.regressor_activation,
            'l2_lambda': self.l2_lambda,
            'output_units': self.output_units,
        }

    @staticmethod
    def _compute_filters(start_filter, max_filter):
        filters = arange(log2(start_filter), log2(max_filter) + 1, dtype=int)
        n_leftover_filters = len(filters) - 4
        if n_leftover_filters > 0:
            leftover_filters = full(n_leftover_filters, log2(max_filter), dtype=int)
            filters = concatenate((filters, leftover_filters))
        if n_leftover_filters < 0:
            filters = filters[:4]

        return 2 ** filters.astype(int)


class InitialConv(Layer):
    def __init__(self, use_2d):
        super().__init__()
        self.use_2d = use_2d

        if use_2d:
            self._conv = Conv2D(64, 7, 2)
            self._pool = MaxPooling2D(2, 2)  # reduced pool_size from 3 to 2, because one of our dimension has only 10
        else:
            self._conv = Conv1D(64, 7, 2)
            self._pool = MaxPooling1D(3, 2)

        self._batch_norm = BatchNormalization()
        self._activation = ReLU()

    def call(self, inputs, *args, **kwargs):
        if self.use_2d:
            x = inputs[:, :, :, tensorflow.newaxis]
        else:
            x = inputs
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._activation(x)
        x = self._pool(x)
        return x


class ResidualOctave(Layer):
    def __init__(self, n_filter, n_block, use_2d):
        super().__init__()

        self.octave = Sequential([
            ExpandResidualBlock(n_filter, use_2d)
        ])
        for _ in range(n_block - 1):
            self.octave.add(ResidualBlock(n_filter, use_2d))

    def call(self, inputs, training=None, mask=None):
        return self.octave(inputs)


class ExpandResidualBlock(Layer):
    def __init__(self, n_filter, use_2d):
        super().__init__()

        self.shortcut = ConvBlock(n_filter, 1, use_2d, stride=2, activation=False)
        self.conv = Sequential([
            ConvBlock(n_filter, 3, use_2d, stride=2),
            ConvBlock(n_filter, 3, use_2d, activation=False)
        ])
        self.add = Add()
        self.relu = ReLU()

    def call(self, inputs, training=None, mask=None):
        residual = self.shortcut(inputs)
        features = self.conv(inputs)
        x = self.add([residual, features])
        x = self.relu(x)
        return x


class ResidualBlock(Layer):
    def __init__(self, n_filter, use_2d):
        super().__init__()

        self.conv = Sequential([
            ConvBlock(n_filter, 3, use_2d),
            ConvBlock(n_filter, 3, use_2d, activation=False)
        ])
        self.add = Add()
        self.relu = ReLU()

    def call(self, inputs, training=None, mask=None):
        features = self.conv(inputs)
        x = self.add([features, inputs])
        x = self.relu(x)
        return x


class ConvBlock(Layer):
    def __init__(self, n_filter, kernel_size, use_2d, stride=1, activation=True):
        super().__init__()

        self.block = Sequential()
        if use_2d:
            self.block.add(
                Conv2D(n_filter, kernel_size, strides=stride, padding='same', activation=None)
            )
        else:
            self.block.add(
                Conv1D(n_filter, kernel_size, strides=stride, padding='same', activation=None)
            )
        self.block.add(BatchNormalization())
        if activation:
            self.block.add(ReLU())

    def call(self, inputs, training=None, mask=None):
        return self.block(inputs)
