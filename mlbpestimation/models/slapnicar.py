import tensorflow
from keras.engine.input_layer import InputLayer
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers import Add, Conv1D, Dense, GRU, ReLU
from tensorflow.python.keras.models import Sequential
from tensorflow.python.layers.pooling import AveragePooling1D

from mlbpestimation.models.basemodel import BloodPressureModel


class Slapnicar(BloodPressureModel):
    def __init__(self):
        super().__init__()
        self._input_layer = None
        self._resnet_blocks = ResNetBlock()
        self._regressor = Regressor()

    def call(self, inputs, training=None, mask=None):
        x = self._input_layer(inputs)
        x = self._resnet_blocks(x)
        return self._regressor(x)

    def set_input_shape(self, dataset_spec):
        self._input_layer = InputLayer(dataset_spec[0].shape[1:])


class Regressor(Layer):
    def __init__(self):
        super().__init__()
        self._layers = Sequential([
            GRU(),
            Dense(),
            Dense(),
            # TODO another Dense for output?
        ])

    def call(self, inputs, training=None, mask=None):
        pass


class ResNetBlock(Layer):
    def __init__(self):
        super().__init__()
        self._conv_blocks = Sequential([
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
        ])
        self._shortcut = Sequential([
            Conv1D(),
            tensorflow.keras.layers.BatchNormalization(),
        ])
        self._add_outputs = Sequential([
            Add(),
            AveragePooling1D(),
        ])

    def call(self, inputs, training=None, mask=None):
        conv_output = self._conv_blocks(inputs)
        shortcut_output = self._shortcut(inputs)
        return self._add_outputs([conv_output, shortcut_output])


class ConvBlock(Layer):
    def __init__(self):
        super().__init__()
        self._layers = Sequential([
            Conv1D(),
            tensorflow.keras.layers.BatchNormalization(),
            ReLU(),
        ])

    def call(self, inputs, training=None, mask=None):
        return self._layers(inputs, training, mask)
