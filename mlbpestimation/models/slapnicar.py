from typing import List

import tensorflow
from keras.engine.input_layer import InputLayer
from keras.layers import Dropout
from numpy import arange, concatenate, full, log2
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers import Add, AveragePooling1D, Conv1D, Dense, GRU, ReLU
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.regularizers import l2

from mlbpestimation.models.basemodel import BloodPressureModel


class Slapnicar(BloodPressureModel):
    def __init__(self,
                 start_filters: int,
                 max_filters: int,
                 resnet_blocks: int,
                 resnet_block_kernels: List[int],
                 pool_size: int,
                 pool_stride: int,
                 gru_units: int,
                 dense_units: int,
                 output_units: int,
                 l2_lambda: float,
                 dropout_rate: float):
        super().__init__()
        resnet_filters = self._compute_resnet_filters(start_filters, max_filters, resnet_blocks)
        self._input_layer = None
        self._resnet_blocks = Sequential()
        for resnet_filter in resnet_filters:
            self._resnet_blocks.add(
                ResNetBlock(resnet_filter, resnet_block_kernels, pool_size, pool_stride)
            )
        self._regressor = Regressor(gru_units, dense_units, output_units, l2_lambda, dropout_rate)

    def call(self, inputs, training=None, mask=None):
        x = self._input_layer(inputs, training, mask)
        x = self._resnet_blocks(x, training, mask)
        return self._regressor(x, training, mask)

    def set_input_shape(self, dataset_spec):
        self._input_layer = InputLayer(dataset_spec[0].shape[1:])

    def _compute_resnet_filters(self, start_filters, max_filters, resnet_blocks):
        filters = arange(log2(start_filters), log2(max_filters))
        remaining_blocks = resnet_blocks - filters.size
        if remaining_blocks > 0:
            remainder_filters = full(remaining_blocks, log2(max_filters))
            filters = concatenate((filters, remainder_filters))
        elif remaining_blocks < 0:
            filters = filters[:-remaining_blocks]

        return 2 ** filters


class Regressor(Layer):
    def __init__(self, gru_units: int, dense_units: int, output_units: int, l2_lambda: float, dropout_rate: float):
        super().__init__()
        self._layers = Sequential([
            GRU(gru_units),
            tensorflow.keras.layers.BatchNormalization(),
            Dense(dense_units, kernel_regularizer=l2(l2_lambda)),
            ReLU(),
            Dropout(dropout_rate),
            Dense(dense_units, kernel_regularizer=l2(l2_lambda)),
            ReLU(),
            Dropout(dropout_rate),
            Dense(output_units, kernel_regularizer=l2(l2_lambda)),
            ReLU(),
        ])

    def call(self, inputs, training=None, mask=None):
        return self._layers(inputs, training, mask)


class ResNetBlock(Layer):
    def __init__(self, filters: int, n_kernels: List[int], pool_size: int, pool_stride: int):
        super().__init__()
        self._conv_blocks = Sequential()
        for kernel in n_kernels:
            self._conv_blocks.add(
                ConvBlock(filters, kernel)
            )
        self._shortcut = Sequential([
            Conv1D(filters, 1, padding='same'),
            tensorflow.keras.layers.BatchNormalization(),
        ])
        self._add_outputs = Sequential([
            Add(),
            AveragePooling1D(pool_size, pool_stride),
        ])

    def call(self, inputs, training=None, mask=None):
        conv_output = self._conv_blocks(inputs, training, mask)
        shortcut_output = self._shortcut(inputs, training, mask)
        return self._add_outputs([conv_output, shortcut_output], training, mask)


class ConvBlock(Layer):
    def __init__(self, filters: int, kernel: int):
        super().__init__()
        self._layers = Sequential([
            Conv1D(filters, kernel, padding='same'),
            tensorflow.keras.layers.BatchNormalization(),
            ReLU(),
        ])

    def call(self, inputs, training=None, mask=None):
        return self._layers(inputs, training, mask)
