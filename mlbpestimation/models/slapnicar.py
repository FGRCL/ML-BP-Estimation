from typing import List

from kapre.time_frequency import Magnitude, STFT
from keras import Sequential
from keras.engine.base_layer import Layer
from keras.engine.input_layer import InputLayer
from keras.layers import Add, AveragePooling1D, BatchNormalization, Concatenate, Conv1D, Dense, Dropout, GRU, ReLU, Reshape
from keras.regularizers import l2
from numpy import arange, concatenate, full, log2
from omegaconf import ListConfig

from mlbpestimation.models.basemodel import BloodPressureModel


class Slapnicar(BloodPressureModel):
    def __init__(self,
                 start_filters: int,
                 max_filters: int,
                 resnet_blocks: int,
                 resnet_block_kernels: ListConfig,
                 pool_size: int,
                 pool_stride: int,
                 gru_units: int,
                 dense_units: int,
                 output_units: int,
                 l2_lambda: float,
                 dropout_rate: float):
        super().__init__()
        self.start_filters = start_filters
        self.max_filters = max_filters
        self.resnet_blocks = resnet_blocks
        self.resnet_block_kernels: List[int] = list(resnet_block_kernels)
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.gru_units = gru_units
        self.dense_units = dense_units
        self.output_units = output_units
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate

        resnet_filters = self._compute_resnet_filters(start_filters, max_filters, resnet_blocks)
        self._input_layer = None
        self._resnet_blocks = Sequential()
        for resnet_filter in resnet_filters:
            self._resnet_blocks.add(
                ResNetBlock(resnet_filter, self.resnet_block_kernels, pool_size, pool_stride)
            )
        self._spectro_temporal_block = SpectroTemporalBlock()
        self._gru_block = Sequential([
            GRU(gru_units),
            BatchNormalization(),
        ])
        self._concatenate = Concatenate()
        self._regressor = Regressor(dense_units, output_units, l2_lambda, dropout_rate)

    def call(self, inputs, training=None, mask=None):
        x = self._input_layer(inputs, training, mask)
        resnet_output = self._resnet_blocks(x, training, mask)
        gru_output = self._gru_block(resnet_output)
        spectrotemporal_output = self._spectro_temporal_block(x, training, mask)
        x = self._concatenate([gru_output, spectrotemporal_output])
        return self._regressor(x, training, mask)

    def get_config(self):
        return {
            'start_filters': self.start_filters,
            'max_filters': self.max_filters,
            'resnet_blocks': self.resnet_blocks,
            'resnet_block_kernels': self.resnet_block_kernels,
            'pool_size': self.pool_size,
            'pool_stride': self.pool_stride,
            'gru_units': self.gru_units,
            'dense_units': self.dense_units,
            'output_units': self.output_units,
            'l2_lambda': self.l2_lambda,
            'dropout_rate': self.dropout_rate,
        }

    def set_input_shape(self, dataset_spec):
        self._input_layer = InputLayer(dataset_spec[0].shape[1:])

    @staticmethod
    def _compute_resnet_filters(start_filters, max_filters, resnet_blocks):
        filters = arange(log2(start_filters), log2(max_filters) + 1, dtype=int)
        remaining_blocks = resnet_blocks - filters.size
        if remaining_blocks > 0:
            remainder_filters = full(remaining_blocks, log2(max_filters), dtype=int)
            filters = concatenate((filters, remainder_filters))
        elif remaining_blocks < 0:
            filters = filters[:-remaining_blocks]

        return 2 ** filters.astype(int)


class Regressor(Layer):
    def __init__(self, dense_units: int, output_units: int, l2_lambda: float, dropout_rate: float):
        super().__init__()
        self._layers = Sequential([
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
            BatchNormalization(),
        ])
        self._add = Add()
        self._average_pooling = AveragePooling1D(pool_size, pool_stride)

    def call(self, inputs, training=None, mask=None):
        conv_output = self._conv_blocks(inputs, training, mask)
        shortcut_output = self._shortcut(inputs, training, mask)
        x = self._add([conv_output, shortcut_output])
        return self._average_pooling(x)


class ConvBlock(Layer):
    def __init__(self, filters: int, kernel: int):
        super().__init__()
        self._layers = Sequential([
            Conv1D(filters, kernel, padding='same'),
            BatchNormalization(),
            ReLU(),
        ])

    def call(self, inputs, training=None, mask=None):
        return self._layers(inputs, training, mask)


class SpectroTemporalBlock(Layer):
    def __init__(self):
        super().__init__()
        self._spectrogram = Sequential([
            STFT(
                n_fft=128,
                input_data_format="channels_last",
                output_data_format="channels_last",
            ),
            Magnitude()
        ])
        self._temporal = Sequential([
            GRU(64),
            BatchNormalization(),
        ])

    def call(self, inputs, training=None, mask=None):
        x = self._spectrogram(inputs, training, mask)
        x = Reshape((*x.shape[1:2], -1))(x)
        return self._temporal(x, training, mask)
