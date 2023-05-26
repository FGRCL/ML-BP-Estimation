from keras import Sequential
from keras.engine.base_layer import Layer
from keras.layers import Add, BatchNormalization, Conv1D, Dense, Dropout, Flatten, ReLU
from keras.regularizers import L2
from omegaconf import ListConfig

from mlbpestimation.models.basemodel import BloodPressureModel


class ResNet(BloodPressureModel):
    def __init__(self, n_residual_blocks: ListConfig, n_filters: ListConfig):
        super().__init__()
        self.n_residual_blocks = list(n_residual_blocks)
        self.n_filters = list(n_filters)

        self.octaves = Sequential()
        for n_residual_block, n_filter in zip(self.n_residual_blocks, self.n_filters):
            self.octaves.add(ResidualOctave(n_filter, n_residual_block))
        self.regressor = Sequential([
            Flatten(),
            Dense(32, 'relu', kernel_regularizer=L2(.001)),
            Dropout(0.25),
            Dense(32, 'relu', kernel_regularizer=L2(.001)),
            Dropout(0.25),
            Dense(2, 'relu')
        ])

    def call(self, inputs, training=None, mask=None):
        x = self.octaves(inputs)
        x = self.regressor(x)
        return x

    def set_input_shape(self, dataset_spec):
        pass

    def get_config(self):
        return {
            'n_residual_blocks': self.n_residual_blocks,
            'n_filters': self.n_filters
        }


class ResidualOctave(Layer):
    def __init__(self, n_filter, n_block):
        super().__init__()

        self.octave = Sequential([
            ExpandResidualBlock(n_filter)
        ])
        for _ in range(n_block - 1):
            self.octave.add(ResidualBlock(n_filter))

    def call(self, inputs, training=None, mask=None):
        return self.octave(inputs)


class ExpandResidualBlock(Layer):
    def __init__(self, n_filter):
        super().__init__()

        self.shortcut = ConvBlock(n_filter, 1, stride=2, activation=False)
        self.conv = Sequential([
            ConvBlock(n_filter, 3, stride=2),
            ConvBlock(n_filter, 3),
            ConvBlock(n_filter, 3, activation=False)
        ])
        self.add = Add()
        self.activation = ReLU()

    def call(self, inputs, training=None, mask=None):
        residual = self.shortcut(inputs)
        features = self.conv(inputs)
        x = self.add([residual, features])
        x = self.activation(x)
        return x


class ResidualBlock(Layer):
    def __init__(self, n_filter):
        super().__init__()

        self.conv = Sequential([
            ConvBlock(n_filter, 3),
            ConvBlock(n_filter, 3),
            ConvBlock(n_filter, 3, activation=False)
        ])
        self.add = Add()
        self.activation = ReLU()

    def call(self, inputs, training=None, mask=None):
        features = self.conv(inputs)
        x = self.add([features, inputs])
        x = self.activation(x)
        return x


class ConvBlock(Layer):
    def __init__(self, n_filter, kernel_size, stride=1, activation=True):
        super().__init__()

        self.block = Sequential([
            Conv1D(n_filter, kernel_size, strides=stride, padding="same"),
            BatchNormalization(),
        ])
        if activation:
            self.block.add(ReLU())

    def call(self, inputs, training=None, mask=None):
        return self.block(inputs)
