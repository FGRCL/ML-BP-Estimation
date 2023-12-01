from keras import Sequential
from keras.activations import gelu
from keras.engine.base_layer import Layer
from keras.layers import Add, Attention, BatchNormalization, Concatenate, Conv1D, Dense, Flatten, LayerNormalization, MultiHeadAttention, ReLU, UpSampling1D
from keras_nlp.src.layers import SinePositionEncoding
from tensorflow import TensorSpec

from mlbpestimation.models.basemodel import BloodPressureModel
from mlbpestimation.models.metricreducer.base import MetricReducer
from mlbpestimation.models.metricreducer.singlestep import SingleStep


class Kim(BloodPressureModel):
    def __init__(self):
        super().__init__()
        self.metric_reducer = SingleStep()

        self.contraction_path = [
            InitialContractionBlock(32),
            ContractionBlock(64),
            ContractionBlock(128),
            ContractionBlock(256),
            ContractionBlock(512),
        ]

        self.initial_expansion = InitialExpansionBlock(512)

        self.expansion_path = [
            ExpansionBlock(256),
            ExpansionBlock(128),
            ExpansionBlock(64),
            ExpansionBlock(32),
        ]

        self.skip = [
            SkipConnection(),
            SkipConnection(),
            SkipConnection(),
            SkipConnection(),
        ]

        self.self_attention = SelfAttention(32, 32)
        self.flatten = Flatten()
        self.regressor = Dense(2)

    def call(self, inputs, training=None, mask=None):
        x = inputs

        expansion_outputs = []
        for block in self.contraction_path:
            x = block(x)
            expansion_outputs.append(x)

        x = self.initial_expansion(x)

        for expansion_features, skip, block in zip(expansion_outputs[-2::-1], self.skip, self.expansion_path):
            skip_features = skip(expansion_features)
            x = block(x, skip_features)

        x = self.self_attention(x)
        x = self.flatten(x)
        x = self.regressor(x)

        return x

    def set_input(self, input_spec: TensorSpec):
        pass

    def set_output(self, output_spec: TensorSpec):
        pass

    def get_metric_reducer_strategy(self) -> MetricReducer:
        return self.metric_reducer


class InitialContractionBlock(Layer):
    def __init__(self, filters: int):
        super().__init__()
        self.main = Sequential([
            Conv1D(filters, 3, strides=1, padding="same"),
            BatchNormalization(),
            ReLU(),
            Conv1D(filters, 3, strides=1, padding="same")
        ])
        self.shortcut = ShortcutBlock(32, 1)
        self.addition = Add()

    def call(self, inputs, *args, **kwargs):
        x = self.main(inputs)
        shortcut = self.shortcut(inputs)
        x = self.addition([x, shortcut])
        return x


class ContractionBlock(Layer):
    def __init__(self, filters: int):
        super().__init__()

        self.shortcut = ShortcutBlock(filters, 2)
        self.main = ConvBlock(filters, 2)
        self.addition = Add()

    def call(self, inputs, *args, **kwargs):
        x = self.main(inputs)
        shortcut = self.shortcut(inputs)
        x = self.addition([x, shortcut])
        return x


class InitialExpansionBlock(Layer):
    def __init__(self, filters: int):
        super().__init__()
        self.conv = ConvBlock(filters, 1)

    def call(self, inputs, *args, **kwargs):
        return self.conv(inputs)


class ExpansionBlock(Layer):
    def __init__(self, filters: int):
        super().__init__()

        self.shortcut = ShortcutBlock(filters, 1)
        self.upsample = UpSampling1D()
        self.concatenate = Concatenate()
        self.main = ConvBlock(filters, 1)
        self.addition = Add()

    def call(self, inputs, contraction_features):
        x = self.upsample(inputs)
        x = x[:, :contraction_features.shape[1]]
        x = self.concatenate([x, contraction_features])
        shortcut = self.shortcut(x)
        x = self.main(x)
        x = self.addition([x, shortcut])
        return x


class ConvBlock(Layer):
    def __init__(self, filters: int, strides: int):
        super().__init__()

        self.sequential = Sequential([
            BatchNormalization(),
            ReLU(),
            Conv1D(filters, 3, strides=strides, padding="same"),
            BatchNormalization(),
            ReLU(),
            Conv1D(filters, 3, strides=1, padding="same")
        ])

    def call(self, inputs):
        return self.sequential(inputs)


class ShortcutBlock(Layer):
    def __init__(self, filters: int, strides: int):
        super().__init__()
        self.shortcut = Sequential([
            Conv1D(filters, 3, strides=strides, padding="same"),
            BatchNormalization()
        ])

    def call(self, inputs, *args, **kwargs):
        return self.shortcut(inputs)


class SkipConnection(Layer):
    def __init__(self):
        super().__init__()
        self.attention = Attention(use_scale=True)

    def call(self, inputs, *args, **kwargs):
        return self.attention([inputs, inputs])


class SelfAttention(Layer):
    def __init__(self, key_dim: int, units: int):
        super().__init__()
        self.sequential = Sequential([
            AttentionBlock(key_dim, units),
            AttentionBlock(key_dim, units),
        ])

    def call(self, inputs, *args, **kwargs):
        return self.sequential(inputs)


class AttentionBlock(Layer):
    def __init__(self, key_dim: int, units: int):
        super().__init__()

        self.positionalencoding = SinePositionEncoding()
        self.multihead = MultiHeadAttention(1, key_dim)
        self.dense1 = Dense(units, activation=gelu)
        self.dense2 = Dense(units)
        self.layernorm = LayerNormalization()
        self.addition = Add()

    def call(self, inputs, *args, **kwargs):
        x = self.positionalencoding(inputs)
        x = self.multihead(x, x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.layernorm(x)
        x = self.addition([x, inputs])
        return x
