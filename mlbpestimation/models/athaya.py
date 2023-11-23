from keras.engine.base_layer import Layer
from keras.layers import Concatenate, Conv1D, Dense, Dropout, LeakyReLU, MaxPooling1D, UpSampling1D
from tensorflow import TensorSpec

from mlbpestimation.models.basemodel import BloodPressureModel
from mlbpestimation.models.metricreducer.base import MetricReducer
from mlbpestimation.models.metricreducer.singlestep import SingleStep


class Athaya(BloodPressureModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_reducer = SingleStep()

        start_filters = 64
        n_levels = 4
        n_dropout = 2

        self.input_conv = ConvBlock(start_filters, 64)

        self.contraction_blocks = [ConvBlock(start_filters, False)]
        self.expansion_blocks = []
        current_filters = start_filters
        for i in range(n_levels):
            current_filters *= 2
            use_dropout = n_levels - i <= n_dropout
            self.contraction_blocks.append(
                Contraction(current_filters, use_dropout)
            )
            self.expansion_blocks.append(
                Expansion(current_filters)
            )

        self.regressor = Dense(2)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        contraction_outputs = []
        for contraction in self.contraction_blocks:
            x = contraction(x)
            contraction_outputs.append(x)

        for expansion, contraction_output in zip(self.expansion_blocks[::-1], contraction_outputs[-2::-1]):
            x = expansion(x, contraction_output)

        x = self.regressor(x)
        return x

    def set_input(self, input_spec: TensorSpec):
        pass

    def set_output(self, output_spec: TensorSpec):
        pass

    def get_metric_reducer_strategy(self) -> MetricReducer:
        return self.metric_reducer


class Contraction(Layer):
    def __init__(self, filters, dropout, **kwargs):
        super().__init__(**kwargs)
        self._max_pool = MaxPooling1D(2)
        self._conv_layers = ConvBlock(filters, dropout)

    def call(self, inputs):
        x = self._max_pool(inputs)
        x = self._conv_layers(x)

        return x


class Expansion(Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self._upsample = UpSampling1D(2)
        self._reduce_conv = Conv1D(filters, 2)
        self._concatenate = Concatenate()
        self._conv_layers = ConvBlock(filters, False)

    def call(self, inputs, contraction_features):
        x = self._upsample(inputs)
        x = self._reduce_conv(x)
        x = self._concatenate([x, contraction_features])
        x = self._conv_layers(x)
        return x


class ConvBlock(Layer):
    def __init__(self, filters: int, dropout: bool, **kwargs):
        super().__init__(**kwargs)
        self._layers = [
            Conv1D(filters, 3),
            LeakyReLU(),
            Conv1D(filters, 3),
            LeakyReLU(),
        ]
        if dropout:
            self._layers.append(
                Dropout(0.5)
            )

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self._layers:
            layer(x)
        return x
