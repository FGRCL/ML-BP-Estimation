from keras import Sequential
from keras.engine.base_layer import Layer
from keras.engine.input_layer import InputLayer
from keras.layers import Add, Dense, Dropout, LayerNormalization, MultiHeadAttention, ReLU

from mlbpestimation.models.basemodel import BloodPressureModel


class TransformerEncoder(BloodPressureModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._input_layer = None
        self._encoders = Sequential()
        for _ in range(n_encoder_modules):
            self._encoders.add(EncoderModule())
        self._regressor = Regressor()

    def call(self, inputs, training=None, mask=None):
        x = self._input_layer()
        x = self._encoders(inputs)
        return self._regressor(x)

    def set_input_shape(self, dataset_spec):
        self._input_layer = InputLayer(dataset_spec[0].shape[1:])

    def get_config(self):
        return {}


class Regressor(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._dense_layers = Sequential()
        for _ in range(n_regressor_layers):
            self._dense_layers.add([
                Dropout(),
                Dense(),
                ReLU()
            ])
        self._output = Dense()

    def call(self, inputs, *args, **kwargs):
        x = self._dense_layers(inputs, *args, **kwargs)
        return self._output(x)


class EncoderModule(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._global_self_attention = GlobalSelfAttentionModule()
        self._feed_forward = FeedForwardModule()

    def call(self, inputs, *args, **kwargs):
        x = self._global_self_attention(inputs, *args, **kwargs)
        return self._feed_forward(x, *args, **kwargs)


class GlobalSelfAttentionModule(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._global_self_attention = MultiHeadAttention()
        self._add = Add()
        self._layer_normalization = LayerNormalization()

    def call(self, inputs, *args, **kwargs):
        x = self._global_self_attention(inputs, inputs, inputs, *args, **kwargs)
        x = self._add([inputs, x])
        return self._layer_normalization(x, *args, **kwargs)


class FeedForwardModule(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feed_forward = Sequential([
            Dense(),
            ReLU(),
            Dense(),
            Dropout()
        ])
        self._add = Add()
        self._layer_normalization = LayerNormalization()

    def call(self, inputs, *args, **kwargs):
        x = self._feed_forward(inputs, *args, **kwargs)
        x = self._add([inputs, x])
        return self._layer_normalization(x, *args, **kwargs)
