from keras import Sequential
from keras.engine.base_layer import Layer
from keras.engine.input_layer import InputLayer
from keras.layers import Add, Dense, Dropout, Flatten, LayerNormalization, MultiHeadAttention, ReLU
from tensorflow import TensorSpec

from mlbpestimation.models.basemodel import BloodPressureModel
from mlbpestimation.models.metricreducer.base import MetricReducer
from mlbpestimation.models.metricreducer.singlestep import SingleStep


class TransformerEncoder(BloodPressureModel):
    def __init__(self,
                 n_encoder_modules: int,
                 n_attention_heads: int,
                 attention_dropout: float,
                 ff_encoder_units: int,
                 ff_encoder_dropout: float,
                 n_regressor_layers: int,
                 regressor_units: int,
                 output_units: int,
                 regressor_dropout: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_encoder_modules = n_encoder_modules
        self.n_attention_heads = n_attention_heads
        self.attention_dropout = attention_dropout
        self.ff_encoder_units = ff_encoder_units
        self.ff_encoder_dropout = ff_encoder_dropout
        self.n_regressor_layers = n_regressor_layers
        self.regressor_units = regressor_units
        self.output_units = output_units
        self.regressor_dropout = regressor_dropout

        self.metric_reducer = SingleStep()

        self._input_layer = None
        self._flatten = Flatten()
        self._encoders = Sequential()
        self._regressor = None

    def set_input(self, input_spec: TensorSpec):
        self._input_layer = InputLayer(input_spec[0].shape[1:])
        embedding_size = input_spec[0].shape[-1]
        for _ in range(self.n_encoder_modules):
            self._encoders.add(EncoderModule(self.n_attention_heads, embedding_size, self.attention_dropout, self.ff_encoder_units, self.ff_encoder_dropout))

    def set_output(self, output_spec: TensorSpec):
        output_units = output_spec.shape[1]
        self._regressor = Regressor(self.n_regressor_layers, self.regressor_units, output_units, self.regressor_dropout)

    def call(self, inputs, training=None, mask=None):
        x = self._input_layer(inputs, training, mask)
        x = self._encoders(x, training, mask)
        x = self._flatten(x)
        return self._regressor(x, training=training, mask=mask)

    def get_metric_reducer_strategy(self) -> MetricReducer:
        return self.metric_reducer

    def get_config(self):
        return {
            'n_encoder_modules': self.n_encoder_modules,
            'n_attention_heads': self.n_attention_heads,
            'attention_dropout': self.attention_dropout,
            'ff_encoder_units': self.ff_encoder_units,
            'ff_encoder_dropout': self.ff_encoder_dropout,
            'n_regressor_layers': self.n_regressor_layers,
            'regressor_units': self.regressor_units,
            'output_units': self.output_units,
            'regressor_dropout': self.regressor_dropout
        }


class Regressor(Layer):
    def __init__(self, n_layers: int, dense_units: int, output_size: int, dropout_rate: float, **kwargs):
        super().__init__(**kwargs)

        self._dense_layers = Sequential()
        for _ in range(n_layers):
            self._dense_layers.add(Dropout(dropout_rate))
            self._dense_layers.add(Dense(dense_units))
            self._dense_layers.add(ReLU())
        self._output = Dense(output_size)

    def call(self, inputs, *args, **kwargs):
        x = self._dense_layers(inputs, *args, **kwargs)
        return self._output(x)


class EncoderModule(Layer):
    def __init__(self, n_heads: int, d_embedding: int, attention_dropout: float, dense_units: int, ff_dropout: float, **kwargs):
        super().__init__(**kwargs)
        self._global_self_attention = GlobalSelfAttentionModule(n_heads, d_embedding, attention_dropout)
        self._feed_forward = FeedForwardModule(dense_units, d_embedding, ff_dropout)

    def call(self, inputs, *args, **kwargs):
        x = self._global_self_attention(inputs, *args, **kwargs)
        return self._feed_forward(x, *args, **kwargs)


class GlobalSelfAttentionModule(Layer):
    def __init__(self, n_heads: int, d_embedding: int, attention_dropout: float, **kwargs):
        super().__init__(**kwargs)
        self._global_self_attention = MultiHeadAttention(n_heads, d_embedding, dropout=attention_dropout)
        self._add = Add()
        self._layer_normalization = LayerNormalization()

    def call(self, inputs, *args, **kwargs):
        x = self._global_self_attention(inputs, inputs, inputs, *args, **kwargs)
        x = self._add([inputs, x])
        return self._layer_normalization(x, *args, **kwargs)


class FeedForwardModule(Layer):
    def __init__(self, dense_units: int, d_embedding: int, dropout_rate: float, **kwargs):
        super().__init__(**kwargs)
        self._feed_forward = Sequential([
            Dense(dense_units),
            ReLU(),
            Dense(d_embedding),
            Dropout(dropout_rate)
        ])
        self._add = Add()
        self._layer_normalization = LayerNormalization()

    def call(self, inputs, *args, **kwargs):
        x = self._feed_forward(inputs, *args, **kwargs)
        x = self._add([inputs, x])
        return self._layer_normalization(x, *args, **kwargs)
