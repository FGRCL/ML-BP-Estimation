import numpy
import tensorflow
from keras import Sequential
from keras.layers import Add, Dense, Dropout, Layer, LayerNormalization, MultiHeadAttention
from numpy import arange, concatenate
from tensorflow import TensorSpec, cast, float32

from mlbpestimation.models.basemodel import BloodPressureModel
from mlbpestimation.models.metricreducer.base import MetricReducer
from mlbpestimation.models.metricreducer.mutlistep import MultiStep


class Transformer(BloodPressureModel):

    def __init__(self,
                 n_layers: int,
                 n_attention_heads: int,
                 ff_units: int,
                 coder_dropout: float,
                 ff_dropout: float,
                 attention_dropout: float):
        super().__init__()
        self.n_layers = n_layers
        self.ff_units = ff_units
        self.n_attention_heads = n_attention_heads
        self.coder_dropout = coder_dropout
        self.ff_dropout = ff_dropout
        self.attention_dropout = attention_dropout

        self.encoder = None
        self.decoder = None
        self.regressor = None

        self.metric_reducer = MultiStep()

    def set_input(self, input_spec: TensorSpec):
        sequence_length = input_spec[0][0].shape[1]
        embedding_size = input_spec[0][0].shape[2]
        output_size = input_spec[0][1].shape[2]

        self.encoder = Encoder(
            sequence_length,
            embedding_size,
            self.n_layers,
            self.ff_units,
            self.n_attention_heads,
            self.coder_dropout,
            self.ff_dropout,
            self.attention_dropout)
        self.decoder = Decoder(
            sequence_length,
            output_size,
            self.n_layers,
            self.ff_units,
            self.n_attention_heads,
            self.coder_dropout,
            self.ff_dropout,
            self.attention_dropout)

    def set_output(self, output_spec: TensorSpec):
        output_size = output_spec.shape[2]

        self.regressor = Dense(output_size)

    def get_metric_reducer_strategy(self) -> MetricReducer:
        return self.metric_reducer

    def get_config(self):
        return {
            "n_layers": self.n_layers,
            "ff_units": self.ff_units,
            "n_attention_heads": self.n_attention_heads,
            "coder_dropout": self.coder_dropout,
            "ff_dropout": self.ff_dropout,
            "attention_dropout": self.attention_dropout

        }

    def call(self, inputs):
        context, x = inputs

        context = self.encoder(context)
        x = self.decoder(x, context)
        x = self.regressor(x)
        return x


class PositionalEncoder(Layer):
    def __init__(self, sequence_length: int, encoding_size: int):
        super().__init__()
        self.pe = self._get_positional_encodings(sequence_length, encoding_size)

    def call(self, inputs):
        x = inputs + self.pe[tensorflow.newaxis, :, :]
        return x

    @staticmethod
    def _get_positional_encodings(sequence_length, encoding_size):
        depth = encoding_size / 2

        positions = arange(sequence_length)[:, numpy.newaxis]
        depths = arange(depth)[numpy.newaxis, :] / depth

        angle_rads = positions * (1 / (10000 ** depths))

        encodings = concatenate([numpy.sin(angle_rads), numpy.cos(angle_rads)], axis=-1)
        encodings = cast(encodings, float32)
        return encodings


class Encoder(Layer):
    def __init__(self, sequence_length: int, embedding_size: int, n_encoders: int, ff_units: int, attention_heads: int, encoder_dropout: float,
                 ff_dropout: float,
                 attention_dropout: float):
        super().__init__()
        self.pe = PositionalEncoder(sequence_length, embedding_size)
        self.dropout = Dropout(encoder_dropout)
        self.encoders = Sequential([])

        for _ in range(n_encoders):
            self.encoders.add(
                EncoderLayer(
                    embedding_size,
                    ff_units,
                    attention_heads,
                    ff_dropout,
                    attention_dropout,
                )
            )

    def call(self, inputs):
        x = self.pe(inputs)
        x = self.dropout(x)
        x = self.encoders(x)
        return x


class Decoder(Layer):
    def __init__(self, sequence_length: int, embedding_size: int, n_decoders: int, ff_units: int, attention_heads: int, decoder_dropout: float,
                 ff_dropout: float,
                 attention_dropout: float):
        super().__init__()
        self.pe = PositionalEncoder(sequence_length, embedding_size)
        self.dropout = Dropout(decoder_dropout)
        self.decoders = []

        for _ in range(n_decoders):
            self.decoders.append(
                DecoderLayer(
                    embedding_size,
                    ff_units,
                    attention_heads,
                    ff_dropout,
                    attention_dropout
                )
            )

    def call(self, inputs, context):
        x = self.pe(inputs)
        x = self.dropout(x)

        for decoder in self.decoders:
            x = decoder(x, context)

        return x


class DecoderLayer(Layer):
    def __init__(self, embedding_size: int, ff_units: int, attention_heads: int, ff_dropout: float, attention_dropout: float):
        super().__init__()
        self.csa = CausalSelfAttention(attention_heads, embedding_size, attention_dropout)
        self.ca = CrossAttention(attention_heads, embedding_size, attention_dropout)
        self.ff = FeedForward(ff_units, embedding_size, ff_dropout)

    def call(self, inputs, context):
        x = self.csa(inputs)
        x = self.ca(x, context)
        x = self.ff(x)
        return x


class EncoderLayer(Layer):
    def __init__(self, embedding_size: int, ff_units: int, attention_heads: int, ff_dropout: float, attention_dropout: float):
        super().__init__()
        self.ff = FeedForward(ff_units, embedding_size, ff_dropout)
        self.gsa = GlobalSelfAttention(attention_heads, embedding_size, attention_dropout)

    def call(self, inputs):
        x = self.gsa(inputs)
        x = self.ff(x)
        return x


class CrossAttention(Layer):
    def __init__(self, heads: int, embedding_size: int, dropout: float):
        super().__init__()
        self.mha = MultiHeadAttention(heads, embedding_size, dropout=dropout)
        self.add = Add()
        self.ln = LayerNormalization()

    def call(self, inputs, context):
        x = self.mha(inputs, context, context)
        x = self.add([inputs, x])
        x = self.ln(x)
        return x


class GlobalSelfAttention(Layer):
    def __init__(self, heads: int, embedding_size: int, dropout: float):
        super().__init__()
        self.mha = MultiHeadAttention(heads, embedding_size, dropout=dropout)
        self.add = Add()
        self.ln = LayerNormalization()

    def call(self, inputs, *args, **kwargs):
        x = self.mha(inputs, inputs, inputs)
        x = self.add([inputs, x])
        x = self.ln(x)
        return x


class CausalSelfAttention(Layer):
    def __init__(self, heads: int, embedding_size: int, dropout: float):
        super().__init__()
        self.mha = MultiHeadAttention(heads, embedding_size, dropout=dropout)
        self.add = Add()
        self.ln = LayerNormalization()

    def call(self, inputs):
        x = self.mha(inputs, inputs, inputs, use_causal_mask=True)
        x = self.add([inputs, x])
        x = self.ln(x)
        return x


class FeedForward(Layer):
    def __init__(self, units: int, embedding_size: int, dropout: float):
        super().__init__()
        self.dense_layers = Sequential([
            Dense(units, activation='relu'),
            Dense(embedding_size),
            Dropout(dropout)
        ])
        self.add = Add()
        self.ln = LayerNormalization()

    def call(self, inputs, *args, **kwargs):
        x = self.dense_layers(inputs)
        x = self.add([inputs, x])
        x = self.ln(x)
        return x
