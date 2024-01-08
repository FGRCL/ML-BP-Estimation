import tensorflow
from keras import Sequential
from keras.activations import sigmoid
from keras.layers import Add, BatchNormalization, Concatenate, Conv1D, Dense, GlobalAveragePooling1D, Lambda, Layer, MaxPooling1D, Multiply, ReLU
from keras_core.src.ops import absolute, add, greater, less, multiply
from keras_nlp.src.backend import ops
from tensorflow import TensorSpec

from mlbpestimation.models.basemodel import BloodPressureModel
from mlbpestimation.models.metricreducer.base import MetricReducer
from mlbpestimation.models.metricreducer.singlestep import SingleStep


class Chen(BloodPressureModel):
    def __init__(self):
        super().__init__()
        self.metric_reducer = SingleStep()

        self.sequential = Sequential([
            ConvolutionBlock(20),
            ConvolutionBlock(40),
            RFSPAB(40),
            RFSPAB(80),
            RFSPAB(160),
            GlobalAveragePooling1D(),
            Dense(2)
        ])

    def call(self, inputs):
        return self.sequential(inputs)

    def set_input(self, input_spec: TensorSpec):
        pass

    def set_output(self, output_spec: TensorSpec):
        pass

    def get_metric_reducer_strategy(self) -> MetricReducer:
        return self.metric_reducer


class ConvolutionBlock(Layer):
    def __init__(self, filters):
        super().__init__()

        self.conv = Conv1D(filters, 9, padding="same")
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.max_pool = MaxPooling1D(4, strides=1)

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return self.max_pool(x)


class RFSPAB(Layer):
    def __init__(self, filters: int):
        super().__init__()
        self.skip = Conv1D(filters, 1, strides=2,
                           padding="same")  # assumed the skip connection uses a 1x1 conv to match the number of filters with the output of the RFSPAB block
        self.receptivefield = MultiScaleReceptiveField(filters)
        self.mixedattention = ParallelMixedDomainAttention(filters)
        self.add = Add()

    def call(self, inputs, *args, **kwargs):
        skip = self.skip(inputs)
        x = self.receptivefield(inputs)
        x = self.mixedattention(x)
        x = self.add([x, skip])
        return x


class MultiScaleReceptiveField(Layer):
    def __init__(self, filters):
        super().__init__()

        sizes = [1, 3, 5]
        self.convolutions = []
        self.dilations = []
        for kernel, dilation in zip(sizes, sizes[::-1]):
            self.convolutions.append(
                Conv1D(filters, kernel, padding="same")
            )
            self.dilations.append(
                Conv1D(filters, 3, dilation_rate=dilation, padding="same")
            )

        self.concatenate = Concatenate()
        self.conv = Conv1D(filters, 1, strides=2, padding="same")  # assumed the strides argument goes here

    def call(self, inputs, *args, **kwargs):
        x = inputs
        scale_output = []
        for convolution, dilation in zip(self.convolutions, self.dilations):
            x = convolution(x)
            x = dilation(x)
            scale_output.append(x)

        x = self.concatenate(scale_output)
        x = self.conv(x)

        return x


class ParallelMixedDomainAttention(Layer):
    def __init__(self, filters: int):
        super().__init__()
        self.absolute = Lambda(absolute)
        # self.averagepooling = GlobalAveragePooling1D(keepdims=True) # removed average pooling, the network won't work with it
        self.channelattention = ChannelAttention(filters, 1)  # assuming a ratio of 1
        self.spatialattention = SpatialAttention()
        self.add = Add()
        self.multiply = Multiply()
        self.softattention = SoftAttention()

    def call(self, inputs, *args, **kwargs):
        absolute_output = self.absolute(inputs)
        # average_pool = self.averagepooling(absolute_output)
        average_pool = absolute_output
        channel_attention = self.channelattention(average_pool)
        spatial_attention = self.spatialattention(average_pool)
        attention = self.add([channel_attention, spatial_attention])
        attention = self.multiply([attention, average_pool])
        soft_attention = self.softattention(absolute_output, attention)
        return soft_attention


class ChannelAttention(Layer):
    def __init__(self, units: int, ratio: float):
        super().__init__()
        self.averagepooling = GlobalChannelAveragePooling1D()
        self.maxpooling = GlobalChannelMaxPooling1D()
        self.mlp = Sequential([
            Dense(units / ratio),
            Dense(units)
        ])
        self.add = Add()
        self.sigmoid = Lambda(sigmoid)

    def call(self, inputs, *args, **kwargs):
        features = []
        for pooling in [self.averagepooling, self.maxpooling]:
            x = pooling(inputs)
            x = self.mlp(x)
            features.append(x)

        x = self.add(features)
        x = self.sigmoid(x)

        return x


class SpatialAttention(Layer):
    def __init__(self):
        super().__init__()
        self.averagepooling = GlobalSpatialAveragePooling1D()
        self.maxpooling = GlobalSpatialMaxPooling1D()
        self.concatenate = Concatenate()
        self.conv = Conv1D(1, 3, padding="same")
        self.sigmoid = Lambda(sigmoid)

    def call(self, inputs, *args, **kwargs):
        features = []
        for pooling in [self.averagepooling, self.maxpooling]:
            x = pooling(inputs)
            features.append(x)

        x = self.concatenate(features)
        x = self.conv(x)
        x = self.sigmoid(x)

        return x


class GlobalChannelAveragePooling1D(Layer):
    def call(self, inputs, *args, **kwargs):
        return ops.mean(inputs, axis=1, keepdims=True)


class GlobalSpatialAveragePooling1D(Layer):
    def call(self, inputs, *args, **kwargs):
        return ops.mean(inputs, axis=2, keepdims=True)


class GlobalChannelMaxPooling1D(Layer):
    def call(self, inputs, *args, **kwargs):
        return ops.max(inputs, axis=1, keepdims=True)


class GlobalSpatialMaxPooling1D(Layer):
    def call(self, inputs, *args, **kwargs):
        return ops.max(inputs, axis=2, keepdims=True)


class SoftAttention(Layer):
    def __init__(self):
        super().__init__()

    def call(self, feature_map, attention_map):
        above_mask = greater(feature_map, attention_map)
        above_updates = add(feature_map, -attention_map)
        above_values = multiply(above_updates, above_mask)

        under_mask = less(feature_map, -attention_map)
        under_updates = add(feature_map, attention_map)
        under_values = multiply(under_updates, under_mask)

        result = (feature_map - feature_map) + above_values + under_values

        tensorflow.print("new", above_mask, under_mask)
        return result
