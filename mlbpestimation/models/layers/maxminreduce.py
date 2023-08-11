from keras.engine.base_layer import Layer
from keras.layers import Flatten
from tensorflow import reduce_max, reduce_min, stack


class MaxMinReduce(Layer):
    def call(self, inputs, *args, **kwargs):
        x = Flatten()(inputs)
        max = reduce_max(x, axis=-1)
        min = reduce_min(x, axis=-1)
        output = stack((max, min), 1)
        return output
